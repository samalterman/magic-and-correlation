import argparse
import json
import numpy as np
import time
import sys
from sys import exit
import pickle
import os

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

from symmer import PauliwordOp
from symmer.operators import QuantumState
from symmer.utils import exact_gs_energy
from symmer.projection import QubitTapering, ContextualSubspace
from joblib import Parallel, delayed, parallel_config


def _fwht(a):
    """Fast Walsh-Hadamard Transform using numpy vectorization.
    Computes f[z] = sum_k a[k] * (-1)^{popcount(k & z)} in O(d log d)."""
    result = np.array(a, dtype=complex)
    h = 1
    while h < len(result):
        result = result.reshape(-1, 2 * h)
        left = result[:, :h].copy()
        right = result[:, h:].copy()
        result[:, :h] = left + right
        result[:, h:] = left - right
        result = result.ravel()
        h *= 2
    return result

def _fwht_batched_gpu(A_gpu):
    """Batched Fast Walsh-Hadamard Transform on GPU using CuPy.
    A_gpu has shape (batch, d). Transforms each row independently."""
    import cupy as cp
    batch, d = A_gpu.shape
    h = 1
    while h < d:
        A_gpu = A_gpu.reshape(batch, -1, 2 * h)
        left = A_gpu[:, :, :h].copy()
        right = A_gpu[:, :, h:].copy()
        A_gpu[:, :, :h] = left + right
        A_gpu[:, :, h:] = left - right
        A_gpu = A_gpu.reshape(batch, -1)
        h *= 2
    return A_gpu

def _build_a_vectors(X_symps_batch, c_states, coeff_vec, conj_vec, d):
    """Build the signal vectors a[s1] = c_{s1} * conj(c_{s1 XOR x}) for a batch of x_symps."""
    A = np.zeros((len(X_symps_batch), d), dtype=np.complex128)
    for j, x_symp in enumerate(X_symps_batch):
        A[j, c_states] = coeff_vec[c_states]*conj_vec[np.bitwise_xor(x_symp,c_states)]
    return A

def stab_entropy_symp(state, order : float = 2, filtered : bool = False, parallel = False, n_proc : int = 4, gpu : bool = False, gpu_device : int = None) -> float:
    """Calculates the exact stabilizer Renyi entropy of the given state by being cheeky in the symplectic representation.
    Args:
        state (QuantumState): the state to calculate the stabilizer entropy for
        order (int): the order of the stabilizer entropy to calculate. default is 2
        filtered (bool): whether to calculate the filtered stabilizer entropy instead of the unfilitered stabilizer entropy. See arXiv:2312.11631 for details. default is False.
        parallel (bool or str, optional): False for serial, True or 'joblib' for joblib multiprocessing,
            'mpi' for MPI distributed computing. Default is False.
        n_proc (int, optional): number of joblib processes. Ignored for MPI. Default is 4.
        gpu (bool, optional): whether to use GPU acceleration via CuPy. Runs batched FWHT on GPU.
            Requires CuPy to be installed and a CUDA-capable GPU available.
            Can be combined with parallel='mpi' for multi-node multi-GPU. Default is False.
        gpu_device (int, optional): which GPU device to use. If None, uses the device set by
            CUDA_VISIBLE_DEVICES or defaults to device 0. For MPI+GPU, auto-assigned from
            local rank if None. Default is None.
    Returns:
        Mq (float): the calculated stabilizer entropy """

    # Normalize parallel parameter for backward compatibility
    if parallel is True:
        _mode = 'joblib'
    elif parallel is False or parallel is None:
        _mode = None
    else:
        _mode = str(parallel).lower()
        if _mode not in ('joblib', 'mpi'):
            raise ValueError(f"Unrecognised parallel mode '{parallel}'. Use False, True, 'joblib', or 'mpi'.")

    # For MPI+GPU, auto-assign GPU device from local MPI rank
    if gpu and _mode == 'mpi' and gpu_device is None:
        gpu_device = _get_mpi_local_rank()

    if gpu:
        try:
            import cupy as cp
        except ImportError:
            raise ImportError("GPU mode requires CuPy. Install with: pip install cupy-cuda12x (or the appropriate CUDA version)")
        try:
            if gpu_device is not None:
                cp.cuda.Device(gpu_device).use()
            dev=cp.cuda.Device()
            dev.compute_capability
        except cp.cuda.runtime.CUDARuntimeError:
            raise RuntimeError(
                f"No CUDA GPU available (are you on a login node?). "
                f"Submit to a GPU node or set CUDA_VISIBLE_DEVICES."
            )
    t1=time.perf_counter()
    n_qubits=state.n_qubits
    d=2**n_qubits
    # build integer-keyed coefficient dict for O(1) lookup without string formatting
    state_dict=state.to_dictionary
    coeff_dict={int(key,2): val for key, val in state_dict.items()}
    c_states=np.unique(np.asarray(list(coeff_dict.keys()),dtype=np.int64))
    coeff_vec=state.to_dense_matrix.flatten()
    conj_vec=np.conj(coeff_vec)    
    t2=time.perf_counter()
#    print(f'Setup time: {round(t2-t1,6)}s')
    # generate all the X symplectic vectors that could give non-zero contributions
    # sorted() ensures deterministic ordering across MPI ranks
    X_symps=np.array(sorted({s1^s2 for s1 in c_states for s2 in c_states}))
    t3=time.perf_counter()
#    print(f'X symps generation time: {round(t3-t2,6)}s')

    if _mode == 'mpi':
        zeta=_symp_mpi(X_symps, c_states, coeff_vec, conj_vec, d, order, gpu=gpu)
    elif gpu:
        zeta=_symp_gpu(cp, X_symps, c_states, coeff_vec, conj_vec, d, order)
    elif _mode == 'joblib':
        zeta=_symp_joblib(X_symps,c_states,coeff_vec,conj_vec,d,order,n_proc)
    else:
        def _zeta_for_x(x_symp):
            a=np.zeros(d, dtype=complex)
            a[c_states]=coeff_vec[c_states]*conj_vec[np.bitwise_xor(x_symp,c_states)]
            f=_fwht(a)
            return np.sum(np.abs(f)**(2*order))/d
        zeta=sum(_zeta_for_x(x) for x in X_symps)
    t4=time.perf_counter()
#    print(f'Zeta calculation time: {round(t4-t3,6)}s')
    if filtered:
        zeta=(zeta-1/d)*d/(d-1)
    Mq=-np.log2(zeta)/(order-1)
    return Mq

def _symp_gpu(cp, X_symps, c_states, coeff_vec, conj_vec, d, order):
    """GPU-accelerated zeta computation. Batches X_symps to fit in GPU memory."""
    num_x=len(X_symps)
    # auto-determine batch size from available GPU memory
    free_mem=cp.cuda.Device().mem_info[0]
    bytes_per_row=d * 16  # complex128
    # use at most 40% of free GPU memory for the working batch (need room for copies in FWHT)
    batch_size=max(1, int(free_mem * 0.4 / (bytes_per_row * 3)))
    batch_size=min(batch_size, num_x)

    zeta=0.0
    for i in range(0, num_x, batch_size):
        batch_x=X_symps[i:i+batch_size]
        # build a-vectors on CPU
        A=_build_a_vectors(batch_x, c_states, coeff_vec, conj_vec, d)
        # transfer to GPU, run batched FWHT, compute contribution
        A_gpu=cp.asarray(A)
        F_gpu=_fwht_batched_gpu(A_gpu)
        zeta+=float(cp.sum(cp.abs(F_gpu)**(2*order)).get())/d
        # free GPU memory between batches
        del A_gpu, F_gpu
        cp.get_default_memory_pool().free_all_blocks()

    return zeta

def _get_mpi_local_rank():
    """Detect MPI local rank from environment variables set by common MPI launchers.
    Used to auto-assign GPU devices in multi-node MPI+GPU runs."""
    import os
    for var in ('OMPI_COMM_WORLD_LOCAL_RANK',   # OpenMPI
                'MV2_COMM_WORLD_LOCAL_RANK',    # MVAPICH2
                'MPI_LOCALRANKID',              # Intel MPI
                'SLURM_LOCALID',                # SLURM
                'LOCAL_RANK'):                   # PyTorch-style / generic
        val = os.environ.get(var)
        if val is not None:
            return int(val)
    # Fallback: use global rank (correct when running on a single node)
    from mpi4py import MPI
    return MPI.COMM_WORLD.Get_rank()

def _symp_mpi(X_symps, c_states, coeff_vec,conj_vec, d, order, gpu=False):
    """MPI-distributed zeta computation. Distributes X_symps across MPI ranks,
    computes partial zeta on each rank (CPU or GPU), and reduces via MPI.SUM.

    All ranks must call this function with the same state data.
    Returns the total zeta on all ranks (broadcast from root)."""
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Distribute X_symps evenly across ranks
    n = len(X_symps)
    chunk = n // size
    remainder = n % size
    if rank < remainder:
        start = rank * (chunk + 1)
        end = start + chunk + 1
    else:
        start = remainder * (chunk + 1) + (rank - remainder) * chunk
        end = start + chunk
    my_X_symps = X_symps[start:end]

    # Compute local zeta contribution
    if len(my_X_symps) == 0:
        local_zeta = 0.0
    elif gpu:
        import cupy as cp
        local_zeta = _symp_gpu(cp, my_X_symps, c_states, coeff_vec, conj_vec, d, order)
    else:
        def _zeta_for_x(x_symp):
            a = np.zeros(d, dtype=complex)
            a[c_states]=coeff_vec[c_states]*conj_vec[np.bitwise_xor(x_symp,c_states)]
            f = _fwht(a)
            return np.sum(np.abs(f)**(2*order)) / d
        local_zeta = sum(_zeta_for_x(x) for x in my_X_symps)

    # Sum partial zeta across all ranks
    total_zeta = comm.reduce(local_zeta, op=MPI.SUM, root=0)
    # Broadcast so all ranks return the same result
    zeta = comm.bcast(total_zeta, root=0)
    return zeta

def _symp_joblib(X_symps, c_states, coeff_vec,conj_vec, d, order,n_proc):
    """Parallel zeta computation designed for use on laptops using joblib"""
    def _zeta_for_x(x_symp):
        a=np.zeros(d, dtype=complex)
        a[c_states]=coeff_vec[c_states]*conj_vec[np.bitwise_xor(x_symp,c_states)]
        f=_fwht(a)
        return np.sum(np.abs(f)**(2*order))/d
    batches=max(int(len(X_symps) / n_proc), len(X_symps) // (n_proc * 10))
    with parallel_config(backend='loky'):
        zeta_vals=Parallel(n_jobs=n_proc,batch_size=batches)(delayed(_zeta_for_x)(x) for x in X_symps)
    zeta=sum(zeta_vals)
    return zeta


def make_hams(filepath,cspath):
    """Make Hamiltonians for a given Symmer Hamiltonian"""
    filename = filepath.split('/')[-1]
    with open(filepath) as f:
        data_dict=json.load(f)
    print(f'Loaded {filename}')
    fci_energy = data_dict['data']['calculated_properties']['FCI']['energy']
    hf_state   = QuantumState(np.asarray(data_dict['data']['hf_array'])) # Hartree-Fock state
    hf_energy  = data_dict['data']['calculated_properties']['HF']['energy']
    H = PauliwordOp.from_dictionary(data_dict['hamiltonian'])
    UCC_q = PauliwordOp.from_dictionary(data_dict['data']['auxiliary_operators']["UCCSD_operator"])
    N_op = PauliwordOp.from_dictionary(data_dict["data"]["auxiliary_operators"]["number_operator"])

    # Print the extracted information
    print(f"Calculated Hartree-Fock Energy: {hf_energy}")
    print(f"Calculated FCI energy: {fci_energy}")

    # Do qubit tapering
    QT = QubitTapering(H)
    print(f'Qubit tapering permits a reduction of {H.n_qubits} -> {H.n_qubits-QT.n_taper} qubits.\n')
    print('The following symmetry generators were identified:\n')
    print(QT.symmetry_generators); print()
    print('which we may rotate onto the single-qubit Pauli operators\n') 
    print(QT.symmetry_generators.rotate_onto_single_qubit_paulis()); print()
    print('via a sequence of Clifford operations R_k = e^{i pi/4 P_k} where:\n')
    for index, (P_k, angle) in enumerate(QT.symmetry_generators.stabilizer_rotations):
        P_k.sigfig=0
        print(f'P_{index} = {P_k}')
    H_taper   = QT.taper_it(ref_state=hf_state)
    UCC_taper = QT.taper_it(aux_operator=UCC_q)
    hf_tap    = QT.tapered_ref_state
    N_op_tap=QT.taper_it(aux_operator=N_op)
    method='LCU'

    # Build the CS-VQE model
    try:
        cs_vqe = ContextualSubspace(H_taper, noncontextual_strategy='StabilizeFirst', unitary_partitioning_method='LCU',reference_state=hf_tap)
    except:
        print("LCU failed, attempting seq_rot")
        try:
            cs_vqe = ContextualSubspace(H_taper, noncontextual_strategy='StabilizeFirst', unitary_partitioning_method='seq_rot',reference_state=hf_tap)
            method='seq_rot'
        except:
            #kill all processes 
            print("Unitary partitioning failed under both methods")
            comm.Abort(1) 
            


    # Now we project into the contextual subspace
    full_qubits=H_taper.n_qubits
    newpath=os.path.join(cspath,filename[:-5]+"_CS")
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    for i in range(1, full_qubits+1, 1):
        cs_vqe.update_stabilizers(n_qubits = i, strategy='aux_preserving', aux_operator=UCC_taper)
        contextual_terms=cs_vqe.contextual_operator.n_terms
        noncon_terms=cs_vqe.noncontextual_operator.n_terms
        n_cliques=cs_vqe.noncontextual_operator.n_cliques
        H_cs = cs_vqe.project_onto_subspace()
        N_cs=cs_vqe.project_onto_subspace(N_op_tap)

        hf_proj=cs_vqe.project_state(hf_tap).to_dense_matrix

        out_filename=filename[:-5]+'_CS_'+str(i)+'.json'
        with open(os.path.join(newpath,out_filename), "w+") as file:
            ham_dict=H_cs.to_dictionary
            for k,v in ham_dict.items():
                ham_dict[k]=np.real(v)
            N_dict=N_cs.to_dictionary
            for k,v in N_dict.items():
                N_dict[k]=np.real(v)

            data={
                "hamiltonian": ham_dict,
                "hf_vec":hf_proj.real.tolist(),
                "N_op": N_dict,
                'con_terms':contextual_terms,
                'noncon_terms':noncon_terms,
                'n_cliques':n_cliques,
                "data": data_dict['data']
            }
            json.dump(data,file) 
        print(f'{i} qubit CS Hamiltonian saved')
    print(f'{full_qubits} qubit CS Hamiltonian saved')
    print('All CS Hamiltonians saved!')
    return newpath

def load_hamiltonian(filepath):
    """Load Hamiltonian from JSON file.

    Expected format:
        {
            "hamiltonian": {"PAULI_STRING": real, ...},
            "data": {"n_qubits": ..., "calculated_properties": {...}, ...}
        }

    Returns:
        H_sparse: scipy sparse matrix of the Hamiltonian
        n_qubits: number of qubits
        hf_proj: the HF reference in that contextual subspace
        N_op: the number operator in that contextual subspace
        metadata: dict of molecular data
        full_qubits:
    """
    with open(filepath) as f:
        data = json.load(f)
    print("Data loaded")
    metadata = data.get('data', {})
    
    #pauli_coeff_dict = {k: v for k, v in ham_dict.items()}
    #H_op = PauliwordOp.from_dictionary(pauli_coeff_dict)
    H_op = PauliwordOp.from_dictionary(data['hamiltonian'])
    print("Hamiltonian PauliwordOp constructed")
    H_sparse = H_op.to_sparse_matrix
    print("Hamiltonian sparse matrix representation constructed")
    N_op=PauliwordOp.from_dictionary(data['N_op'])
    print("Number operator PauliwordOp constructed")
    hf_proj=QuantumState.from_array(np.array(data['hf_vec']))
    print("HF QuantumState created")
    n_qubits = H_op.n_qubits
    return H_sparse, n_qubits,hf_proj,N_op, metadata


def eigvec_to_quantumstate(eigvec, n_qubits, threshold=1e-12):
    """Convert dense eigenvector to QuantumState.

    Filters out amplitudes below threshold so the SRE computation
    only works with the non-negligible part of the state.
    """
    nonzero_idx = np.where(np.abs(eigvec) > threshold)[0]
    state_dict = {
        format(idx, f'0{n_qubits}b'): complex(eigvec[idx])
        for idx in nonzero_idx
    }
    return QuantumState.from_dictionary(state_dict)


def diag_scipy(H_sparse,hf_proj,N_proj,N_target, k=4,tol=1e-9):
    """Find k lowest eigenpairs using scipy Lanczos (single-node)."""
    M_N = N_proj.to_sparse_matrix
    hf_vec = hf_proj.normalize.to_sparse_matrix.toarray().flatten()

    if H_sparse.shape[0]-1<=k:
        from numpy.linalg import eigh
        eigenvalues, eigenvectors = eigh(H_sparse.toarray())
        idx = np.argsort(eigenvalues)
        eigenvalues,eigenvectors=eigenvalues[idx],eigenvectors[:,idx]
        for i in range(len(eigenvalues)):
            n_exp = (eigenvectors[:,i].conj() @(M_N @ eigenvectors[:,i])).real
            if abs(n_exp-N_target) < 1e-3:
                return np.array([float(eigenvalues[i].real)]),eigenvectors[:,i:i+1]
        raise RuntimeError(f"No N={N_target} eigenvector found in full diagonalization!")
    else:
        from scipy.sparse.linalg import eigsh
        for k_try in (k, 2*k, 4*k):
            if k_try >= H_sparse.shape[0]:
                break
            vals, vecs = eigsh(H_sparse, k=k_try, which="SA", v0=hf_vec, tol=tol)
            for i in np.argsort(vals.real):
                n_exp = (vecs[:, i].conj() @ (M_N @ vecs[:, i])).real
                if abs(n_exp - N_target) < 1e-3:
                    return np.array([float(vals[i].real)]), vecs[:, i:i+1]
        raise RuntimeError(f"No N={N_target} eigenvector found within k={k_try}")


def main():
    parser = argparse.ArgumentParser(
        description='Hamiltonian JSON -> Ground State SRE Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  mpirun -n 4  python sre_pipeline.py ../hamiltonian_18_or_less/H2O_STO-3G_SINGLET_JW.json
  mpirun -n 16 python sre_pipeline.py big_hamiltonian.json --solver slepc --k 8
  srun python sre_pipeline.py hamiltonian.json --output results.txt
        """)
    parser.add_argument('hamiltonian', help='Hamiltonian JSON file')
    parser.add_argument('--solver', choices=['scipy', 'slepc'], default='scipy',
                        help='Eigensolver: scipy (single-node Lanczos) or slepc '
                             '(MPI-distributed). Default: scipy')
    parser.add_argument('--k', type=int, default=5,
                        help='Number of lowest eigenstates to compute. '
                             'Extra states help verify ground state stability. Default: 5')
    parser.add_argument('--order', type=int, default=2,
                        help='SRE Renyi order. Default: 2')
    parser.add_argument('--filtered', action='store_true',
                        help='Compute filtered SRE (excludes identity)')
    parser.add_argument('--output', default=None,
                        help='Save results to this folder')
    parser.add_argument('--threshold', type=float, default=1e-15,
                        help='Amplitude threshold for eigenvector coefficients. '
                             'Coefficients below this are zeroed and the state is '
                             'renormalized. Reduces SRE computation time for dense '
                             'eigenvectors (e.g. from SLEPc). Default: 1e-15')

    args = parser.parse_args()

    # Load Hamiltonian
    if rank == 0:
        print(f"Making Hamiltonians for {args.hamiltonian}")
        try:
            folder=make_hams(args.hamiltonian,'/cluster/tufts/lovelab/salter02/magiccluster/cs_data/')
        except Exception as e:
            print(f"Error: {e}", flush=True)
            comm.Abort(1) 
        print(f"Loading: {folder}")
    else:
        folder=None
    folder=comm.bcast(folder,root=0)
    cs_size_list = []
    tap_nrg_true = []
    gs_sre = []
    overlaps=[]
    unsorted=os.listdir(folder)
    def sortfunc(file):
        return int(file[:-5].split('_')[-1])
    sort_list=sorted(unsorted,key=sortfunc)

    for ham_file in sort_list:
        if rank == 0:
            # Only rank 0 should load the file!
            print(f"Loading: {ham_file}")
            H_sparse, n_qubits,hf_proj,N_op, metadata = load_hamiltonian(os.path.join(folder,ham_file))
            n_particles=metadata["n_particles"]["total"]
            print(f"  n_qubits  = {n_qubits}")
            print(f"  dim       = {2**n_qubits}")
            print(f"  nnz       = {H_sparse.nnz}")
            if metadata:
                print(f"  basis     = {metadata.get('basis', 'N/A')}")
            print()
            import sys
            import pickle

            serialized_size = len(pickle.dumps(H_sparse))
            print(f"Size of H_sparse: {serialized_size / (1024**3):.2f} GB")
        else:
            H_sparse = None
            n_qubits = None
            n_particles=None
            metadata = None

        n_qubits = comm.bcast(n_qubits, root=0)
        metadata = comm.bcast(metadata, root=0)
        n_particles=comm.bcast(n_particles,root=0)

        #  Diagonalize 
        if rank == 0:
            print(f"Diagonalizing ({args.solver}, k={args.k}) ...")

        t0 = time.perf_counter()

        if args.solver == 'scipy':
            # Only rank 0 diagonalizes, then broadcasts results
            if rank == 0:
                eigenvalues, eigenvectors = diag_scipy(H_sparse,hf_proj=hf_proj,N_proj=N_op,N_target=n_particles,k=args.k)
            else:
                eigenvalues = None
                eigenvectors = None
            eigenvalues = comm.bcast(eigenvalues, root=0)
            eigenvectors = comm.bcast(eigenvectors, root=0)

        t_diag = time.perf_counter() - t0
        terminate = False
        if rank == 0:
            print(f"  Time: {t_diag:.2f}s\n")
            print(f"  Eigenvalues:")
            for i, E in enumerate(eigenvalues):
                gap = f"  (gap = {E - eigenvalues[0]:.8f})" if i > 0 else ""
                marker = " <-- ground state" if i == 0 else ""
                print(f"    E_{i} = {E:18.10f}{gap}{marker}")

            # Compare with reference energies if available
            ref = metadata.get('calculated_properties', {})
            if ref:
                print(f"\n  Reference energies:")
                for method in ['HF', 'MP2', 'CCSD', 'FCI']:
                    if method in ref and ref[method]['converged']:
                        print(f"    {method:6s} = {ref[method]['energy']:18.10f}")
                if 'FCI' in ref:
                    fci_en=ref['FCI']['energy']
                if 'HF' in ref:
                    hf_en=ref['HF']['energy']
                    corr = abs(eigenvalues[0] - hf_en)
                    print(f"    |E0 - HF| = {corr:.2e}")
                # Check for Symmer failures
                    if eigenvalues[0]-hf_en>= 1e-8:
                        print(f"Symmer failure detected!")
                        print(f"E0-HF = {eigenvalues[0] - hf_en:.2e}>0")
                        print(f"Terminating evaluation")
                        terminate = True
                
            # Degeneracy warning
            if len(eigenvalues) > 1:
                gap = eigenvalues[1] - eigenvalues[0]
                if gap < 1e-6:
                    print(f"\n  WARNING: near-degenerate ground state! "
                        f"gap = {gap:.2e}")
                    print(f"  The ground state eigenvector may be unreliable.")
            print()
        terminate=comm.bcast(terminate, root=0)
        if terminate:
            exit('Symmer error')
        # Build ground state QuantumState
        if rank == 0:
            gs_vec = eigenvectors[:, 0]

            # Truncate small coefficients and renormalize
            raw_nonzero = np.count_nonzero(np.abs(gs_vec) > 1e-15)
            if n_qubits>2:
                mask = np.abs(gs_vec) > args.threshold
                discarded_weight = np.sum(np.abs(gs_vec[~mask])**2)
                gs_vec[~mask] = 0.0
                norm = np.linalg.norm(gs_vec)
                if norm > 0:
                    gs_vec /= norm
                print(f"Coefficient thresholding (threshold={args.threshold:.1e}):")
                print(f"  Before: {raw_nonzero} non-zero amplitudes")
                print(f"  After:  {np.count_nonzero(mask)} non-zero amplitudes")
                print(f"  Discarded weight: {discarded_weight:.2e}")

            ground_state = eigvec_to_quantumstate(gs_vec, n_qubits, args.threshold)
            n_nonzero = len(ground_state.to_dictionary)
            print(f"Ground state: {n_nonzero} non-zero amplitudes "
                f"({100 * n_nonzero / 2**n_qubits:.1f}% of {2**n_qubits})")
        else:
            ground_state = None
        ground_state = comm.bcast(ground_state, root=0)

        #  Compute SRE with MPI
        if rank == 0:
            print(f"Computing SRE (order={args.order}, {size} MPI ranks) ...")

        comm.Barrier()
        t0 = time.perf_counter()
        sre=stab_entropy_symp(
            ground_state,
            order=args.order,
            filtered=args.filtered,
            parallel='mpi'
        )

        if rank==0:
            print(f'Calculating HF-CS overlap')
            overlap=np.abs(ground_state.dagger*hf_proj)**2
            overlaps.append(overlap)

        t_sre = time.perf_counter() - t0

        #  Output
        if rank == 0:
            cs_size_list.append(n_qubits)
            tap_nrg_true.append(eigenvalues[0])
            gs_sre.append(sre)
            print(f"  Time: {t_sre:.2f}s\n")
            print("=" * 50)
            print(f"  CS Qubits:            {n_qubits}")
            print(f"  Ground state energy:  {eigenvalues[0]:.10f}")
            print(f"  Correlation energy:   {corr:.10f}")
            print(f"  SRE (order {args.order}):        {sre:.10f}")
            if len(eigenvalues) > 1:
                print(f"  Energy gap (E1-E0):   {eigenvalues[1] - eigenvalues[0]:.10f}")
            print(f"  Diag time:            {t_diag:.2f}s")
            print(f"  SRE time:             {t_sre:.2f}s")
            print(f"  MPI ranks:            {size}")
            print("=" * 50)
            print()

    if rank==0:
        print("Scan completed!")
        if args.output:
            if not os.path.isdir(args.output):
                os.makedirs(args.output)
            filename = args.hamiltonian.split('/')[-1]
            outfile=os.path.join(args.output,"data_"+filename)
            data = {
                    'n_qubits' : n_qubits,
                    'qubits'    : cs_size_list,
                    'fci_energy' : fci_en,
                    'hf_energy' : metadata['calculated_properties']['HF']['energy'],
                    "CS_SRE" : gs_sre,
                    "CS_energy": tap_nrg_true,
                    'overlaps': overlaps,
                    "solver": args.solver,
                    'data' : metadata
                }
            with open(outfile, "w+") as file:
                json.dump(data,file) 
            
if __name__ == '__main__':
    try:
        main()
    except Exception as e:
            print(f"Error: {e}", flush=True)
            comm.Abort(1) 
        
