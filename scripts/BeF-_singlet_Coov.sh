#!/bin/bash
#SBATCH -p batch
#SBATCH --job-name BeF-_singlet_Coov_magic_analysis
#SBATCH -n 25
#SBATCH -t 2-00:00:00
#SBATCH --mem=250g
#SBATCH -o outputs/BeF-_singlet_Coov_%A_%a_output.out
#SBATCH -e outputs/BeF-_singlet_Coov_%A_%a_error.out            
#SBATCH --array=1-10

# Load environment
module purge
module load anaconda/2025.06.0
module load openmpi/5.0.7-cuda
module load cuda/12.9.0
source activate /cluster/tufts/lovelab/salter02/condaenv/magicsymmerenv

files=(/cluster/tufts/lovelab/salter02/magiccluster/hamiltonians/BeF-_singlet_Coov/*.json)
fq=${files[$SLURM_ARRAY_TASK_ID-1]}
mpirun -n 25 python /cluster/tufts/lovelab/salter02/magiccluster/sre_pipeline.py "$fq" --k 1 --solver "scipy" --output "/cluster/tufts/lovelab/salter02/magiccluster/final_data/BeF-_singlet_Coov"
conda deactivate
    