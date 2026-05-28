#!/bin/bash
#SBATCH -p batch
#SBATCH --job-name H+_singlet_Kh_magic_analysis
#SBATCH -n 25
#SBATCH -t 2-00:00:00
#SBATCH --mem=250g
#SBATCH -o outputs/H+_singlet_Kh_%A_%a_output.out
#SBATCH -e outputs/H+_singlet_Kh_%A_%a_error.out            
#SBATCH --array=1-5

# Load environment
module purge
module load anaconda/2025.06.0
module load openmpi/5.0.7-cuda
module load cuda/12.9.0
source activate /cluster/tufts/lovelab/salter02/condaenv/magicsymmerenv

files=(/cluster/tufts/lovelab/salter02/magiccluster/hamiltonians/H+_singlet_Kh/*.json)
fq=${files[$SLURM_ARRAY_TASK_ID-1]}
mpirun -n 25 python /cluster/tufts/lovelab/salter02/magiccluster/sre_pipeline.py "$fq" --k 1 --solver "scipy" --output "/cluster/tufts/lovelab/salter02/magiccluster/final_data/H+_singlet_Kh"
conda deactivate
    