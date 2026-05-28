#!/bin/bash
#SBATCH -p batch
#SBATCH --job-name B2H2_singlet_Dooh_magic_analysis
#SBATCH -n 25
#SBATCH -t 2-00:00:00
#SBATCH --mem=250g
#SBATCH -o outputs/B2H2_singlet_Dooh_%A_%a_output.out
#SBATCH -e outputs/B2H2_singlet_Dooh_%A_%a_error.out            
#SBATCH --array=1-10

# Load environment
module purge
module load anaconda/2025.06.0
module load openmpi/5.0.7-cuda
module load cuda/12.9.0
source activate /cluster/tufts/lovelab/salter02/condaenv/magicsymmerenv

files=(/cluster/tufts/lovelab/salter02/magiccluster/hamiltonians/B2H2_singlet_Dooh/*.json)
fq=${files[$SLURM_ARRAY_TASK_ID-1]}
mpirun -n 25 python /cluster/tufts/lovelab/salter02/magiccluster/sre_pipeline.py "$fq" --k 1 --solver "scipy" --output "/cluster/tufts/lovelab/salter02/magiccluster/final_data/B2H2_singlet_Dooh"
conda deactivate
    