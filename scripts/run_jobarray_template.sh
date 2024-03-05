#!/bin/bash 
#SBATCH --job-name=JOBNAME
#SBATCH -t HOURS:MINUTES:00
#SBATCH --array=START-END
#SBATCH -c 1
#SBATCH --mem=2G
#SBATCH -e ./logs/error.%A-%a.out 
#SBATCH -o ./logs/output.%A-%a.out


module load Python/3.10.8-GCCcore-12.2.0

source ${HOME}/EA_Drone_Delivery/ea_drone_delivery/bin/activate

python ${1} ${2}_${SLURM_ARRAY_TASK_ID}.json