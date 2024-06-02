#!/bin/bash
#SBATCH --account MRI-INFARCT-SEGMENTATION
#SBATCH -c 2
#SBATCH --mem 25g
#SBATCH --time=00:40:00
#SBATCH --partition gpu
#SBATCH --gres=gpu:1
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out

# Run the job​
/home/rosengaard/mri-infarct-segmentation/Anders/V3/Unet_C3/JAG/Slurm/FindLr/FindLr_Unet_C3.py
