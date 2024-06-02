#!/bin/bash
#SBATCH --account MRI-INFARCT-SEGMENTATION
#SBATCH --job-name=UNet_C3_JAG
#SBATCH -c 2
#SBATCH --mem 200g
#SBATCH --time=3-23:00:00
#SBATCH --partition gpu
#SBATCH --gres=gpu:2
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out
#SBATCH --error=/home/rosengaard/mri-infarct-segmentation/Anders/V3/Unet_C3/JAG/Slurm/jobs/job.%J.err
#SBATCH --output=/home/rosengaard/mri-infarct-segmentation/Anders/V3/Unet_C3/JAG/Slurm/jobs/job.%J.out

# Run the job​
/home/rosengaard/mri-infarct-segmentation/Anders/V3/Unet_C3/JAG/Slurm/trainUnet_C3.py
