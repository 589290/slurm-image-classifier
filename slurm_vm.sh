#!/bin/bash

#SBATCH --workdir=/slurm_storage/miernickig/sic
#SBATCH --output=/slurm_storage/miernickig/sic/slurm-%j.out
#SBATCH --error=/slurm_storage/miernickig/sic/slurm-%j.err
#SBATCH --job-name=sic
#SBATCH --cpus-per-task=4

module cuda/8.0
module python/3.6
env
nvidia-smi 

python check_images.py --dir pet_images/ --arch resnet  --dogfile dognames.txt 1> /slurm_storage/miernickig/sic/2a_out 2> /slurm_storage/miernickig/sic/2a_err
python check_images.py --dir pet_images/ --arch alexnet --dogfile dognames.txt 1> /slurm_storage/miernickig/sic/2b_out 2> /slurm_storage/miernickig/sic/2b_err
python check_images.py --dir pet_images/ --arch vgg  --dogfile dognames.txt 1> /slurm_storage/miernickig/sic/2c_out 2> /slurm_storage/miernickig/sic/2c_err