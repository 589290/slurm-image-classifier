# Slurm Image Classifier
SIC is a benchmarking tool that measures several different CNNs (Convulutional Neural Networks) against once another. The CNNs (AlexNet, VGG, and ResNet) are available from ImageNet. The purpose of this project is to optimize the efficiency of the tests on varying slurm configurations.

## data from >>
https://people.eecs.berkeley.edu/~kanazawa/

## benchmarks
|system|resnet|alexnet|vgg|
|---|---|---|---|
|SLURM (via ./run_models_batch.sh)|15.31 seconds|6.24 seconds|87.09 seconds|
|SLURM (via sbatch slurm_vm.sh)|5.66 seconds|3.37 seconds|29.18 seconds|
|SLURM (via sbatch slurm_dgx1.sh)|10.49 seconds|2.95 seconds|31.70 seconds|
|MacPro (via ./run_models_batch.sh)|4.73 seconds|2.28 seconds|18.35 seconds|

### dgx1 specifics
SBATCH --job-name=sic  
SBATCH --partition=dgx1  
SBATCH --cpus-per-task=4  
SBATCH --gres=gpu:4  
module cuda/8.0  
module python/3.6  
env  
nvidia-smi

### vm specifics
SBATCH --job-name=sic  
SBATCH --cpus-per-task=4  
module cuda/8.0  
module python/3.6  
env  
nvidia-smi

## run slurm image classifier

### get code
git clone
cd sic/

### update conda env (optional)
conda update -n root conda  
conda update conda  
conda update anaconda  
conda update --all

### create & activate env
conda create -n py36 python=3.6  
source activate py36  

### update pip & install required python packages
pip install --upgrade pip  
pip install --upgrade -r requirements.txt  

### execute benchmarks
./run_models_batch.sh  
sbatch slurm_vm.sh  
sbatch slurm_dgx1.sh
