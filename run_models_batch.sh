#!/bin/sh
#                                                                             
# PURPOSE: Runs all three models to test which provides best solution
#
# Usage: sh run_models_batch.sh    -- will run program from commandline

python check_images.py --dir pet_images/ --arch resnet  --dogfile dognames.txt
echo '------------------------------------------------------------------------\n'
python check_images.py --dir pet_images/ --arch alexnet --dogfile dognames.txt
echo '------------------------------------------------------------------------\n'
python check_images.py --dir pet_images/ --arch vgg  --dogfile dognames.txt