import os

# Prepare dataset
os.system("python ./prepare_dataset.py --images_dir ../data/DIV2K/original/train --output_dir ../data/DIV2K/EDSR/train --image_size 216 --step 108 --num_workers 10")
os.system("python ./prepare_dataset.py --images_dir ../data/DIV2K/original/valid --output_dir ../data/DIV2K/EDSR/valid --image_size 216 --step 108 --num_workers 10")
