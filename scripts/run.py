import os

# Prepare single scale dataset
os.system("python ./prepare_dataset.py --images_dir ../data/DIV2K/original/train --output_dir ../data/DIV2K/EDSR/train")
os.system("python ./prepare_dataset.py --images_dir ../data/DIV2K/original/valid --output_dir ../data/DIV2K/EDSR/valid")
