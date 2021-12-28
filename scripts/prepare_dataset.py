# Copyright 2021 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import argparse
import os
import shutil

from PIL import Image
from tqdm import tqdm


def main() -> None:
    step = int(args.image_size * 0.8)

    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir)

    file_names = os.listdir(args.images_dir)
    for file_name in tqdm(file_names, total=len(file_names)):
        # Use PIL to read high-resolution image
        image = Image.open(f"{args.images_dir}/{file_name}")
        index = 1
        for pos_x in range(0, image.size[0] - args.image_size + 1, step):
            for pos_y in range(0, image.size[1] - args.image_size + 1, step):
                # crop box xywh
                crop_image = image.crop([pos_x, pos_y, pos_x + args.image_size, pos_y + args.image_size])
                # Save all images
                crop_image.save(f"{args.output_dir}/{file_name.split('.')[-2]}_{index:05d}.{file_name.split('.')[-1]}")
                index += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare database scripts.")
    parser.add_argument("--images_dir", type=str, default="DIV2K/original", help="Path to input image directory. (Default: `DIV2K/original`)")
    parser.add_argument("--output_dir", type=str, default="DIV2K/EDSR/train", help="Path to generator image directory. (Default: `DIV2K/EDSR/train`)")
    parser.add_argument("--image_size", type=int, default=400, help="Low-resolution image size from raw image. (Default: 400)")
    args = parser.parse_args()

    main()
