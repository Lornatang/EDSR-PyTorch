# Copyright 2020 Dakewe Biotech Corporation. All Rights Reserved.
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

import cv2
import torch.backends.cudnn as cudnn
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image
from sewar.full_ref import mse
from sewar.full_ref import msssim
from sewar.full_ref import psnr
from sewar.full_ref import rmse
from sewar.full_ref import sam
from sewar.full_ref import ssim
from sewar.full_ref import vifp

from edsr_pytorch import EDSR
from edsr_pytorch import cal_niqe
from edsr_pytorch import img2tensor

parser = argparse.ArgumentParser(description="Enhanced Deep Residual Networks for Single Image Super-Resolution")
parser.add_argument("--dataroot", type=str, default="./data/Set5",
                    help="The directory address where the image needs "
                         "to be processed. (default: `./data/Set5`).")
parser.add_argument("--scale-factor", type=int, default=4, choices=[2, 4],
                    help="Image scaling ratio. (default: 4).")
parser.add_argument("--weights", type=str, default="weights/edsr_4x.pth",
                    help="Generator model name.  (default:`weights/edsr_4x.pth`)")
parser.add_argument("--outf", default="./result",
                    help="folder to output images. (default:`./result`).")
parser.add_argument("--cuda", action="store_true", help="Enables cuda")

args = parser.parse_args()
print(args)

try:
    os.makedirs(args.outf)
except OSError:
    pass

cudnn.benchmark = True

if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda:0" if args.cuda else "cpu")

# create model
model = EDSR(scale_factor=args.scale_factor).to(device)

# Load state dicts
model.load_state_dict(torch.load(args.weights, map_location=device))

# Set eval mode
model.eval()

# Evaluate algorithm performance
total_mse_value = 0.0
total_rmse_value = 0.0
total_psnr_value = 0.0
total_ssim_value = 0.0
total_ms_ssim_value = 0.0
total_niqe_value = 0.0
total_sam_value = 0.0
total_vif_value = 0.0
# Count the number of files in the directory
total_file = 0

dataroot = f"{args.dataroot}/{args.scale_factor}x/data"
target = f"{args.dataroot}/{args.scale_factor}x/target"

lr_process = transforms.Compose([transforms.CenterCrop(args.image_size * args.scale_factor),
                                 transforms.Resize(args.image_size),
                                 img2tensor()])
hr_process = transforms.Compose([transforms.CenterCrop(args.image_size * args.scale_factor),
                                 img2tensor()])

for filename in os.listdir(dataroot):
    # Open image
    image = Image.open(f"{dataroot}/{filename}")
    lr_real_image = lr_process(image).unsqueeze(0)
    hr_real_image = hr_process(image).unsqueeze(0)

    lr_real_image = lr_real_image.to(device)
    hr_fake_image = model(lr_real_image)
    vutils.save_image(hr_real_image, f"{args.outf}/{filename.split('.')[-1]}_edsr.png", normalize=True)
    vutils.save_image(hr_fake_image, f"{target}/{filename.split('.')[-1]}_hr.png", normalize=True)

    # Evaluate performance
    src_img = cv2.imread(f"{args.outf}/{filename.split('.')[-1]}_edsr.png")
    dst_img = cv2.imread(f"{target}/{filename.split('.')[-1]}_hr.png")

    total_mse_value += mse(src_img, dst_img)
    total_rmse_value += rmse(src_img, dst_img)
    total_psnr_value += psnr(src_img, dst_img)
    total_ssim_value += ssim(src_img, dst_img)
    total_ms_ssim_value += msssim(src_img, dst_img)
    total_niqe_value += cal_niqe(f"{args.outf}/{filename.split('.')[-1]}_edsr.png")
    total_sam_value += sam(src_img, dst_img)
    total_vif_value += vifp(src_img, dst_img)

    total_file += 1

print(f"Avg MSE: {total_mse_value / total_file:.2f}\n"
      f"Avg RMSE: {total_rmse_value / total_file:.2f}\n"
      f"Avg PSNR: {total_psnr_value / total_file:.2f}\n"
      f"Avg SSIM: {total_ssim_value / total_file:.4f}\n"
      f"Avg MS-SSIM: {total_ms_ssim_value / total_file:.4f}\n"
      f"Avg NIQE: {total_niqe_value / total_file:.2f}\n"
      f"Avg SAM: {total_sam_value / total_file:.4f}\n"
      f"Avg VIF: {total_vif_value / total_file:.4f}")
