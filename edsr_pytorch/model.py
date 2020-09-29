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
import math

import torch
import torch.nn as nn
from torch.cuda import amp


class ResidualBlock(nn.Module):
    r"""Main residual block structure"""

    def __init__(self, channels):
        r"""Initializes internal Module state, shared by both nn.Module and ScriptModule.

        Args:
            channels (int): Number of channels in the input image.

        """
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, inputs):
        out = self.conv1(inputs)
        out = self.relu(out)
        out = self.conv2(out)
        out *= 0.1
        return out + inputs


class EDSR(nn.Module):

    def __init__(self, scale_factor, init_weights=True):
        super(EDSR, self).__init__()
        upsample_block_num = int(math.log(scale_factor, 2))

        self.sub_mean = MeanShift()
        self.add_mean = MeanShift(sign=1)

        # First layer
        self.conv1 = nn.Conv2d(3, 256, kernel_size=3, padding=1, bias=False)

        # Residual blocks
        residual_blocks = []
        for _ in range(32):
            residual_blocks.append(ResidualBlock(256))
        self.residual_blocks = nn.Sequential(*residual_blocks)

        # First layer
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)

        # Upsampling layers
        upsampling = []
        for _ in range(upsample_block_num):
            upsampling += [
                nn.Conv2d(256, 1024, kernel_size=3, stride=1, padding=1, bias=False),
                nn.PixelShuffle(upscale_factor=2),
                nn.ReLU(inplace=True)
            ]
        self.upsampling = nn.Sequential(*upsampling)

        # Final output layer
        self.conv3 = nn.Conv2d(256, 3, kernel_size=3, stride=1, padding=1, bias=False)

        # Init weights
        if init_weights:
            self._initialize_weights()

    @amp.autocast()
    def forward(self, inputs):
        out = self.sub_mean(inputs)
        out = self.conv1(out)

        residual = out
        out = self.residual_blocks(out)
        out = self.conv2(out)
        out = out + residual

        out = self.upsampling(out)
        out = self.conv3(out)
        out = self.add_mean(out)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class MeanShift(nn.Conv2d):
    def __init__(self, mean=(0.4488, 0.4371, 0.4040), std=(1.0, 1.0, 1.0), sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * 255 * torch.Tensor(mean) / std
        for p in self.parameters():
            p.requires_grad = False
