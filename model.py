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
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, channels) -> None:
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        out = self.conv1(inputs)
        out = self.relu(out)
        out = self.conv2(out)
        out = torch.mul(out, 0.1)
        out = torch.add(out, inputs)

        return out


class EDSR(nn.Module):
    def __init__(self, upscale_factor: int) -> None:
        super(EDSR, self).__init__()
        # First layer
        self.conv1 = nn.Conv2d(3, 256, (3, 3), (1, 1), (1, 1))

        # Residual blocks
        trunk = []
        for _ in range(32):
            trunk.append(ResidualBlock(256))
        self.trunk = nn.Sequential(*trunk)

        # First layer
        self.conv2 = nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1))

        # Upsampling layers
        upsampling = []
        if upscale_factor == 2 or upscale_factor == 4:
            for _ in range(int(math.log(upscale_factor, 2))):
                upsampling += [
                    nn.Conv2d(256, 1024, (3, 3), (1, 1), (1, 1)),
                    nn.PixelShuffle(2),
                ]
        elif upscale_factor == 3:
            upsampling += [
                nn.Conv2d(256, 2304, (3, 3), (1, 1), (1, 1)),
                nn.PixelShuffle(3),
            ]
        self.upsampling = nn.Sequential(*upsampling)

        # Final output layer
        self.conv3 = nn.Conv2d(256, 3, (3, 3), (1, 1), (1, 1))

        # Init weights
        self._initialize_weights()

        self.register_buffer("mean", torch.Tensor([0.4488, 0.4371, 0.4040]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.Tensor([1.0, 1.0, 1.0]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The images by subtracting the mean RGB value of the DIV2K dataset.
        out = x.sub_(self.mean).div_(self.std)

        out1 = self.conv1(out)
        out = self.trunk(out1)
        out = self.conv2(out)
        out = torch.add(out, out1)
        out = self.upsampling(out)
        out = self.conv3(out)

        out = out.mul_(self.std).add_(self.mean)

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
