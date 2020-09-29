#!/bin/bash

echo "Start downloading pre training model..."
wget https://github.com/Lornatang/EDSR-PyTorch/releases/download/1.0/edsr_2x.pth
wget https://github.com/Lornatang/EDSR-PyTorch/releases/download/1.0/edsr_4x.pth
echo "All pre training models have been downloaded!"
