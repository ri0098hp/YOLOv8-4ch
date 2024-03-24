#!/bin/bash

# PyTorch Install
wget https://developer.download.nvidia.cn/compute/redist/jp/v60dp/pytorch/torch-2.2.0a0+81ea7a4.nv24.01-cp310-cp310-linux_aarch64.whl
pip install torch-2.2.0a0+81ea7a4.nv24.01-cp310-cp310-linux_aarch64.whl

# torchvision build and install
export BUILD_VERSION=0.17.0
git clone --branch v$BUILD_VERSION https://github.com/pytorch/vision torchvision
cd torchvision
python3 setup.py bdist_wheel
cd ..
cp torchvision/dist/torchvision-$BUILD_VERSION-cp310-cp310-linux_aarch64.whl .
pip install torchvision-$BUILD_VERSION-cp310-cp310-linux_aarch64.whl
rm -rf torchvision

# onnxruntime_gpu install
wget https://nvidia.box.com/shared/static/i7n40ki3pl2x57vyn4u7e9asyiqlnl7n.whl -O onnxruntime_gpu-1.17.0-cp310-cp310-linux_aarch64.whl
pipenv run pip install onnxruntime_gpu-1.17.0-cp310-cp310-linux_aarch64.whl
rm onnxruntime_gpu-1.17.0-cp310-cp310-linux_aarch64.whl

# OpenCV with CUDA compile build
wget https://github.com/Qengineering/Install-OpenCV-Jetson-Nano/raw/main/OpenCV-4-9-0.sh
sudo chmod 755 ./OpenCV-4-9-0.sh
./OpenCV-4-9-0.sh
