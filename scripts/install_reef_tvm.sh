#!/bin/bash
apt-get update
apt-get install -y python3 python3-dev python3-setuptools gcc libtinfo-dev zlib1g-dev build-essential cmake libedit-dev libxml2-dev
apt-get install -y git python3-pip lsb-release wget software-properties-common gnupg
bash -c "$(wget -O - https://apt.llvm.org/llvm.sh)"

cd /usr
git clone -b reef https://github.com/francis0407/tvm.git --recursive
cd /usr/tvm

touch config.cmake
echo set\(USE_LLVM ON\) >> config.cmake
echo set\(USE_CUDA ON\) >> config.cmake
echo set\(USE_CUDNN ON\) >> config.cmake
echo set\(USE_BLAS openblas\) >> config.cmake
echo set\(USE_CUBLAS ON\) >> config.cmake
echo set\(USE_MPS ON\) >> config.cmake
mkdir build
cp config.cmake build

cd build
cmake ..
make -j4

apt-get install -y libprotobuf-dev protobuf-compiler
pip3 install --upgrade pip3
pip3 install --user numpy decorator attrs
pip3 install protobuf
pip3 install onnx