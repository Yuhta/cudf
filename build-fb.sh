#!/bin/sh

GCC_PATH=$HOME/fbsource/fbcode/third-party2/gcc/11.x/centos7-native/886b5eb/bin
BINUTILS_PATH=$HOME/fbsource/fbcode/third-party2/binutils/2.37/centos7-native/da39a3e/bin
export CUDA_PATH=$HOME/local/cuda-11.8/bin
export CC=$GCC_PATH/gcc
export CXX=$GCC_PATH/g++
export LD=$BINUTILS_PATH/ld
export NVCC_PREPEND_FLAGS="-ccbin $CXX"
export PATH=$CUDA_PATH:$BINUTILS_PATH:$GCC_PATH:$HOME/local/cmake-3.25.0-linux-x86_64/bin:$PATH
export LD_LIBRARY_PATH=$GCC_PATH/../lib64:$LD_LIBRARY_PATH
INSTALL_PREFIX=./build/install ./build.sh --ptds libcudf benchmarks
