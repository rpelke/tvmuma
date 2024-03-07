#!/bin/bash

# Get directory of script
SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do 
    DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"
    SOURCE="$(readlink "$SOURCE")"
    [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE"
done
DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"

TVM_PATH=$DIR/tvm
VTA_HW_PATH=$TVM_PATH/3rdparty/vta-hw
BUILD_DIR=$TVM_PATH/BUILD/DEBUG/BUILD
INSTALL_DIR=$TVM_PATH/BUILD/DEBUG

git clone --recursive https://github.com/apache/tvm.git --depth=1 --branch=v0.15.0 $TVM_PATH

mkdir -p $BUILD_DIR
cd $BUILD_DIR
cmake \
    -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR \
    -DCMAKE_BUILD_TYPE=DEBUG \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_C_COMPILER=clang \
    -DUSE_LLVM=ON \
    -DUSE_GRAPH_EXECUTOR=ON\
    -DUSE_PROFILER=ON \
    -DUSE_VTA_FSIM=ON \
    -DUSE_RELAY_DEBUG=ON \
    -DUSE_MICRO=ON \
    -DUSE_UMA=ON \
    -G Ninja \
    ../../..

cmake --build $BUILD_DIR -j `nproc`
cmake --install $BUILD_DIR

export PYTHONPATH="$TVM_PATH/python:$TVM_PATH/vta/python:$PYTHONPATH"
export TVM_LIBRARY_PATH="$BUILD_DIR"
export LD_LIBRARY_PATH="$INSTALL_DIR/lib"

echo "PYTHONPATH: $PYTHONPATH"
echo "TVM_LIBRARY_PATH: $TVM_LIBRARY_PATH"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
