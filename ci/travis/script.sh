#!/bin/bash -ex

function install_cuda_linux()
{
    wget https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda-repo-ubuntu1404-8-0-local-ga2_8.0.61-1_amd64-deb -O cuda-repo-ubuntu1404-8-0-local-ga2_8.0.61-1_amd64.deb
    sudo dpkg -i cuda-repo-ubuntu1404-8-0-local-ga2_8.0.61-1_amd64.deb
    sudo apt-get update
    sudo apt-get install cuda    
}


if [ "${CB_BUILD_AGENT}" == 'clang-linux-x86_64-release-cuda' ]; then
    install_cuda_linux;
    ./ya make --stat -T -r -j 1 catboost/app -DCUDA_ROOT=/usr/local/cuda-8.0;
    cp $(readlink -f catboost/app/catboost) catboost-cuda-linux;
fi

if [ "${CB_BUILD_AGENT}" == 'python2-linux-x86_64-release' ]; then
     install_cuda_linux;
     cd catboost/python-package;
     python2 ./mk_wheel.py -T -j 1 -DCUDA_ROOT=/usr/local/cuda-8.0;
fi

if [ "${CB_BUILD_AGENT}" == 'python35-linux-x86_64-release' ]; then
     install_cuda_linux;
     cd catboost/python-package;
     python3 ./mk_wheel.py -T -j 1 -DCUDA_ROOT=/usr/local/cuda-8.0;
fi

if [ "${CB_BUILD_AGENT}" == 'python36-linux-x86_64-release' ]; then
     install_cuda_linux;
     cd catboost/python-package;
     python3 ./mk_wheel.py -T -j 1 -DCUDA_ROOT=/usr/local/cuda-8.0;
fi

if [ "${CB_BUILD_AGENT}" == 'clang-darwin-x86_64-release' ]; then
    ./ya make --stat -T -r -j 1 catboost/app;
    cp $(readlink catboost/app/catboost) catboost-darwin;
fi
