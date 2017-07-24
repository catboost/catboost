#!/bin/bash -ex

if [ "${CB_BUILD_AGENT}" == 'clang-linux-x86_64-debug' ]; then
    ./ya make --stat -d -j 1 catboost/app catboost/pytest;
fi

if [ "${CB_BUILD_AGENT}" == 'clang-linux-x86_64-release' ]; then
    ./ya make --stat -rttt -j 1 catboost/app catboost/pytest;
fi

if [ "${CB_BUILD_AGENT}" == 'python2-linux-x86_64-release' ]; then
    cd catboost/python-package;
    python2 ./mk_wheel.py -j 1;
fi

if [ "${CB_BUILD_AGENT}" == 'python3-linux-x86_64-release' ]; then
    cd catboost/python-package;
    python3 ./mk_wheel.py -j 1;
fi

if [ "${CB_BUILD_AGENT}" == 'clang-darwin-x86_64-debug' ]; then
    ./ya make --stat -d -j 1 catboost/app catboost/pytest;
fi

if [ "${CB_BUILD_AGENT}" == 'clang-darwin-x86_64-release' ]; then
    ./ya make --stat -rttt -j 1 catboost/app catboost/pytest;
fi

if [ "${CB_BUILD_AGENT}" == 'python2-darwin-x86_64-release' ]; then
    cd catboost/python-package;
    python2 ./mk_wheel.py -j 1;
fi

if [ "${CB_BUILD_AGENT}" == 'python3-darwin-x86_64-release' ]; then
    cd catboost/python-package;
    python3 ./mk_wheel.py -j 1;
fi
