#!/bin/bash

#TODO(kizill): split this into subscripts to make it prettier

eval "$(pyenv init -)"

set -x
set -e

if [[ -z "${CUDA_ARG}" ]]; then
  echo 'missing CUDA_ARG variable, should be something like "-DCUDA_ROOT=/usr/local/cuda-9.1/"'
fi

function python_version {
    case `$1 --version 2>&1` in
        Python*2.7*) echo 2.7 ;;
        Python*3.5*) echo 3.5 ;;
        Python*3.6*) echo 3.6 ;;
        Python*3.7*) echo 3.7 ;;
        *) echo "Cannot determine python version" ; exit 1 ;;
    esac
}

case `uname -s` in
    Linux) OS_SDK="-DOS_SDK=ubuntu-10" USE_SYSTEM_PYTHON="-DUSE_SYSTEM_PYTHON=`python_version python`" ;;
    *) OS_SDK="-DOS_SDK=local" ;;
esac

python ya make -r -DNO_DEBUGINFO $OS_SDK $USE_SYSTEM_PYTHON $CUDA_ARG -o . catboost/app

echo "Starting R package build"
cd catboost/R-package
mkdir -pv catboost

cp DESCRIPTION catboost
cp NAMESPACE catboost
cp README.md catboost

cp -r R catboost

cp -r inst catboost
cp -r man catboost
cp -r tests catboost

python ../../ya make -r -DNO_DEBUGINFO -T src $OS_SDK $USE_SYSTEM_PYTHON $CUDA_ARG

mkdir -p catboost/inst/libs
cp $(readlink src/libcatboostr.so) catboost/inst/libs

tar -cvzf catboost-R-$(uname).tgz catboost

cd ../python-package

PY27=2.7.14
pyenv install -s $PY27
pyenv shell $PY27
python mk_wheel.py $OS_SDK -DUSE_SYSTEM_PYTHON=2.7 $CUDA_ARG -DPYTHON_CONFIG=$(pyenv prefix)/bin/python2-config

PY35=3.5.5
pyenv install -s $PY35
pyenv shell $PY35
python mk_wheel.py $OS_SDK -DUSE_SYSTEM_PYTHON=3.5 $CUDA_ARG -DPYTHON_CONFIG=$(pyenv prefix)/bin/python3-config

PY36=3.6.6
pyenv install -s $PY36
pyenv shell $PY36
python mk_wheel.py $OS_SDK -DUSE_SYSTEM_PYTHON=3.6 $CUDA_ARG -DPYTHON_CONFIG=$(pyenv prefix)/bin/python3-config

PY37=3.7.0
pyenv install -s $PY37
pyenv shell $PY37
python mk_wheel.py $OS_SDK -DUSE_SYSTEM_PYTHON=3.7 $CUDA_ARG -DPYTHON_CONFIG=$(pyenv prefix)/bin/python3-config
