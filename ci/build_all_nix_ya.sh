#!/bin/bash

#TODO(kizill): split this into subscripts to make it prettier

eval "$(pyenv init --path)"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"  # This loads nvm

set -x
set -e

if [[ -z "${CUDA_ARG}" ]]; then
  echo 'missing CUDA_ARG variable, should be something like "-DCUDA_ROOT=/usr/local/cuda-9.1/"'
fi

function python_version {
    case `$1 --version 2>&1` in
        Python*3.5*) echo 3.5 ;;
        Python*3.6*) echo 3.6 ;;
        Python*3.7*) echo 3.7 ;;
        Python*3.8*) echo 3.8 ;;
        Python*3.9*) echo 3.9 ;;
        Python*3.10*) echo 3.10 ;;
        *) echo "Cannot determine python version" ; exit 1 ;;
    esac
}

function os_sdk {
    python_version=`python_version python`
    case `uname -s` in
        Linux) echo "-DOS_SDK=ubuntu-12 -DUSE_SYSTEM_PYTHON=$python_version" ;;
        *) echo "-DOS_SDK=local" ;;
    esac
}


lnx_common_flags="-DNO_DEBUGINFO $CUDA_ARG"

python ya make -r $lnx_common_flags $(os_sdk) $YA_MAKE_EXTRA_ARGS -o . catboost/app
python ya make -r $lnx_common_flags $(os_sdk) $YA_MAKE_EXTRA_ARGS -o . catboost/libs/model_interface

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

python ../../ya make -r $lnx_common_flags $(os_sdk) $YA_MAKE_EXTRA_ARGS -T src

mkdir -p catboost/inst/libs
[ -s "src/libcatboostr.so" ] && cp $(readlink src/libcatboostr.so) catboost/inst/libs
[ -s "src/libcatboostr.dylib" ] && cp $(readlink src/libcatboostr.dylib) catboost/inst/libs && ln -s libcatboostr.dylib catboost/inst/libs/libcatboostr.so

tar -cvzf catboost-R-$(uname).tgz catboost

cd ../python-package

PY36=3.6.6
pyenv install -s $PY36
pyenv shell $PY36
python mk_wheel.py --build-system YA $lnx_common_flags $(os_sdk) $YA_MAKE_EXTRA_ARGS -DPYTHON_CONFIG=$(pyenv prefix)/bin/python3-config

PY37=3.7.0
pyenv install -s $PY37
pyenv shell $PY37
python mk_wheel.py --build-system YA $lnx_common_flags $(os_sdk) $YA_MAKE_EXTRA_ARGS -DPYTHON_CONFIG=$(pyenv prefix)/bin/python3-config

PY38=3.8.0
pyenv install -s $PY38
pyenv shell $PY38
python mk_wheel.py --build-system YA $lnx_common_flags $(os_sdk) $YA_MAKE_EXTRA_ARGS -DPYTHON_CONFIG=$(pyenv prefix)/bin/python3-config

PY39=3.9.0
pyenv install -s $PY39
pyenv shell $PY39
python mk_wheel.py --build-system YA $lnx_common_flags $(os_sdk) $YA_MAKE_EXTRA_ARGS -DPYTHON_CONFIG=$(pyenv prefix)/bin/python3-config

PY310=3.10.0
pyenv install -s $PY310
pyenv shell $PY310
python mk_wheel.py --build-system YA $lnx_common_flags $(os_sdk) $YA_MAKE_EXTRA_ARGS -DPYTHON_CONFIG=$(pyenv prefix)/bin/python3-config


# JVM prediction native shared library

cd ../jvm-packages/catboost4j-prediction

python ../tools/build_native_for_maven.py . catboost4j-prediction --build release --no-src-links \
-DOS_SDK=local -DHAVE_CUDA=no -DUSE_SYSTEM_JDK=$JAVA_HOME -DJAVA_HOME=$JAVA_HOME $YA_MAKE_EXTRA_ARGS

# Spark native shared library

cd ../../spark/catboost4j-spark/core

python ../../../jvm-packages/tools/build_native_for_maven.py . catboost4j-spark-impl --build release --no-src-links \
-DOS_SDK=local -DHAVE_CUDA=no -DUSE_LOCAL_SWIG=yes -DUSE_SYSTEM_JDK=$JAVA_HOME -DJAVA_HOME=$JAVA_HOME $YA_MAKE_EXTRA_ARGS

