#!/bin/bash

#TODO(kizill): split this into subscripts to make it prettier

eval "$(pyenv init --path)"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"  # This loads nvm

set -x
set -e

CMAKE_COMMON_ARGS="./catboost/ -G \"Ninja\" -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=$PWD/catboost/build/toolchains/clang.toolchain $CMAKE_EXTRA_ARGS"

src_root_dir=$PWD

# generate common build dirs (non python-version specific)

build_dir_nopic=../build_nopic
build_dir_pic=../build_pic

cmake $CMAKE_COMMON_ARGS -B $build_dir_nopic
cmake $CMAKE_COMMON_ARGS -DCMAKE_POSITION_INDEPENDENT_CODE=On -B $build_dir_pic

# catboost app
cd $build_dir_nopic && make catboost

# model interface
cd $build_dir_nopic && make libcatboostmodel
cd $build_dir_pic && make catboostmodel

cd $src_root_dir

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

cd $build_dir_pic && make catboostr

cd $src_root_dir/R-package

mkdir -p catboost/inst/libs
[ -s "src/libcatboostr.so" ] && cp $(readlink src/libcatboostr.so) catboost/inst/libs
[ -s "src/libcatboostr.dylib" ] && cp $(readlink src/libcatboostr.dylib) catboost/inst/libs && ln -s libcatboostr.dylib catboost/inst/libs/libcatboostr.so

tar -cvzf catboost-R-$(uname).tgz catboost

echo "Starting Python package build"

cd ../python-package

PY36=3.6.6
pyenv install -s $PY36
pyenv shell $PY36
python mk_wheel.py --build-system CMAKE

PY37=3.7.0
pyenv install -s $PY37
pyenv shell $PY37
python mk_wheel.py --build-system CMAKE

PY38=3.8.0
pyenv install -s $PY38
pyenv shell $PY38
python mk_wheel.py --build-system CMAKE

PY39=3.9.0
pyenv install -s $PY39
pyenv shell $PY39
python mk_wheel.py --build-system CMAKE

PY310=3.10.0
pyenv install -s $PY310
pyenv shell $PY310
python mk_wheel.py --build-system CMAKE

PY311=3.11.0
pyenv install -s $PY311
pyenv shell $PY311
python mk_wheel.py --build-system CMAKE


# JVM prediction native shared library

cd $build_dir_pic && make catboost4j-prediction

# Spark native shared library

cd $build_dir_pic && make catboost4j-spark-impl-cpp
