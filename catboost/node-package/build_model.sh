#!/usr/bin/env sh

CATBOOST_PATH_ROOT="${CATBOOST_SRC_PATH:-../../}"

node-gyp configure && \
    cp $CATBOOST_PATH_ROOT/catboost/libs/model_interface/c_api.h ./build/c_api.h && \
    cp -r $CATBOOST_PATH_ROOT/util ./build/util && \
    cp -r $CATBOOST_PATH_ROOT/contrib/libs/cxxsupp/system_stl/include ./build/libcxx_include && \
    $CATBOOST_PATH_ROOT/ya make -r $CATBOOST_PATH_ROOT/catboost/libs/model_interface -o ./build
