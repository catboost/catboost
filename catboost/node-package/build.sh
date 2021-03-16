#!/usr/bin/env sh

node-gyp configure && \
    cp ../libs/model_interface/c_api.h ./build/c_api.h && \
    cp -r ../../util ./build/util && \
    cp -r ../../contrib/libs/cxxsupp/system_stl/include ./build/libcxx_include && \
    tsc && \
    cp bindings/catboost.* lib && \
    ../../ya make -r ../libs/model_interface -o ./build && \
    node-gyp build