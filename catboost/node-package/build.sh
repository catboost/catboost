#!/usr/bin/env sh

tsc && cp bindings/catboost.* lib && \
    ../../ya make -r ../libs/model_interface -o ./build && \
    node-gyp build