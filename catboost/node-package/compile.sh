#!/usr/bin/env sh

tsc && \
    cp bindings/catboost.* lib && \
    node-gyp build