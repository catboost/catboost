#!/usr/bin/env sh

if [ ! -f "../../ya" ]
then
  git clone https://github.com/catboost/catboost && \
    CATBOOST_SRC_PATH="./catboost" ./build_model.sh && \
    rm -rf ./catboost && \
    node-gyp build
else
  ./build.sh &&
    node-gyp build
fi
