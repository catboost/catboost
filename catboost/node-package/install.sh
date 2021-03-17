#!/usr/bin/env sh

if [ ! -d "./build" ] 
then
  git clone https://github.com/catboost/catboost && \
    CATBOOST_SRC_PATH="./catboost" ./build_model.sh && \
    rm -rf ./catboost && \
    node-gyp build
else
  node-gyp configure && \
    node-gyp build
fi 
