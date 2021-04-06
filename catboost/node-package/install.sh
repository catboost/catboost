#!/usr/bin/env sh

CATBOOST_RELEASE="0.24.4"

if [ ! -f "../../ya" ]
then
  curl -L https://github.com/catboost/catboost/archive/v${CATBOOST_RELEASE}.tar.gz --output ./catboost.tar.gz && \
    tar -xf ./catboost.tar.gz && \
    CATBOOST_SRC_PATH="./catboost-${CATBOOST_RELEASE}" node ./build_scripts/bootstrap.js build_model && \
    node-gyp build && \
    rm -rf ./catboost-${CATBOOST_RELEASE} ./catboost.tar.gz
else
  ./build.sh &&
    node-gyp build
fi
