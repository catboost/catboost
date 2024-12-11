#!/usr/bin/env sh

docker build -f ./ubuntu_22.04.dockerfile --target base -t ghcr.io/catboost/build_env_no_cuda:ubuntu_22.04 .
docker build -f ./ubuntu_22.04.dockerfile --target with_cuda -t ghcr.io/catboost/build_env_with_cuda:ubuntu_22.04 .
