#!/usr/bin/env bash

# For conan >= 2.4.1
# For CatBoost Linux CI Docker container only

set -e
set -x

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

conan config install $SCRIPT_DIR/settings_user.yml
conan profile detect -e
conan install /src/catboost --profile:host=$SCRIPT_DIR/profiles/manylinux2014.x86_64.profile --profile:build=$SCRIPT_DIR/profiles/build.manylinux2014.x86_64.profile --build=missing
conan install /src/catboost --profile:host=$SCRIPT_DIR/profiles/dockcross.manylinux2014_aarch64.profile --profile:build=$SCRIPT_DIR/profiles/build.manylinux2014.x86_64.profile --build=missing
