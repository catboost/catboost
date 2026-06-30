# CatBoost Model Node package

A Node.js package for applying pretrained CatBoost models.

## Installation

Install the package. You have two options:
   - Install from npm registry:
        ```sh
        npm i catboost
        ```
   - Build package from source.

        CatBoost Node.js package is a wrapper around [`libcatboostmodel` library](https://catboost.ai/docs/en/concepts/c-plus-plus-api_dynamic-c-pluplus-wrapper) with an exposed C API.

        In order to build it some environment setup is necessary. Modern versions of CatBoost use CMake build system, build environment setup for CMake is described [here](https://catboost.ai/docs/en/installation/build-environment-setup-for-cmake), CatBoost versions before 1.2 used Ya Make build system, build environment setup for YaMake is described [here](https://catboost.ai/docs/en/installation/build-environment-setup-for-ya-make).

        ------

        CatBoost source code is stored as a [Git](https://git-scm.com/) repository on GitHub at <https://github.com/catboost/catboost/>. You can obtain a local copy of this Git repository by running the following command from a command line interpreter (you need to have Git command line tools installed):

        ```sh
        git clone https://github.com/catboost/catboost.git
        ```

        Navigate to `$PATH_TO_CATBOOST_REPO/catboost/node-package` directory inside the repo and run:

        ```sh
        npm run install [-- <build_native arguments>]
        ```
        See [build_native documentation](https://catboost.ai/docs/en/installation/build-native-artifacts#build-build-native) about possible arguments. Don't specify already defined `--target` or `--build-root-dir` arguments.

        For example, build with CUDA support:

        ```sh
        npm run install -- --have-cuda
        ```

        Inference on CUDA GPUs is currently supported only for models with exclusively numerical features.

        CUDA architectures to generate device code for are specified using [`CMAKE_CUDA_ARCHITECTURES` variable](https://cmake.org/cmake/help/v3.24/variable/CMAKE_CUDA_ARCHITECTURES.html), although the default value is non-standard, [specified in `cuda.cmake`](https://github.com/catboost/catboost/blob/5fb7b9def07f4ea2df6dcc31b5cd1e81a8b00217/cmake/cuda.cmake#L7). The default value is intended to provide broad GPU compatibility and supported only when building with CUDA 11.8.
        The most convenient way to override the default value is to use [`CUDAARCHS` environment variable](https://cmake.org/cmake/help/v3.24/envvar/CUDAARCHS.html).

        Now you can link this package in your project via:

        ```sh
        npm install $PATH_TO_CATBOOST_REPO/catboost/node-package
        ```

## Usage

Apply the pretrained model.

Example with numerical and categorical features (they must be passed in separate arrays containing features of
each type for all samples):

```js
catboost = require('catboost');

model = new catboost.Model();
model.loadModel('test_data/adult.cbm');

prediction = model.predict([
            [40., 85019., 16., 0., 0., 45.],
            [28., 85019., 13., 0., 0., 13.],
        ],
        [
            ["Private", "Doctorate", "Married-civ-spouce", "Prof-specialty", "Husband", "Asian-Pac-Islander", "Male", "nan"],
            ["Self-emp-not-inc", "Bachelors", "Married-civ-spouce", "Exec-managerial", "Husband", "White", "Male", "United-States"],
        ]
);
console.log(prediction);
```

## Release procedure

See [DEPLOYMENT.md](./DEPLOYMENT.md).
