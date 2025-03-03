# CatBoost Model Node package

A Node.js package for applying pretrained CatBoost models.

## Installation

Install the package. You have two options:
   - Install from npm registry:
        ```sh
        npm i catboost
        ```
   - Build package from source.

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
