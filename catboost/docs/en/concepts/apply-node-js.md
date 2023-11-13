# CatBoost Model Node package

A Node.js package for applying pretrained CatBoost models.

## Installation

Install the package. You have two options:
   - Install from npm registry:
        ```sh
        npm i catboost
        ```
   - Build package from source.

        {% include [get-source-code-from-github](../_includes/work_src/reusage-installation/get-source-code-from-github.md) %}

        Navigate to `$PATH_TO_CATBOOST_REPO/catboost/node-package` directory inside the repo and run:

        ```sh
        npm install
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
