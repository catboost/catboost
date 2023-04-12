# CatBoost Model Node package

A Node.js package for applying pretrained CatBoost model.

## Usage example

1. Install the package. You have two options:
   - Install from npm registry:
   ```sh
   npm i catboost
   ```
   - Build package from source. Navigate to this directory inside the repo and run:

    ```sh
    npm install
    ```

    Now you can link this package in your project via:

    ```sh
    npm install $PATH_TO_CATBOOST_REPO/catboost/node-package
    ```

1. Apply the pretrained model:

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

### Release procedure

See [DEPLOYMENT.md](./DEPLOYMENT.md).
