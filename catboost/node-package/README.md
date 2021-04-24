# CatBoost Model Node package

A Node.js package for applying pretrained CatBoost model.

> NOTE: The package is still under heavy development and can introduce breaking changes.  

## Usage example

1. Install the package. You have two options:
   - Install from npm registry:
   ```sh
   npm i catboost-model
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
catboost = require('catboost-model');

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

### Development roadmap

 - [x] Implement basic API calls.
 - [ ] Extend exposed API, improve test coverage.
 - [X] Migrate away from shell scripts.
 - [X] Support Windows and MacOS platforms.
 - [X] Switch to downloading and verifying the pre-built binary instead of building it from scratch.
 - [X] Publish the alpha version of the package.
 - [ ] Publish the generally available version of the package.