# CatBoost Model Node package

A Node.js package for applying pretrained CatBoost model.

> NOTE: The package is still under heavy development and can introduce breaking changes.  

> NOTE: Only Linux platform is supported at the moment.

## Usage example

1. Install the package. As it is not yet published, it has to be built from source. Navigate to this directory inside the repo and run:

```sh
npm install
```

Now you can link this package in your project via:

```sh
npm install $PATH_TO_CATBOOST_REPO/catboost/node-package
```

2. Apply the pretrained model:

```js
catboost = require('catboost-model');

model = new catboost.Model();
model.loadFullFromFile('test_data/adult.cbm');

prediction = model.calcPrediction([
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

### Development plans

 - [x] Implement basic API calls.
 - [ ] Extend exposed API, improve test coverage.
 - [ ] Migrate away from shell scripts.
 - [ ] Support Windows and MacOS platforms.
 - [ ] Switch to downloading and verifying the pre-built binary instead of building it from scratch. Right now the distribution of this package will attempt to download and build source code on installation.
 - [ ] Publish the package.