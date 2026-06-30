
The following is an example of exporting a model trained with [CatBoostClassifier](../../../concepts/python-reference_catboostclassifier.md) to Apple CoreML for further usage on iOS devices:
1. Train the model and save it in CoreML format.

    For example, if training on the Iris dataset:
    ```python
    import catboost
    import sklearn

    iris = sklearn.datasets.load_iris()
    cls = catboost.CatBoostClassifier(loss_function='MultiClass')

    cls.fit(iris.data, iris.target)

    # Save model to catboost format
    cls.save_model("iris.mlmodel", format="coreml", export_parameters={'prediction_type': 'probability'})
    ```

1. Import the resulting model to XCode.

    The following is an example of importing with Swift:
    ```python
    import CoreML

    let model = iris()
    let sepal_l = 7.0
    let sepal_w = 3.2
    let petal_l = 4.7
    let petal_w = 1.4

    guard let output = try? model.prediction(input: irisInput(feature_0: sepal_l, feature_1: sepal_w, feature_2: petal_l, feature_3: petal_w)) else {
    fatalError("Unexpected runtime error.")
    }

    print(String(
    format: "Output probabilities: %1.5f; %1.5f; %1.5f",
    output.prediction[0].doubleValue,
    output.prediction[1].doubleValue,
    output.prediction[2].doubleValue
    ))
    ```
