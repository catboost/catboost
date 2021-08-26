# Quick start 

To get started:
1. Prepare a dataset using the [catboost.load_pool](r-reference_catboost-load_pool.md) function:
    
    ```r
    library(catboost)
    
    features <- data.frame(feature1 = c(1, 2, 3), feature2 = c('A', 'B', 'C'))
    labels <- c(0, 0, 1)
    train_pool <- catboost.load_pool(data = features, label = labels)
    ```
    
    The dataset is created from a synthetic `data.frame` called `features` in this example. The `data` argument can also reference a [dataset file](input-data_values-file.md) or a matrix of numerical features.
    
1. Train the model using the [catboost.train](r-reference_catboost-train.md) function:
    ```r
    model <- catboost.train(train_pool,  NULL,
    params = list(loss_function = 'Logloss',
    iterations = 100, metric_period=10))
    ```
    
    The second argument in this example (`test_pool`) is set to NULL. It can also be used to pass a validation dataset (the labelled data used for estimating the prediction error while training). The `params` argument is used to specify the training parameters.
    
1. Apply the trained model using the [catboost.predict](r-reference_catboost-predict.md) function:
    
    ```r
    real_data <- data.frame(feature1 = c(2, 1, 3), feature2 = c('D', 'B', 'C'))
    real_pool <- catboost.load_pool(real_data)
    
    prediction <- catboost.predict(model, real_pool)
    print(prediction)
    ```

