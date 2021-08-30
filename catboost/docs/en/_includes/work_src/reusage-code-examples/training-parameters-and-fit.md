
Load a dataset with numerical features, define the training parameters and start the training:
```r
library(catboost)

dataset = matrix(c(1900,7,
                   1896,1,
                   1896,41),
                 nrow=3, 
                 ncol=2, 
                 byrow = TRUE)
label_values = c(0,1,1)

fit_params <- list(iterations = 100,
                   loss_function = 'Logloss')

pool = catboost.load_pool(dataset, label = label_values)

model <- catboost.train(pool, params = fit_params)
```
