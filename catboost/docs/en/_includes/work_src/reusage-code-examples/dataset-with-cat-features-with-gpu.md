
Load a dataset with numerical and categorical features, define the training parameters and start the training on GPU:
```r
library(catboost)

countries = c('RUS','USA','SUI')
years = c(1900,1896,1896)
phone_codes = c(7,1,41)
domains = c('ru','us','ch')

dataset = data.frame(countries, years, phone_codes, domains)

label_values = c(0,1,1)

fit_params <- list(iterations = 100,
                   loss_function = 'Logloss',
                   ignored_features = c(4,9),
                   border_count = 32,
                   depth = 5,
                   learning_rate = 0.03,
                   l2_leaf_reg = 3.5,
                   task_type = 'GPU')

pool = catboost.load_pool(dataset, label = label_values, cat_features = c(0,3))

model <- catboost.train(pool, params = fit_params)
```
