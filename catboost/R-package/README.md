### Installation
#### Install via github
Run `R` in `catboost/R-package` directory, execute the following commands:

```r
install.packages('devtools')
devtools::install_github('catboost/catboost', subdir = 'catboost/R-package')
```

#### Install via local copy
Run `R` in `catboost/R-package` directory, execute the following commands:

```r
install.packages('devtools')
devtools::build()
devtools::install()
```

### Quick start

```r
# load data to R
pool_path = 'train_full3'
column_description_path = 'train_full3.cd'
pool <- catboost.load_pool(pool_path, column_description = column_description_path)

# fit model
fit_params <- list(iterations=100, thread_count=10, loss_function='Logloss')
model <- catboost.train(pool, pool, fit_params)

# predict
prediction <- catboost.predict(model, pool)
head(prediction)
```

### Tutorials

CatBoost tutorial with base features demonstration and Caret demo. See [notebook](../tutorials/catboost_r_tutorial.ipynb).
