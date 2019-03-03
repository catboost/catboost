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

CatBoost tutorial with base features demonstration and Caret demo. See [notebook](../tutorials/r_tutorial.ipynb).


### Maintaining the package

Here are the commands to accomplish frequent tasks:

* Regenerate documentation
    ```
    python mk_package.py --generate-doc-with-r
    ```
    or
    ```
    R -e 'devtools::document()'
    ```

* Check the package
    ```
    python mk_package.py --check-with-r
    ```
    or
    ```
    R CMD check . --no-examples
    ```

* Build the package without R
    ```
    python mk_package.py --build
    ```

* Build the package with R
    ```
    python mk_package.py --build-with-r
    ```
