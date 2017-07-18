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

```sh
# download data
sky get -wu --network=Backbone rbtorrent:45de565860210ca922cc684d33f6a6b989320456
```

```r
# load data to R
pool_path = 'train_full3'
column_description_path = 'train_full3.cd'
pool <- catboost.load(pool_path, column_description_path)

# fit model
fit_params <- list(iterations=100, thread_count=10, loss_function='Logloss')
model <- catboost.train(pool, pool, cat_features, fit_params)

# predict
prediction <- catboost.predict(model, pool)
head(prediction)
```


### Make catboost pool

#### `catboost.load`

Two files are needed to create catboost pool in R:

* file with features

```sh
> cat train_full3 | head -2
 1       81076   14638   117961  118052  118992  118784  120474  290919  118786
 1       81482   6454    117961  118343  118856  120773  165589  118960  120774
```

* column description file

```sh
> cat train_full3.cd | head -3
 0       Target
 1       Categ
 2       Categ
```

Column indices are 0-based, column types must be one of:

* Target (one column);

* Categ;

* Num (default type).

Indices and description of numeric columns can be omitted.

```r
# load pool from path
pool_path = 'train_full3'
column_description_path = 'train_full3.cd'
pool <- catboost.load(pool_path, column_description_path)

# load pool from package
pool_path = system.file("extdata", "adult_train.1000", package="catboost")
column_description_path = system.file("extdata", "adult.cd", package="catboost")
pool <- catboost.load(pool_path, column_description_path)
```


#### `catboost.from_matrix`

Categorical features must be transformed to numeric columns using your own method (e.g. string hash).
Indices in `cat_features` vector are 0-based and can be different from indices in `.cd` file.

```r
pool_path = 'train_full3'
data <- read.table(pool_path, head=F, sep="\t", colClasses=rep('numeric', 10))
target <- c(1)
cat_features <- seq(1,8)
data_matrix <- as.matrix(data)
pool <- catboost.from_matrix(data = as.matrix(data[,-target]),
                             target = as.matrix(data[,target]),
                             cat_features = cat_features)
```


#### `catboost.from_data_frame`
Categorical features must be converted to factors (use `as.factor()`, colClasses argument of `read.table()` etc).
Numeric features must be presented as type numeric.
Target feature must be presented as type numeric.

```r
pool_path = 'train_full3'
column_description_vector <- c('numeric',                             # target
               rep('numeric',2), rep('factor',7))     # features
data <- read.table(pool_path, head=F, sep="\t", colClasses=column_description_vector)
target <- c(1)
learn_size <- floor(0.8 * nrow(data))
learn_ind <- sample(nrow(data), learn_size)
learn <- data[learn_ind,]
test <- data[-learn_ind,]
learn_pool <- catboost.from_data_frame(data=learn[,-target], target=learn[,target])
test_pool <- catboost.from_data_frame(data=test[,-target], target=test[,target])
```

### Explore pool
```r
# number of rows
cat("Nrows: ", catboost.nrow(learn_pool),"\n")
# first rows of pool
cat("First row: ", catboost.head(learn_pool, n = 1),"\n")
```

### Train model
See `help(catboost.train)` for all arguments and description.
Loss functions: RMSE, MAE, Logloss, CrossEntropy, Quantile, LogLinQuantile, Poisson, MAPE, MultiClass, AUC.

```r
fit_params <- list(iterations = 100,
                   thread_count = 10,
                   loss_function = 'Logloss',
                   ignored_features = c(4,9),
                   border_count = 32,
                   depth = 5,
                   learning_rate = 0.03,
                   l2_leaf_reg = 3.5,
                   border = 0.5,
                   train_dir = 'train_dir')
model <- catboost.train(learn_pool, test_pool, fit_params)
```

### Predict and evaluate

```r
calc_accuracy <- function(prediction, expected) {
  labels <- ifelse(prediction > 0.5, 1, -1)
  accuracy <- sum(labels == expected) / length(labels)
  return(accuracy)
}

prediction <- catboost.predict(model, test_pool, type = 'Probability')
cat("Sample predictions: ", sample(prediction, 5), "\n")

labels <- catboost.predict(model, test_pool, type = 'Class')
table(labels, test[,target])

# works properly only for Logloss
accuracy <- calc_accuracy(prediction, test[,target])
cat("Accuracy: ", accuracy, "\n")

# feature splits importances (not finished)
cat(catboost.importance(model, learn_pool), "\n")
```
### Catboosting with caret
See [demo](demo/caret.R).
