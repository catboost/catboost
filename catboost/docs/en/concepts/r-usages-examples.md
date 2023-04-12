# Usage examples

## Load datasets {#load}

Load the [Dataset description in delimiter-separated values format](../concepts/input-data_values-file.md) and the [object descriptions](../concepts/input-data_column-descfile.md) from the `train` and `train.cd` files respectively (both stored in the current directory): 
```r
library(catboost)

pool_path = system.file("extdata", 
                        "adult_train.1000", 
                        package = "catboost")
column_description_path = system.file("extdata", 
                                      "adult.cd", 
                                      package = "catboost")
pool <- catboost.load_pool(pool_path, 
                           column_description = column_description_path)
head(pool, 1)[[1]]
```

Load the dataset from the {{ product }}{{ r-package }} (this dataset is a subset of the [Adult Data Set](http://archive.ics.uci.edu/ml/datasets/adult) distributed through the [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml)): 
```r
library(catboost)

pool_path = system.file("extdata", 
                        "adult_train.1000", 
                        package="catboost")

column_description_vector = rep('numeric', 15)
cat_features <- c(3, 5, 7, 8, 9, 10, 11, 15)
for (i in cat_features)
  column_description_vector[i] <- 'factor'

data <- read.table(pool_path, 
                   head = F, 
                   sep = "\t", 
                   colClasses = column_description_vector, 
                   na.strings='NAN')

# Transform categorical features to numerical
for (i in cat_features)
  data[,i] <- as.numeric(factor(data[,i]))

pool <- catboost.load_pool(as.matrix(data[,-target]),
                           label = as.matrix(data[,target]),
                           cat_features = cat_features - 2)
head(pool, 1)[[1]]
```

Load the dataset from data.frame: 
```r
library(catboost)

train_path = system.file("extdata", 
                         "adult_train.1000", 
                         package="catboost")
test_path = system.file("extdata", 
                        "adult_test.1000", 
                        package="catboost")

column_description_vector = rep('numeric', 15)
cat_features <- c(3, 5, 7, 8, 9, 10, 11, 15)
for (i in cat_features)
  column_description_vector[i] <- 'factor'

train <- read.table(train_path, 
                    head = F, 
                    sep = "\t", 
                    colClasses = column_description_vector, 
                    na.strings='NAN')
test <- read.table(test_path, 
                   head = F, 
                   sep = "\t", 
                   colClasses = column_description_vector, 
                   na.strings='NAN')
target <- c(1)
train_pool <- catboost.load_pool(data=train[,-target], 
                                 label = train[,target])
test_pool <- catboost.load_pool(data=test[,-target], 
                                label = test[,target])
head(train_pool, 1)[[1]]
head(test_pool, 1)[[1]]
```


## Train a model {#train}

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

Load a dataset with numerical features, define the training parameters and start the training on GPU:
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
                   loss_function = 'Logloss',
                   task_type = 'GPU')

pool = catboost.load_pool(dataset, label = label_values)

model <- catboost.train(pool, params = fit_params)
```

Load a dataset with numerical and categorical features, define the training parameters and start the training:
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
                   l2_leaf_reg = 3.5)

pool = catboost.load_pool(dataset, label = label_values, cat_features = c(0,3))

model <- catboost.train(pool, params = fit_params)
```

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


## Apply the model {#apply-the-model}

Apply the model to the given dataset using the {{ prediction-type--RawFormulaVal }} output type for calculating the approximated values of the formula:

```r
library(catboost)

prediction <- catboost.predict(model, 
                               pool, 
                               prediction_type = 'RawFormulaVal')
```


## Return the first n objects of the dataset {#head}

Return the first 10 objects of the dataset:
```r
library(catboost)

head(pool, n = 10)
```


## Select hyperparameters {#selecting-hyperparameters}

Return the identifier of the iteration with the best result of the evaluation metric or loss function on the last validation set

```r
library(caret)
library(titanic)
library(catboost)

set.seed(12345)

data <- as.data.frame(as.matrix(titanic_train), stringsAsFactors = TRUE)

drop_columns = c("PassengerId", "Survived", "Name", "Ticket", "Cabin")
x <- data[,!(names(data) %in% drop_columns)]
y <- data[,c("Survived")]

fit_control <- trainControl(method = "cv",
                            number = 4,
                            classProbs = TRUE)

grid <- expand.grid(depth = c(4, 6, 8),
                    learning_rate = 0.1,
                    iterations = 100,
                    l2_leaf_reg = 1e-3,
                    rsm = 0.95,
                    border_count = 64)

report <- train(x, as.factor(make.names(y)),
                method = catboost.caret,
                logging_level = 'Verbose', preProc = NULL,
                tuneGrid = grid, trControl = fit_control)

print(report)

importance <- varImp(report, scale = FALSE)
print(importance)
```


## Calculate object strength {#ostr__r}

Calculate the object strength:
```r
library(catboost)

train_dataset = matrix(c(1900,7,1,
                         1896,1,1),
                        nrow=2, 
                        ncol=3, 
                        byrow = TRUE)

label_values = c(0, 1)

train_pool = catboost.load_pool(train_dataset, 
                                label_values)

input_dataset = matrix(c(1900,47,1,
                         1904,27,1),
                 nrow=2, 
                 ncol=3, 
                 byrow = TRUE)

input_pool = catboost.load_pool(input_dataset, 
                                label_values)

trained_model <- catboost.train(train_pool,
                               params = list(iterations = 10))

object_importance <- catboost.get_object_importance(trained_model,
                                                    input_pool,
                                                    train_pool)
```

