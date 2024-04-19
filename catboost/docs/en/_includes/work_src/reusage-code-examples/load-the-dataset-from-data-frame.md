
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
