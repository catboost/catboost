
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
