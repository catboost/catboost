install.packages("caret", repos="https://cloud.r-project.org/")
require(catboost)
require(caret)

load_pool <- function() {
  pool_path = system.file("extdata", "adult_train.1000", package="catboost")

  column_description_vector = rep('numeric', 15)
  cat_features <- c(3, 5, 7, 8, 9, 10, 11, 15)
  for (i in cat_features) {
    column_description_vector[i] <- 'factor'
  }

  data <- read.table(pool_path, head=F, sep="\t", colClasses=column_description_vector, na.strings='NAN')

  # Transform categorical features to numeric.
  for (i in cat_features) {
    data[,i] <- as.numeric(factor(data[,i]))
  }

  target <- c(1)
  data_matrix <- as.matrix(data)
  X <- as.data.frame(as.matrix(data[,-target]), stringsAsFactors=TRUE)
  y <- as.matrix(data[,target])
  return(list('X' = X, 'y' = y))
}

test_that("test caret train and parameter tuning on adult pool", {
  data <- load_pool()
  X <- data$X
  y <- data$y

  fit_control <- caret::trainControl(method="cv",
                                     number=5,
                                     classProbs=TRUE)

  grid <- expand.grid(depth=4,
                      learning_rate=0.1,
                      iterations=10,
                      l2_leaf_reg=1e-3,
                      rsm=0.95,
                      border_count=64)

  report <- caret::train(X, as.factor(make.names(y)),
                         method=catboost::catboost.caret, preProc=NULL,
                         tuneGrid=grid, trControl=fit_control)

  expect_true(report$results$Accuracy > 0.75)
})
