context("test_caret_parameter_tuning.R")

load_adult_pool <- function(name) {
  pool_path <- system.file("extdata", paste("adult_", name, ".1000", sep = ""), package = "catboost")

  column_description_vector <- rep("numeric", 15)
  cat_features <- c(3, 5, 7, 8, 9, 10, 11, 15) # same as extdata/adult.cd
  for (i in cat_features) {
    column_description_vector[i] <- "factor"
  }

  data <- read.table(pool_path, head = FALSE, sep = "\t", colClasses = column_description_vector, na.strings = "NAN")

  # Transform categorical features to numeric.
  for (i in cat_features) {
    data[, i] <- as.numeric(factor(data[, i]))
  }

  target <- c(1)
  # WS: unused variable data_matrix
  # data_matrix <- as.matrix(data)
  X <- as.data.frame(as.matrix(data[, -target]), stringsAsFactors = TRUE)
  y <- as.matrix(data[, target])
  return(list("X" = X, "y" = y))
}

test_that("test caret train and parameter tuning on adult pool", {
  data_train <- load_adult_pool("train")
  X_train <- data_train$X
  y_train <- data_train$y

  data_test <- load_adult_pool("test")
  test_pool <- catboost.load_pool(data_test$X, data_test$y)

  fit_control <- caret::trainControl(method = "cv", number = 5, classProbs = TRUE)

  grid <- expand.grid(
    depth = 4,
    learning_rate = 0.1,
    iterations = 10,
    l2_leaf_reg = 1e-3,
    rsm = 0.95,
    border_count = 64
  )

  report <- caret::train(
    X_train,
    as.factor(make.names(y_train)),
    method = catboost::catboost.caret,
    preProc = NULL,
    tuneGrid = grid,
    trControl = fit_control,
    test_pool = test_pool
  )

  expect_true(report$results$Accuracy > 0.75)
})
