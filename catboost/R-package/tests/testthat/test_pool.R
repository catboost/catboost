require(testthat)
require(catboost)


train_and_predict <- function(pool_train, pool_test, iterations, params) {
  catboost_model <- catboost.train(pool_train, pool_test, params = params)
  prediction <- catboost.predict(catboost_model, pool_test,
                                 type = "Probability",
                                 tree_count_limit = iterations)
  return(prediction)
}

calc_accuracy <- function(prediction, expected) {
  labels <- ifelse(prediction > 0.5, 1, -1)
  accuracy <- sum(labels == expected) / length(labels)
  return(accuracy)
}

load_data_frame <- function(pool_path, column_description_path) {
  column_description <- read.csv(column_description_path, sep = "\t", col.names = c("index", "type"), header = F)
  pool <- read.csv(pool_path, sep = "\t", header = F)
  column_description$index <- column_description$index + 1
  names(pool)[as.integer(column_description$index[column_description$type == "Target"])] <- "Target"
  for (i in column_description$index[column_description$type == "Categ"]) {
    if (is.factor(pool[, i]) == FALSE) {
      pool[, i] <- as.factor(pool[, i])
    }
  }
  return(pool)
}

test_that("pool: catboost.from_matrix", {
  target <- sample(c(1, -1), size = 1000, replace = TRUE)

  f1 <- target + rnorm(length(target), mean = 0, sd = 1)
  f2 <- target + rbinom(length(target), 5, prob = 0.5)

  features <- data.frame(f1 = f1, f2 = f2)

  split <- sample(nrow(features), size = floor(0.75 * nrow(features)))

  cat_features <- c(1)

  pool_train <- catboost.from_matrix(as.matrix(features[split,]), target[split], NULL, NULL, cat_features)
  pool_test <- catboost.from_matrix(as.matrix(features[-split,]), target[-split], NULL, NULL, cat_features)

  expect_equal(catboost.ncol(pool_train), ncol(features))
  expect_equal(catboost.nrow(pool_train), length(split))
  expect_equal(catboost.nrow(pool_test), length(target) - length(split))

  iterations <- 10
  params <- list(iterations = iterations,
                 loss_function = "Logloss")

  prediction <- train_and_predict(pool_train, pool_test, iterations, params)
  accuracy <- calc_accuracy(prediction, target[-split])

  expect_true(accuracy > 0)
})

test_that("pool: catboost.from_data_frame", {
  target <- sample(c(1, -1), size = 1000, replace = TRUE)

  features <- data.frame(f_numeric = target + rnorm(length(target), mean = 0, sd = 1),
                         f_logical = (target + rnorm(length(target), mean = 0, sd = 1)) > 0,
                         f_factor = as.factor(round(10 * (target + rnorm(length(target), mean = 0, sd = 1)))),
                         f_character = as.character(round(10 * (target + rnorm(length(target), mean = 0, sd = 1)))))

  features$f_logical = as.factor(features$f_logical)
  features$f_character = as.factor(features$f_character)

  pool <- catboost.from_data_frame(features, target)

  expect_equal(catboost.nrow(pool), nrow(features))

  iterations <- 10
  params <- list(iterations = iterations,
                 loss_function = "Logloss")

  prediction <- train_and_predict(pool, pool, iterations, params)
  accuracy <- calc_accuracy(prediction, target)

  expect_true(accuracy > 0)
})

test_that("pool: data.frame vs pool", {
  pool_path <- system.file("extdata", "adult_train.1000", package="catboost")
  column_description_path <- system.file("extdata", "adult.cd", package="catboost")

  data_frame <- load_data_frame(pool_path, column_description_path)
  data_frame_pool <- catboost.from_data_frame(data_frame[, -which(names(data_frame) == "Target")],
                                              as.double(data_frame$Target))
  data_frame_test_pool <- catboost.from_data_frame(data_frame[, -which(names(data_frame) == "Target")])

  params <- list(iterations = 10,
                 loss_function = "Logloss")

  model <- catboost.train(data_frame_pool, NULL, params)

  pool <- catboost.load_pool(pool_path, column_description_path)

  head_pool <- catboost.head(pool)
  head_data_frame_pool <- catboost.head(data_frame_pool)
  expect_equal(head_pool, head_data_frame_pool)

  prediction <- catboost.predict(model, pool)
  data_frame_prediction <- catboost.predict(model, data_frame_pool)
  data_frame_test_predicion <- catboost.predict(model, data_frame_test_pool)

  expect_equal(prediction, data_frame_prediction)
  expect_equal(prediction, data_frame_test_predicion)
})

test_that("pool: catboost.save_pool", {
  target <- sample(c(1, 2, 3), size = 1000, replace = TRUE)
  data <- data.frame(f_numeric = target + rnorm(length(target), mean = 0, sd = 1),
                     f_logical = (target + rnorm(length(target), mean = 0, sd = 1)) > 0,
                     f_factor = as.factor(round(10 * (target + rnorm(length(target), mean = 0, sd = 1)))),
                     f_character = as.character(round(10 * (target + rnorm(length(target), mean = 0, sd = 1)))))

  data$f_logical <- as.factor(data$f_logical)
  data$f_character <- as.factor(data$f_character)

  pool_path <- "test_pool.tmp"
  column_description_path <- "test_cd.tmp"

  pool <- catboost.from_data_frame(data, target)

  params <- list(iterations = 10,
                 loss_function = "MultiClass")

  model <- catboost.train(pool, NULL, params)
  prediction <- catboost.predict(model, pool)

  weights <- rep(1, nrow(data))
  baseline <- rep(0, nrow(data) * length(unique(target)))

  catboost.save_pool(data, target, weights, baseline, pool_path, column_description_path)

  loaded_pool <- catboost.load_pool(pool_path, column_description_path)
  loaded_pool_prediction <- catboost.predict(model, loaded_pool)

  expect_equal(prediction, loaded_pool_prediction)
})

test_that("pool: data.frame weights", {
  target <- sample(c(-1, 1), size = 1000, replace = TRUE, prob = c(.75, .25))
  data <- data.frame(f_numeric = target + rnorm(length(target), mean = 0, sd = 1),
                     f_logical = (target + rnorm(length(target), mean = 0, sd = 1)) > 0,
                     f_factor = as.factor(round(10 * (target + rnorm(length(target), mean = 0, sd = 1)))),
                     f_character = as.character(round(10 * (target + rnorm(length(target), mean = 0, sd = 1)))))

  data$f_logical <- as.factor(data$f_logical)
  data$f_character <- as.factor(data$f_character)

  params <- list(iterations = 10,
                 loss_function = "Logloss")

  count <- table(target)
  weights <- ifelse(target == -1, 1 / count[1], 1 / count[2])

  pool <- catboost.from_data_frame(data, target, weights)

  model <- catboost.train(pool, NULL, params)
  prediction <- catboost.predict(model, pool)
  expect_equal(length(prediction), catboost.nrow(pool))
})
