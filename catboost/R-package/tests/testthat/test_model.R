require(testthat)
require(catboost)


test_that("model: catboost.train", {
  target <- sample(c(1, -1), size = 1000, replace = TRUE)
  features <- data.frame(f1 = rnorm(length(target), mean = 0, sd = 1),
                         f2 = rnorm(length(target), mean = 0, sd = 1))

  split <- sample(nrow(features), size = floor(0.75 * nrow(features)))

  pool_train <- catboost.from_data_frame(features[split,], target[split])
  pool_test <- catboost.from_data_frame(features[-split,], target[-split])

  iterations <- 10
  params <- list(iterations = iterations,
                 loss_function = "Logloss",
                 random_seed = 12345)

  model_train_test <- catboost.train(pool_train, pool_test, params)
  prediction_train_test <- catboost.predict(model_train_test, pool_test)

  model_train <- catboost.train(pool_train, NULL, params)
  prediction_train <- catboost.predict(model_train, pool_test)

  expect_equal(prediction_train_test, prediction_train)
})

test_that("model: catboost.importance", {
  target <- sample(c(1, -1), size = 1000, replace = TRUE)
  features <- data.frame(f1 = rnorm(length(target), mean = 0, sd = 1),
                         f2 = rnorm(length(target), mean = 0, sd = 1))

  pool <- catboost.from_data_frame(features, target)

  iterations <- 10
  params <- list(iterations = iterations,
                 loss_function = "Logloss",
                 random_seed = 12345)

  model <- catboost.train(pool, NULL, params, calc_importance = TRUE)
  expect_false(is.null(model$var_imp))
  expect_equal(length(model$var_imp), ncol(features))
})

test_that("model: catboost.train multiclass", {
  classes <- c(0, 1, 2)
  target <- sample(classes, size = 1000, replace = TRUE)
  features <- data.frame(f1 = rnorm(length(target), mean = 0, sd = 1),
                         f2 = rnorm(length(target), mean = 0, sd = 1))

  pool <- catboost.from_data_frame(features, target)
  params <- list(iterations = 10,
                 loss_function = "MultiClass",
                 random_seed = 12345)

  model <- catboost.train(pool, NULL, params)

  prediction <- catboost.predict(model, pool)
  expect_equal(catboost.nrow(pool), nrow(prediction))
  expect_equal(length(classes), ncol(prediction))
})

test_that("model: catboost.load_model", {
  target <- sample(c(1, -1), size = 1000, replace = TRUE)
  features <- data.frame(f1 = rnorm(length(target), mean = 0, sd = 1),
                         f2 = rnorm(length(target), mean = 0, sd = 1))

  pool <- catboost.from_data_frame(features, target)

  params <- list(iterations = 10,
                 loss_function = "Logloss",
                 random_seed = 12345)

  model <- catboost.train(pool, NULL, params)
  prediction <- catboost.predict(model, pool)

  model_path <- "catboost.model"
  catboost.save_model(model, model_path)

  loaded_model <- catboost.load_model(model_path)
  loaded_model_prediction <- catboost.predict(loaded_model, pool)

  expect_equal(prediction, loaded_model_prediction)
})

test_that("model: loss_function = multiclass", {
  target <- sample(c(0, 1, 2), size = 1000, replace = TRUE)
  data <- data.frame(f_numeric = target + rnorm(length(target), mean = 0, sd = 1),
                     f_logical = (target + rnorm(length(target), mean = 0, sd = 1)) > 0,
                     f_factor = as.factor(round(10 * (target + rnorm(length(target), mean = 0, sd = 1)))),
                     f_character = as.character(round(10 * (target + rnorm(length(target), mean = 0, sd = 1)))))

  data$f_logical = as.factor(data$f_logical)
  data$f_character = as.factor(data$f_character)

  pool <- catboost.from_data_frame(data, target)

  params <- list(iterations = 10,
                 loss_function = "MultiClass",
                 random_seed = 12345)

  model <- catboost.train(pool, NULL, params)
  prediction <- catboost.predict(model, pool, type = "Class")

  unique_prediction = unique(prediction)
  unique_target = unique(target)

  expect_equal(unique_target[order(unique_target)],
               unique_prediction[order(unique_prediction)])
})

test_that("model: baseline", {
  target <- sample(c(0, 1, 2), size = 1000, replace = TRUE)
  data <- data.frame(f_numeric = target + rnorm(length(target), mean = 0, sd = 1),
                     f_logical = (target + rnorm(length(target), mean = 0, sd = 1)) > 0,
                     f_factor = as.factor(round(10 * (target + rnorm(length(target), mean = 0, sd = 1)))),
                     f_character = as.character(round(10 * (target + rnorm(length(target), mean = 0, sd = 1)))))

  unique_target = unique(target)

  data$f_logical = as.factor(data$f_logical)
  data$f_character = as.factor(data$f_character)

  pool <- catboost.from_data_frame(data, target)

  params_baseline <- list(iterations = 3,
                          loss_function = "MultiClass",
                          random_seed = 12345)

  model_baseline <- catboost.train(pool, NULL, params_baseline)
  baseline <- catboost.predict(model_baseline, pool, type = "RawFormulaVal")

  expect_equal(nrow(baseline), catboost.nrow(pool))
  expect_equal(ncol(baseline), length(unique_target))

  pool_with_baseline <- catboost.from_data_frame(data, target, NULL, baseline)

  params <- list(iterations = 10,
                 loss_function = "MultiClass",
                 random_seed = 12345)

  model <- catboost.train(pool_with_baseline, NULL, params)

  prediction <- catboost.predict(model, pool_with_baseline, type = "Class")

  unique_prediction = unique(prediction)

  expect_equal(unique_target[order(unique_target)],
               unique_prediction[order(unique_prediction)])
})
