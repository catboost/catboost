context("test_model.R")

test_that("model: catboost.train", {
  target <- sample(c(1, -1), size = 1000, replace = TRUE)
  features <- data.frame(f1 = rnorm(length(target), mean = 0, sd = 1),
                         f2 = rnorm(length(target), mean = 0, sd = 1))

  split <- sample(nrow(features), size = floor(0.75 * nrow(features)))

  pool_train <- catboost.load_pool(features[split,], target[split])
  pool_test <- catboost.load_pool(features[-split,], target[-split])

  iterations <- 10
  params <- list(iterations = iterations,
                 loss_function = "Logloss",
                 random_seed = 12345,
                 use_best_model = FALSE)

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

  pool <- catboost.load_pool(features, target)

  iterations <- 10
  params <- list(iterations = iterations,
                 loss_function = "Logloss",
                 random_seed = 12345)

  model <- catboost.train(pool, NULL, params)
  feature_importance <- model$feature_importances
  expect_equal(length(feature_importance), ncol(features))
  feature_importance_with_pool <- catboost.get_feature_importance(model, pool)
  expect_equal(feature_importance, feature_importance_with_pool)
})

test_that("model: catboost.train & catboost.predict multiclass", {
  classes <- c(0, 1, 2)
  target <- sample(classes, size = 1000, replace = TRUE)
  features <- data.frame(f1 = rnorm(length(target), mean = 0, sd = 1),
                         f2 = rnorm(length(target), mean = 0, sd = 1))

  pool <- catboost.load_pool(features, target)
  params <- list(iterations = 10,
                 loss_function = "MultiClass",
                 random_seed = 12345)

  model <- catboost.train(pool, NULL, params)

  prediction <- catboost.predict(model, pool)
  expect_equal(nrow(pool), nrow(prediction))
  expect_equal(length(classes), ncol(prediction))

  prediction_first <- catboost.predict(model, pool, ntree_start = 2, ntree_end = 4)
  prediction_second <- catboost.predict(model, pool, ntree_start = 2, ntree_end = 5)

  staged_preds = catboost.staged_predict(model, pool, ntree_start = 2, ntree_end = 5, eval_period = 2)

  expect_equal(prediction_first, staged_preds$nextElem())
  expect_equal(prediction_second, staged_preds$nextElem())

})

test_that("model: catboost.load_model", {
  target <- sample(c(1, -1), size = 1000, replace = TRUE)
  features <- data.frame(f1 = rnorm(length(target), mean = 0, sd = 1),
                         f2 = rnorm(length(target), mean = 0, sd = 1))

  pool <- catboost.load_pool(features, target)

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

  pool <- catboost.load_pool(data, target)

  params <- list(iterations = 10,
                 loss_function = "MultiClass",
                 random_seed = 12345)

  model <- catboost.train(pool, NULL, params)
  prediction <- catboost.predict(model, pool, prediction_type = "Class")

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

  pool <- catboost.load_pool(data, target)

  params_baseline <- list(iterations = 3,
                          loss_function = "MultiClass",
                          random_seed = 12345)

  model_baseline <- catboost.train(pool, NULL, params_baseline)
  baseline <- catboost.predict(model_baseline, pool, prediction_type = "RawFormulaVal")

  expect_equal(nrow(baseline), nrow(pool))
  expect_equal(ncol(baseline), length(unique_target))

  pool_with_baseline <- catboost.load_pool(data, target, baseline = baseline)

  params <- list(iterations = 10,
                 loss_function = "MultiClass",
                 random_seed = 12345)

  model <- catboost.train(pool_with_baseline, NULL, params)

  prediction <- catboost.predict(model, pool_with_baseline, prediction_type = "Class")

  unique_prediction = unique(prediction)

  expect_equal(unique_target[order(unique_target)],
               unique_prediction[order(unique_prediction)])
})

test_that("model: full_history", {
  target <- sample(c(0, 1, 2), size = 1000, replace = TRUE)
  data <- data.frame(f_numeric = target + rnorm(length(target), mean = 0, sd = 1),
                     f_logical = (target + rnorm(length(target), mean = 0, sd = 1)) > 0,
                     f_factor = as.factor(round(10 * (target + rnorm(length(target), mean = 0, sd = 1)))),
                     f_character = as.character(round(10 * (target + rnorm(length(target), mean = 0, sd = 1)))))

  unique_target = unique(target)

  data$f_logical = as.factor(data$f_logical)
  data$f_character = as.factor(data$f_character)

  pool <- catboost.load_pool(data, target)

  params <- list(iterations = 3,
                 loss_function = "Logloss",
                 random_seed = 12345,
                 approx_on_full_history = TRUE)
  model <- catboost.train(pool, NULL, params)
  pred <- catboost.predict(model, pool, prediction_type = "RawFormulaVal")

  expect_true(TRUE)
})

test_that("model: catboost.predict vs catboost.staged_predict", {
  pool_path <- system.file("extdata", "adult_train.1000", package="catboost")
  column_description_path <- system.file("extdata", "adult.cd", package="catboost")

  pool <- catboost.load_pool(pool_path, column_description = column_description_path)

  params <- list(iterations = 10,
                 loss_function = "Logloss")

  model <- catboost.train(pool, NULL, params)
  prediction_first <- catboost.predict(model, pool, ntree_start = 2, ntree_end = 4)
  prediction_second <- catboost.predict(model, pool, ntree_start = 2, ntree_end = 5)

  staged_preds = catboost.staged_predict(model, pool, ntree_start = 2, ntree_end = 5, eval_period = 2)

  expect_equal(prediction_first, staged_preds$nextElem())
  expect_equal(prediction_second, staged_preds$nextElem())
})

test_that("model: save/load by R", {
  train_path <- system.file("extdata", "adult_train.1000", package="catboost")
  test_path <- system.file("extdata", "adult_test.1000", package="catboost")
  cd_path <- system.file("extdata", "adult.cd", package="catboost")
  train_pool <- catboost.load_pool(train_path, column_description=cd_path)
  test_pool <- catboost.load_pool(test_path, column_description=cd_path)
  fit_params <- list(iterations=4, thread_count=1, loss_function='Logloss')

  model <- catboost.train(train_pool, params = fit_params)
  prediction <- catboost.predict(model, test_pool)

  save(model, file = 'tmp.rda')
  model <- NULL
  load('tmp.rda')

  prediction_after_save_load <- catboost.predict(model, test_pool)
  expect_equal(prediction, prediction_after_save_load)
})

test_that("model: saveRDS/readRDS by R", {
  train_path <- system.file("extdata", "adult_train.1000", package="catboost")
  test_path <- system.file("extdata", "adult_test.1000", package="catboost")
  cd_path <- system.file("extdata", "adult.cd", package="catboost")
  train_pool <- catboost.load_pool(train_path, column_description=cd_path)
  test_pool <- catboost.load_pool(test_path, column_description=cd_path)
  fit_params <- list(iterations=4, thread_count=1, loss_function='Logloss')

  model <- catboost.train(train_pool, params = fit_params)
  prediction <- catboost.predict(model, test_pool)

  saveRDS(model, 'tmp.rds')
  model <- readRDS('tmp.rds')

  prediction_after_save_load <- catboost.predict(model, test_pool)
  expect_equal(prediction, prediction_after_save_load)
})

test_that("model: catboost.cv", {
  target <- sample(c(1, -1), size = 1000, replace = TRUE)
  features <- data.frame(f1 = rnorm(length(target), mean = 0, sd = 1),
                         f2 = rnorm(length(target), mean = 0, sd = 1))

  pool <- catboost.load_pool(features, target)

  iterations <- 10
  params <- list(iterations = iterations,
                 loss_function = "Logloss",
                 random_seed = 12345,
                 use_best_model = FALSE)

  fold_count <- 5
  cv_result <- catboost.cv(pool = pool, params = params, fold_count = fold_count)
  print(cv_result)

  expect_true(all(cv_result$train.Logloss.std >= 0))
  expect_true(all(cv_result$test.Logloss.std >= 0))

  expect_true(all(cv_result$train.Logloss.mean >= 0))
  expect_true(all(cv_result$test.Logloss.mean >= 0))
})

