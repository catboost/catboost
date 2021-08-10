context("test_pool.R")

train_and_predict <- function(pool_train, pool_test, iterations, params) {
  catboost_model <- catboost.train(pool_train, pool_test, params = params)
  prediction <- catboost.predict(catboost_model, pool_test,
                                 prediction_type = "Probability",
                                 ntree_end = iterations)
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
  names(pool)[as.integer(column_description$index[column_description$type == "Label"])] <- "Label"
  for (i in column_description$index[column_description$type == "Categ"]) {
    if (is.factor(pool[, i]) == FALSE) {
      pool[, i] <- as.factor(pool[, i])
    }
  }
  return(pool)
}

almost_equal <- function(tensor_a, tensor_b, tol=1e-5) {
    a_nan_mask = is.nan(tensor_a)
    b_nan_mask = is.nan(tensor_b)
    if (!identical(a_nan_mask, b_nan_mask)) {
        return(FALSE)
    }
    return(max(abs(tensor_a[!a_nan_mask] - tensor_b[!b_nan_mask])) < tol)
}

test_that("pool: load_pool from file multitarget", {
  train_path <- system.file("extdata", "multitarget.train", package = "catboost")
  cd_path <- system.file("extdata", "multitarget.cd", package = "catboost")

  first_pool <- catboost.load_pool(train_path, column_description = cd_path)

  data <- read.table(train_path, head = FALSE, sep = "\t", colClasses = rep("numeric", 5), na.strings = "NAN")
  target_idx <- c(1, 2)

  second_pool <- catboost.load_pool(data = data[, -target_idx], label = as.matrix(data[, target_idx]))

  expect_true(almost_equal(head(first_pool, nrow(first_pool)), head(second_pool, nrow(second_pool))))
})

test_that("pool: load_pool from matrix multitarget", {
  target <- as.matrix(
    data.frame(      
      sample(c(1, -1), size = 1000, replace = TRUE),
      sample(c(1, -1), size = 1000, replace = TRUE)
    )
  )
  f1 <- target[, 1] + rnorm(nrow(target), mean = 0, sd = 1)
  f2 <- target[, 2] + rbinom(nrow(target), 5, prob = 0.5)

  features <- data.frame(f1 = f1, f2 = f2)

  split <- sample(nrow(features), size = floor(0.75 * nrow(features)))

  pool_train <- catboost.load_pool(as.matrix(features[split, ]), target[split,])
  pool_test <- catboost.load_pool(as.matrix(features[-split, ]), target[-split,])

  expect_equal(ncol(pool_train), ncol(features))
  expect_equal(nrow(pool_train), length(split))
  expect_equal(nrow(pool_test), nrow(target) - length(split))
})

test_that("pool: load_pool from matrix", {
  target <- sample(c(1, -1), size = 1000, replace = TRUE)

  f1 <- target + rnorm(length(target), mean = 0, sd = 1)
  f2 <- target + rbinom(length(target), 5, prob = 0.5)

  features <- data.frame(f1 = f1, f2 = f2)

  split <- sample(nrow(features), size = floor(0.75 * nrow(features)))

  cat_features <- c(1)

  pool_train <- catboost.load_pool(as.matrix(features[split, ]), target[split], cat_features = cat_features)
  pool_test <- catboost.load_pool(as.matrix(features[-split, ]), target[-split], cat_features = cat_features)

  expect_equal(ncol(pool_train), ncol(features))
  expect_equal(nrow(pool_train), length(split))
  expect_equal(nrow(pool_test), length(target) - length(split))

  iterations <- 10
  params <- list(iterations = iterations,
                 loss_function = "Logloss",
                 allow_writing_files = FALSE)

  prediction <- train_and_predict(pool_train, pool_test, iterations, params)
  accuracy <- calc_accuracy(prediction, target[-split])

  expect_true(accuracy > 0)
})

test_that("pool: load_pool from data.frame", {
  target <- sample(c(1, -1), size = 1000, replace = TRUE)

  features <- data.frame(f_numeric = target + rnorm(length(target), mean = 0, sd = 1),
                         f_logical = (target + rnorm(length(target), mean = 0, sd = 1)) > 0,
                         f_factor = as.factor(round(10 * (target + rnorm(length(target), mean = 0, sd = 1)))),
                         f_character = as.character(round(10 * (target + rnorm(length(target), mean = 0, sd = 1)))))

  features$f_logical <- as.factor(features$f_logical)
  features$f_character <- as.factor(features$f_character)

  pool <- catboost.load_pool(features, target)

  expect_equal(nrow(pool), nrow(features))

  iterations <- 10
  params <- list(iterations = iterations,
                 loss_function = "Logloss",
                 allow_writing_files = FALSE)

  prediction <- train_and_predict(pool, pool, iterations, params)
  accuracy <- calc_accuracy(prediction, target)

  expect_true(accuracy > 0)
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

  pool <- catboost.load_pool(data, target)

  params <- list(iterations = 10,
                 loss_function = "MultiClass",
                 allow_writing_files = FALSE)

  model <- catboost.train(pool, NULL, params)
  prediction <- catboost.predict(model, pool)

  weights <- rep(1, nrow(data))
  baseline <- rep(0, nrow(data) * length(unique(target)))

  catboost.save_pool(data, target, weights, baseline, pool_path, column_description_path)

  loaded_pool <- catboost.load_pool(pool_path, column_description = column_description_path)
  loaded_pool_prediction <- catboost.predict(model, loaded_pool)

  expect_equal(prediction, loaded_pool_prediction)

  unlink(pool_path)
  unlink(column_description_path)
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
                 loss_function = "Logloss",
                 allow_writing_files = FALSE)

  count <- table(target)
  weights <- ifelse(target == -1, 1 / count[1], 1 / count[2])

  pool <- catboost.load_pool(data, target, weight = weights)

  model <- catboost.train(pool, NULL, params)
  prediction <- catboost.predict(model, pool)
  expect_equal(length(prediction), nrow(pool))
})

test_that("pool: nan", {
  train_path <- system.file("extdata", "adult_train.1000", package = "catboost")
  cd_path <- system.file("extdata", "adult.cd", package = "catboost")

  first_pool <- catboost.load_pool(train_path, column_description = cd_path)

  column_description_vector <- rep("numeric", 15)
  cd <- read.table(cd_path)
  cat_features <- cd[cd[, 2] == "Categ", 1] + 1

  for (i in cat_features)
      column_description_vector[i] <- "factor"

  train <- read.table(train_path, head = FALSE, sep = "\t", colClasses = column_description_vector, na.strings = "NAN")
  target <- c(1)
  second_pool <- catboost.load_pool(data = train[, -target], label = train[, target])

  expect_true(identical(matrix(unlist(head(first_pool, nrow(first_pool))), nrow = nrow(first_pool), byrow = TRUE),
                        matrix(unlist(head(second_pool, nrow(second_pool))), nrow = nrow(second_pool), byrow = TRUE)))
})

test_that("pool: data.frame vs tibble::tbl_df vs pool", {
  pool_path <- system.file("extdata", "adult_train.1000", package = "catboost")
  column_description_path <- system.file("extdata", "adult.cd", package = "catboost")

  data_frame <- load_data_frame(pool_path, column_description_path)
  data_frame_pool <- catboost.load_pool(data_frame[, -which(names(data_frame) == "Label")],
                                                                                      as.double(data_frame$Label))
  data_frame_test_pool <- catboost.load_pool(data_frame[, -which(names(data_frame) == "Label")])

  tbl_df_pool <- catboost.load_pool(tibble::as_tibble(data_frame[, -which(names(data_frame) == "Label")]),
                                                                                  as.double(data_frame$Label))
  tbl_df_test_pool <- catboost.load_pool(tibble::as_tibble(data_frame[, -which(names(data_frame) == "Label")]))

  params <- list(iterations = 10,
                                  loss_function = "Logloss", allow_writing_files = FALSE)

  model <- catboost.train(data_frame_pool, NULL, params)
  model_tbl_df <- catboost.train(tbl_df_pool, NULL, params)

  pool <- catboost.load_pool(pool_path, column_description = column_description_path)

  head_pool <- head(pool)
  head_data_frame_pool <- head(data_frame_pool)
  head_tbl_df_pool <- head(tbl_df_pool)

  expect_equal(head_pool, head_data_frame_pool)
  expect_equal(head_pool, head_tbl_df_pool)

  prediction <- catboost.predict(model, pool)
  data_frame_prediction <- catboost.predict(model, data_frame_pool)
  data_frame_test_predicion <- catboost.predict(model, data_frame_test_pool)
  tbl_df_prediction <- catboost.predict(model, tbl_df_pool)
  tbl_df_test_predicion <- catboost.predict(model, tbl_df_test_pool)

  expect_equal(prediction, data_frame_prediction)
  expect_equal(prediction, data_frame_test_predicion)
  expect_equal(prediction, tbl_df_prediction)
  expect_equal(prediction, tbl_df_test_predicion)
})

test_that("bad params handled correctly", {
  pool_path <- system.file("extdata", "adult_train.1000", package = "catboost")
  cd_path <- system.file("extdata", "adult.cd", package = "catboost")
  label <- list(1, 2, 3)
  expect_error(catboost.load_pool(pool_path, column_description = cd_path, label = label), ".*should be NULL.*")
})

test_that("pool: text features in data.frame", {
  dfTrain <- data.frame(f_numeric = c(150, 120, 30), f_factor = factor(c('m', 'f', 'm')),
                        text_feature = c('hello good I am good I hello good',
                                        'good I hello I am good hello','bad bad bad bad'),
                        target=c(0, 0, 1))
  features <- dfTrain[, !(names(dfTrain) %in% c('target'))]
  target <- dfTrain[, c('target')]
  dfTest <- data.frame(f_numeric = c(150, 10), f_factor = factor(c('m', 'f')),
                        f_text = c('hello I hello I hello good hello good I hello good',
                                   'bad bad bad bad bad'))
  train_pool <- catboost.load_pool(features, target)

  expect_equal(nrow(train_pool), nrow(features))

  iterations <- 10
  params <- list(iterations = iterations,
                 loss_function = "Logloss",
                 allow_writing_files = FALSE)

  prediction <- train_and_predict(train_pool, train_pool, iterations, params)
  accuracy <- calc_accuracy(prediction, target)

  expect_true(accuracy > 0)
})