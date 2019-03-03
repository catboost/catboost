context("test_on_trimmed_adult_dataset.R")

load_adult_pool <- function() {
  pool.path <- system.file("extdata", "adult_train.1000", package = "catboost")
  column_description.path <- system.file("extdata", "adult.cd", package = "catboost")
  pool <- catboost::catboost.load_pool(pool.path, column_description = column_description.path)
  pool
}

test_that("load adult pool", {
  load_adult_pool()
  expect_true(TRUE)
})

test_that("train on adult pool", {
  pool <- load_adult_pool()
  fit_params <- list(
    iterations = 100,
    loss_function = "Logloss"
  )

  model <- catboost.train(pool, pool, fit_params)
  expect_true(TRUE)
})
