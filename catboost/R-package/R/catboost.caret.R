#' Support caret interface
#' @export
catboost.caret <- list(label = "Catboost",
                       library = "catboost",
                       type = c("Regression", "Classification"))

#' Define tuning parameters
catboost.caret$parameters <- data.frame(parameter = c("depth",
                                                      "learning_rate",
                                                      "iterations",
                                                      "l2_leaf_reg",
                                                      "rsm",
                                                      "border_count"),
                                        class = c("numeric",
                                                  "numeric",
                                                  "numeric",
                                                  "numeric",
                                                  "numeric",
                                                  "numeric"),
                                        label = c("Tree Depth",
                                                  "Learning rate",
                                                  "Number of trees",
                                                  "L2 regularization coefficient",
                                                  "The percentage of features to use at each iteration",
                                                  "The number of splits for numerical features"))

#' Init tuning param values
#' @param x, y: the current data used to fit the model
#' @param len: the value of tuneLength that is potentially passed in through train
#' @param search: can be either "grid" or "random"
catboost.caret$grid <- function(x, y, len = 5, search = "grid") {
  if (search == "grid") {
    grid <- expand.grid(depth = c(2, 4, 6),
                        learning_rate = exp(-(0:len)),
                        iterations = 100,
                        l2_leaf_reg = 1e-6,
                        rsm = 0.9,
                        border_count = 255)
  }
  else {  # search == "random"
    grid <- data.frame(depth = sample.int(len, len, replace = TRUE),
                       learning_rate = runif(len, min = 1e-6, max = 1),
                       iterations = rep(100, len),
                       l2_leaf_reg = sample(c(1e-1, 1e-3, 1e-6), len, replace = TRUE),
                       rsm = sample(c(1., 0.9, 0.8, 0.7), len, replace = TRUE),
                       border_count = sample(c(255), len, replace = TRUE))
  }
  return(grid)
}

#' Fit model based on input data
#' @param x, y: the current data used to fit the model
#' @param wts: optional instance weights (not applicable for this particular model)
#' @param param: the current tuning parameter values
#' @param lev: the class levels of the outcome (or NULL in regression)
#' @param last: a logical for whether the current fit is the final fit
#' @param weights
#' @param classProbs: a logical for whether class probabilities should be computed.
catboost.caret$fit <- function(x, y, wts, param, lev, last, weights, classProbs, ...) {
  param <- c(param, list(...))
  if (is.null(param$loss_function)) {
    param$loss_function = "RMSE"
    if (is.factor(y)) {
      param$loss_function = "Logloss"
      if (length(lev) > 2) {
        param$loss_function = "MultiClass"
      }
      y = as.double(y) - 1
    }
  }
  pool <- catboost.from_data_frame(x, y, wts)
  model <- catboost.train(pool, NULL, param, calc_importance = TRUE)
  model$lev <- lev
  return(model)
}

#' Returns model predictions
#' @param modelFit: the model produced by the fit code shown above.
#' @param newdata: the predictor values of the instances being predicted (e.g. out-of-bag samples)
#' @param preProc: preprcess data option
#' @param submodels: only used with the loop element
catboost.caret$predict <- function(modelFit, newdata, preProc = NULL, submodels = NULL) {
  pool <- catboost.from_data_frame(newdata)
  param <- catboost.get_model_params(modelFit)
  pred_type <- if (modelFit$problemType == 'Regression') 'RawFormulaVal' else 'Class'
  prediction <- catboost.predict(modelFit, pool, type = pred_type)
  if (!is.null(modelFit$lev) && !is.na(modelFit$lev)) {
    prediction <- modelFit$lev[prediction + 1]
  }
  return(prediction)
}

#' Predict class probabillties
#' @param modelFit: the model produced by the fit code shown above
#' @param newdata: the predictor values of the instances being predicted (e.g. out-of-bag samples)
#' @param preProc: preprcess data option
#' @param submodels: only used with the loop element
catboost.caret$prob <- function(modelFit, newdata, preProc = NULL, submodels = NULL) {
  pool <- catboost.from_data_frame(newdata)
  prediction <- catboost.predict(modelFit, pool, type = "Probability")
  if (is.matrix(prediction)) {
    colnames(prediction) <- modelFit$lev
    prediction <- as.data.frame(prediction)
  }
  param <- catboost.get_model_params(modelFit)
  if (param$loss_function == "Logloss") {
    prediction <- cbind(1 - prediction, prediction)
    colnames(prediction) <- modelFit$lev
  }
  return(prediction)
}

#' Calculates variable importance metrics for the model
#' @param modelFit: the model produced by the fit code shown above
#' @param x, y: the current data used to fit the model
catboost.caret$varImp <- function(modelFit, x = NULL, y = NULL, ...) {
  pool <- NULL
  if (!is.null(x) && !is.null(y)) {
    pool <- catboost.from_data_frame(x, y)
  }
  importance <- catboost.importance(modelFit, pool)
  importance <- as.data.frame(importance)
  colnames(importance) <- "Overall"
  return(importance)
}
