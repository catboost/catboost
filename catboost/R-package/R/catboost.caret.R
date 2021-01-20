#' @name catboost.caret
#' @title Support caret interface
#'
#' @export
catboost.caret <- list(label = "Catboost",
                       library = "catboost",
                       type = c("Regression", "Classification"))


#' Define tuning parameters
#' @noRd
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
#'
#' @param x, y: the current data used to fit the model
#' @param len: the value of tuneLength that is potentially passed in through train
#' @param search: can be either "grid" or "random"
#'
#' @noRd
catboost.caret$grid <- function(x, y, len = 5, search = "grid") {
  if (search == "grid") {
    grid <- expand.grid(depth = c(2, 4, 6),
                        learning_rate = exp(- (0:len)),
                        iterations = 100,
                        l2_leaf_reg = 1e-6,
                        rsm = 0.9,
                        border_count = 255)
  }
  else {
    # search == "random"
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
#'
#' @param x, y: the current data used to fit the model
#' @param wts: optional instance weights (not applicable for this particular model)
#' @param param: the current tuning parameter values
#' @param lev: the class levels of the outcome (or NULL in regression)
#' @param last: a logical for whether the current fit is the final fit
#' @param weights: weights
#' @param classProbs: a logical for whether class probabilities should be computed
#'
#' @noRd
catboost.caret$fit <- function(x, y, wts, param, lev, last, weights, classProbs, ...) {
    param <- c(param, list(...))
    if (is.null(param$loss_function)) {
        param$loss_function <- "RMSE"
        if (is.factor(y)) {
            param$loss_function <- "Logloss"
            if (length(lev) > 2) {
                param$loss_function <- "MultiClass"
            }
            y <- as.double(y) - 1
        }
    }
    test_pool <- NULL
    if (!is.null(param$test_pool)) {
        test_pool <- param$test_pool
        if (class(test_pool) != "catboost.Pool")
            stop("Expected catboost.Pool, got: ", class(test_pool))
        param <- within(param, rm(test_pool))
    }
    pool <- catboost.from_data_frame(x, y, weight = wts)
    model <- catboost.train(pool, test_pool, param)
    model$lev <- lev
    return(model)
}


#' Returns model predictions
#'
#' @param modelFit: the model produced by the fit code shown above.
#' @param newdata: the predictor values of the instances being predicted (e.g. out-of-bag samples)
#' @param preProc: preprcess data option
#' @param submodels: only used with the loop element
#'
#' @noRd
catboost.caret$predict <- function(modelFit, newdata, preProc = NULL, submodels = NULL) {
    pool <- catboost.from_data_frame(newdata)
    # WS: unused variable param
    # param <- catboost.get_model_params(modelFit)
    pred_type <- if (modelFit$problemType == "Regression") "RawFormulaVal" else "Class"
    prediction <- catboost.predict(modelFit, pool, prediction_type = pred_type)
    if (!is.null(modelFit$lev) && !any(is.na(modelFit$lev))) {
        prediction <- modelFit$lev[prediction + 1]
    }
    if (!is.null(submodels)) {
        tmp <- vector(mode = "list", length = nrow(submodels) + 1)
        tmp[[1]] <- prediction
        for (j in seq(along = submodels$iterations)) {
            tmp_pred <- catboost.predict(modelFit, pool, prediction_type = pred_type, ntree_end = submodels$iterations[j])
            if (!is.null(modelFit$lev) && !any(is.na(modelFit$lev))) {
                tmp_pred <- modelFit$lev[tmp_pred + 1]
            }
            tmp[[j + 1]]  <- tmp_pred
        }
        prediction <- tmp
    }

    return(prediction)
}


#' Predict class probabilities
#'
#' @param modelFit: the model produced by the fit code shown above
#' @param newdata: the predictor values of the instances being predicted (e.g. out-of-bag samples)
#' @param preProc: preprcess data option
#' @param submodels: only used with the loop element
#'
#' @noRd
catboost.caret$prob <- function(modelFit, newdata, preProc = NULL, submodels = NULL) {
    pool <- catboost.from_data_frame(newdata)
    prediction <- catboost.predict(modelFit, pool, prediction_type = "Probability")
    if (is.matrix(prediction)) {
        colnames(prediction) <- modelFit$lev
        prediction <- as.data.frame(prediction)
    }
    param <- catboost.get_model_params(modelFit)
    if (param$loss_function$type == "Logloss") {
        prediction <- cbind(1 - prediction, prediction)
        colnames(prediction) <- modelFit$lev
    }

    if (!is.null(submodels)) {
        tmp <- vector(mode = "list", length = nrow(submodels) + 1)
        tmp[[1]] <- prediction
        for (j in seq(along = submodels$iterations)) {
            tmp_pred <- catboost.predict(modelFit, pool, prediction_type = "Probability", ntree_end = submodels$iterations[j])
            if (is.matrix(tmp_pred)) {
                colnames(tmp_pred) <- modelFit$lev
                tmp_pred <- as.data.frame(tmp_pred)
            }
            param <- catboost.get_model_params(modelFit)
            if (param$loss_function$type == "Logloss") {
                tmp_pred <- cbind(1 - tmp_pred, tmp_pred)
                colnames(tmp_pred) <- modelFit$lev
            }
            tmp[[j + 1]]  <- tmp_pred
        }
        prediction <- tmp
    }
    return(prediction)
}


#' Calculates variable importance metrics for the model
#'
#' @param modelFit: the model produced by the fit code shown above
#' @param x, y: the current data used to fit the model
#'
#' @noRd
catboost.caret$varImp <- function(modelFit, x = NULL, y = NULL, ...) {
    pool <- NULL
    if (!is.null(x) && !is.null(y)) {
        pool <- catboost.from_data_frame(x, y)
    }
    importance <- catboost.get_feature_importance(modelFit, pool)
    importance <- as.data.frame(importance)
    colnames(importance) <- "Overall"
    return(importance)
}


#' Create multiple submodel predictions from the same object.
#'
#' @param grid: the grid of parameters to search over.
#'
#' @noRd
catboost.caret$loop <- function(grid) {
    loop <- plyr::ddply(grid, c("depth",
                                "learning_rate",
                                "l2_leaf_reg",
                                "rsm",
                                "border_count"),
                        function(x) c(iterations = max(x$iterations)))
    submodels <- vector(mode = "list", length = nrow(loop))
    for (i in seq(along = loop$iterations)) {
        index <- which(grid$depth == loop$depth[i] &
                       grid$learning_rate == loop$learning_rate[i] &
                       grid$l2_leaf_reg == loop$l2_leaf_reg[i] &
                       grid$rsm == loop$rsm[i] &
                       grid$border_count == loop$border_count[i])
        trees <- grid[index, "iterations"]
        submodels[[i]] <- data.frame(iterations = trees[trees != loop$iterations[i]])
    }
    return(list(loop = loop, submodels = submodels))
}
