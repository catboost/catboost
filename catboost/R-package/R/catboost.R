require(jsonlite)

#' @useDynLib libcatboostr
NULL


#' Load the catboost pool
#' @param pool_path A path to the catboost pool file (First 4 columns has special meaning. Colums after 5-th contain features).
#' @param column_description_path Feature descriptor file path.
#' @param thread_count Number of threads used while reading data.
#' @param verbose Log output.
#' @export
catboost.load_pool <- function(pool_path, column_description_path = "", thread_count = 1, verbose = FALSE) {
  if (missing(pool_path))
        stop("Need to specify pool path.")
  if (!is.character(pool_path) || !is.character(column_description_path))
        stop("Path must be a string.")

  pool <- .Call("CatBoostCreateFromFile_R", pool_path, column_description_path, thread_count, verbose)
  attributes(pool) <- list(.Dimnames = list(NULL, NULL), class = "catboost.Pool")
  return(pool)
}

#' Writes data.frame in pool format
#' @param data Features data.frame
#' @param target Target vector
#' @param weights Weights vector
#' @param baseline Baseline vector
#' @param pool_path Pool file path
#' @param column_description_path Column description file path
#' @export
catboost.save_pool <- function(data, target, weight, baseline, pool_path, column_description_path) {
  if (missing(pool_path) || missing(column_description_path))
        stop("Need to specify pool_path and column_description_path.")
  if (!is.character(pool_path) || !is.character(column_description_path))
        stop("Path must be a string.")

  pool <- target
  column_description <- c("Target")
  if (is.null(weight) == FALSE) {
    pool <- cbind(pool, weight)
    column_description <- c(column_description, "Weight")
  }
  if (is.null(baseline) == FALSE) {
    baseline <- matrix(baseline, nrow = nrow(data), byrow = TRUE)
    pool <- cbind(pool, baseline)
    column_description <- c(column_description, rep("Baseline", ncol(baseline)))
  }
  pool <- cbind(pool, data)
  column_description <- data.frame(index = seq(0, length(column_description) - 1), type = column_description)
  factors <- which(sapply(data, class) == "factor")
  if (length(factors) != 0) {
    column_description <- rbind(column_description, data.frame(index = nrow(column_description) + factors - 1,
                               type = rep("Categ", length(factors))))
  }
  write.table(pool, file = pool_path, sep = "\t", row.names = FALSE, col.names= FALSE, quote = FALSE)
  write.table(column_description, file = column_description_path, sep = "\t", row.names = FALSE, col.names= FALSE, quote = FALSE)
}

#' Create pool from matrix
#' @param data Feature matrix
#' @param target Target vector
#' @param weight Weight vector
#' @param baseline Baseline vector
#' @param cat_features Vector of categorical feature indices
#' @export
catboost.from_matrix <- function(data, target = NULL, weight = NULL, baseline = NULL, cat_features = NULL) {
  if (!is.matrix(data))
      stop("Unsupported data type, expecting matrix, got: ", class(data))
  if (!is.double(target) && !is.null(target))
      stop("Unsupported target type, expecting double, got: ", typeof(target))
  if (length(target) != nrow(data) && !is.null(target))
      stop("Data has ", nrow(data), " rows, target has ", length(target), " rows.")
  if (!is.double(weight) && !is.null(weight))
      stop("Unsupported weight type, expecting double, got: ", typeof(target))
  if (length(weight) != nrow(data) && !is.null(weight))
      stop("Data has ", nrow(data), " rows, weight vector has ", length(weight), " rows.")
  if (!is.matrix(baseline) && !is.null(baseline))
      stop("Baseline should be matrix, got: ", class(baseline))
  if (!is.double(baseline) && !is.null(baseline))
      stop("Unsupported baseline type, expecting double, got: ", typeof(baseline))
  if (nrow(baseline) != nrow(data) && !is.null(baseline))
      stop("Baseline must be matrix of size n_objects*n_classes. Data has ", nrow(data), " objects, baseline has ", nrow(baseline), " rows.")
  if (!all(cat_features == as.integer(cat_features)) && !is.null(cat_features))
      stop("Unsupported cat_features type, expecting integer, got: ", typeof(cat_features))

  pool <- .Call("CatBoostCreateFromMatrix_R", data, target, weight, baseline, cat_features)
  attributes(pool) <- list(.Dimnames = list(NULL, colnames(data)), class = "catboost.Pool")
  return(pool)
}

#' Create pool from data frame
#' @param data data.frame object
#' @param target Target vector
#' @export
catboost.from_data_frame <- function(data, target = NULL, weight = NULL, baseline = NULL) {
  if (!is.data.frame(data))
      stop("Unsupported data type, expecting data.frame, got: ", class(data))

  cat_features <- c()
  preprocessed <- data.frame(row.names = seq(1, nrow(data)))
  column_names <- colnames(data)
  for (column_index in seq(1, length(column_names))) {
    if (any(is.na(data[, column_index]) || is.null(data[, column_index]))) {
      stop("NA and NULL values are not supported.")
    }
    if (is.double(data[, column_index])) {
      preprocessed[, column_names[column_index]] <- data[, column_index]
    }
    else if (is.factor(data[, column_index])) {
      preprocessed[, column_names[column_index]] <- .Call("CatBoostHashStrings_R", as.character(data[, column_index]))
      cat_features <- c(cat_features, column_index - 1)
    }
    else {
      stop("Unsupported column type: ", typeof(data[, column_index]))
    }
  }
  pool <- catboost.from_matrix(as.matrix(preprocessed), target, weight, baseline, cat_features)
  return(pool)
}

#' Returns number of objects in a pool
#' @export
catboost.nrow <- function(pool) {
  if (class(pool) != "catboost.Pool")
      stop("Expected catboost.Pool, got: ", class(pool))
  num_row <- .Call("CatBoostPoolNumRow_R", pool)
  return(num_row)
}

#' Returns number of columns in a pool
#' @export
catboost.ncol <- function(pool) {
  if (class(pool) != "catboost.Pool")
      stop("Expected catboost.Pool, got: ", class(pool))
  num_col <- .Call("CatBoostPoolNumCol_R", pool)
  return(num_col)
}

#' Returns number of trees in a model
#' @export
catboost.ntrees <- function(model) {
  if (class(model) != "catboost.Model")
      stop("Expected catboost.Model, got: ", class(pool))
  num_trees <- .Call("CatBoostPoolNumTrees_R", model$handle)
  return(num_trees)
}

#' Returns top n objects from pool
#' @export
catboost.head <- function(pool, n = 10) {
  if (class(pool) != "catboost.Pool")
      stop("Expected catboost.Pool, got: ", class(pool))
  result <- .Call("CatBoostPoolHead_R", pool, n)
  return(result)
}

#' Train the catboost model on catboost pool
#' @export
catboost.train <- function(learn_pool, test_pool = NULL, params = list(), calc_importance = FALSE) {
  if (class(learn_pool) != "catboost.Pool")
      stop("Expected catboost.Pool, got: ", class(learn_pool))
  if (class(test_pool) != "catboost.Pool" && !is.null(test_pool))
      stop("Expected catboost.Pool, got: ", class(test_pool))
  if (length(params) == 0)
      message("Training catboost with default parameters! See help(catboost.train).")

  json_params <- jsonlite::toJSON(params, auto_unbox = TRUE)
  handle <- .Call("CatBoostFit_R", learn_pool, test_pool, json_params)
  model <- list(handle = handle)
  class(model) <- "catboost.Model"
  if (calc_importance) {
    model$var_imp <- catboost.importance(model, learn_pool)
  }
  return(model)
}

#' Loads model from file
#' @export
catboost.load_model <- function(model_path) {
  handle <- .Call("CatBoostReadModel_R", model_path)
  model <- list(handle = handle, var_imp = NULL)
  class(model) <- "catboost.Model"
  return(model)
}

#' Writes model to file
#' @export
catboost.save_model <- function(model, model_path) {
  status <- .Call("CatBoostOutputModel_R", model$handle, model_path)
  return(status)
}

#' Get model prediction
#' @param model trained model
#' @param pool data to make predictions on
#' @param verbose output log while making prediction
#' @param type 'Probability', 'Class' or 'RawFormulaVal'
#' @param tree_count_limit tree number to make prediction on
#' @param thread_count number of parallel threads
#' @export
catboost.predict <- function(model, pool,
                             verbose = FALSE, type = 'RawFormulaVal',
                             tree_count_limit = 0, thread_count = 1) {
  if (class(model) != "catboost.Model")
      stop("Expected catboost.Model, got: ", class(model))
  if (class(pool) != "catboost.Pool")
      stop("Expected catboost.Pool, got: ", class(pool))

  params <- catboost.get_model_params(model)
  prediction <- .Call("CatBoostPredictMulti_R", model$handle, pool,
                      verbose, type, tree_count_limit, thread_count)
  prediction_columns <- length(prediction) / catboost.nrow(pool)
  if (prediction_columns != 1) {
    prediction <- matrix(prediction, ncol = prediction_columns, byrow = TRUE)
  }
  return(prediction)
}

#' Estimate feature importance
#' @param model Trained model
#' @param pool Pool to estimate importance on
#' @param thread_count Number of parallel threads
#' @export
catboost.importance <- function(model, pool = NULL, thread_count = 1) {
  if (class(model) != "catboost.Model")
      stop("Expected catboost.Model, got: ", class(model))
  if (!is.null(pool) && class(pool) != "catboost.Pool")
      stop("Expected catboost.Pool, got: ", class(pool))
  if (is.null(pool)) {
    return(model$var_imp)
  }
  importance <- .Call("CatBoostCalcRegularFeatureEffect_R", model$handle, pool, thread_count)
  return(importance)
}

#' Returns model params
#' @export
catboost.get_model_params <- function(model) {
  if (class(model) != "catboost.Model")
      stop("Expected catboost.Model, got: ", class(pool))
  params <- .Call("CatBoostGetModelParams_R", model$handle)
  params <- jsonlite::fromJSON(params)
  return(params)
}
