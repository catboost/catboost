require(jsonlite)

#' @useDynLib libcatboostr
NULL

#' catboost.load_pool
#'
#' Load the CatBoost dataset.
#' @param pool_path The path to the input file that contains the dataset description.
#' @param cd_path The path to the input file that contains the column descriptions.
#' @param thread_count The number of threads to use while reading the data. Optimizes reading time. This parameter doesn't affect results.
#' @param verbose Verbose output to stdout.
#'
#' @return catboost.Pool
#'
#' @examples
#' pool_path <- system.file("extdata", "adult_train.1000", package = "catboost")
#' test_pool_path <- system.file("extdata", "adult_test.1000", package = "catboost")
#' cd_path <- system.file("extdata", "adult.cd", package = "catboost")
#' pool <- catboost.load_pool(pool_path, cd_path)
#' test_pool <- catboost.load_pool(test_pool_path, cd_path)
#'
#' @export
#' @seealso \url{https://tech.yandex.com/catboost/doc/dg/concepts/r-reference_catboost-load-docpage/}
catboost.load_pool <- function(pool_path, cd_path = "", thread_count = 1, verbose = FALSE) {
  if (missing(pool_path))
        stop("Need to specify pool path.")
  if (!is.character(pool_path) || !is.character(cd_path))
        stop("Path must be a string.")

  pool <- .Call("CatBoostCreateFromFile_R", pool_path, cd_path, thread_count, verbose)
  attributes(pool) <- list(.Dimnames = list(NULL, NULL), class = "catboost.Pool")
  return(pool)
}

#' catboost.save_pool
#'
#' Save the dataset to the CatBoost format.
#' Files with the following data are created:
#' \itemize{
#' \item Dataset description
#' \item Column descriptions
#' }
#' Use the catboost.load_pool function to read the resulting files.
#' These files can also be used in the \href{https://tech.yandex.com/catboost/doc/dg/concepts/cli-installation-docpage/}{Command-line version} and the \href{https://tech.yandex.com/catboost/doc/dg/concepts/python-installation-docpage/}{Python library}.
#'
#' @param data A data.frame with features.
#' The following column types are supported:
#' \itemize{
#' \item double
#' \item factor.
#' It is assumed that categorical features are given in this type of columns.
#' A standard CatBoost processing procedure is applied to this type of columns:
#'   \describe{
#'   \item{1.}{The values are converted to strings.}
#'   \item{2.}{The ConvertCatFeatureToFloat function is applied to the resulting string.}
#'   }
#' }
#' @param target The target vector.
#' @param weight The weights of the target vector.
#' @param baseline Vector of initial (raw) values of the target function for the object.
#' Used in the calculation of final values of trees.
#' @param pool_path The path to the otuptut file that contains the dataset description.
#' @param cd_path The path to the output file that contains the column descriptions.
#'
#' @export
#' @seealso \url{https://tech.yandex.com/catboost/doc/dg/concepts/r-reference_catboost_save_pool-docpage/}
catboost.save_pool <- function(data, target, weight, baseline, pool_path, cd_path) {
  if (missing(pool_path) || missing(cd_path))
        stop("Need to specify pool_path and cd_path.")
  if (!is.character(pool_path) || !is.character(cd_path))
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
  write.table(column_description, file = cd_path, sep = "\t", row.names = FALSE, col.names= FALSE, quote = FALSE)
}

#' catboost.from_matrix
#'
#' Create a dataset from the given matrix.
#' Only numeric features are supported (their type should be double or factor).
#' Categorical features must be converted to numerical first.
#' For example, use \code{as.factor()} or the \code{colClasses} argument of the \code{read.table} method.
#' The target type should be double.
#'
#' @param data A matrix with features.
#' @param target The target vector.
#' @param weight The weights of the target vector.
#' @param baseline Vector of initial (raw) values of the target function for the object.
#' Used in the calculation of final values of trees.
#' @param cat_features A vector of categorical features indices.
#' The indices are zero based and can differ from the given in the Column descriptions file.
#'
#' @return catboost.Pool
#'
#' @examples
#' pool_path <- 'train_full3'
#' data <- read.table(pool_path, head = F, sep = "\t", colClasses = rep('numeric', 10))
#' target <- c(1)
#' cat_features <- seq(1,8)
#' data_matrix <- as.matrix(data)
#' pool <- catboost.from_matrix(data = as.matrix(data[,-target]), target = as.matrix(data[,target]), cat_features = cat_features)
#'
#' @export
#' @seealso \url{https://tech.yandex.com/catboost/doc/dg/concepts/r-reference_catboost-from_matrix-docpage/}
catboost.from_matrix <- function(data, target = NULL, weight = NULL, baseline = NULL, cat_features = NULL) {
  if (!is.matrix(data))
      stop("Unsupported data type, expecting matrix, got: ", class(data))
  if (!is.double(target) && !is.integer(target) && !is.null(target))
      stop("Unsupported target type, expecting double or int, got: ", typeof(target))
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

  if (!is.double(target) && !is.null(target))
      target <- as.double(target)

  pool <- .Call("CatBoostCreateFromMatrix_R", data, target, weight, baseline, cat_features)
  attributes(pool) <- list(.Names = colnames(data), class = "catboost.Pool")
  return(pool)
}

#' catboost.from_data_frame
#'
#' Create a dataset from the given data.frame.
#' Only numeric features are supported (their type should be double or factor).
#' Categorical features must be converted to numerical first.
#' For example, use \code{as.factor()} or the \code{colClasses} argument of the \code{read.table} method. The target type should be double.
#'
#' @param data A data.frame with features.
#' The following column types are supported:
#' \itemize{
#' \item double
#' \item factor.
#' It is assumed that categorical features are given in this type of columns.
#' A standard CatBoost processing procedure is applied to this type of columns:
#'   \describe{
#'   \item{1.}{The values are converted to strings.}
#'   \item{2.}{The ConvertCatFeatureToFloat function is applied to the resulting string.}
#'   }
#' }
#' @param target The target vector.
#' @param weight The weights of the target vector.
#' @param baseline Vector of initial (raw) values of the target function for the object.
#' Used in the calculation of final values of trees.
#'
#' @return catboost.Pool
#'
#' @examples
#' pool_path <- 'train_full3'
#' cd_vector <- c('numeric',  rep('numeric',2), rep('factor',7))
#' data <- read.table(pool_path, head = F, sep = "\t", colClasses = cd_vector)
#' target <- c(1)
#' learn_size <- floor(0.8 * nrow(data))
#' learn_ind <- sample(nrow(data), learn_size)
#' learn <- data[learn_ind,]
#' test <- data[-learn_ind,]
#' learn_pool <- catboost.from_data_frame(data = learn[,-target], target = learn[,target])
#' test_pool <- catboost.from_data_frame(data = test[,-target], target = test[,target])
#'
#' @export
#' @seealso \url{https://tech.yandex.com/catboost/doc/dg/concepts/r-reference_catboost-from_data_frame-docpage/}
catboost.from_data_frame <- function(data, target = NULL, weight = NULL, baseline = NULL) {
  if (!is.data.frame(data))
      stop("Unsupported data type, expecting data.frame, got: ", class(data))

  cat_features <- c()
  preprocessed <- data.frame(row.names = seq(1, nrow(data)))
  column_names <- colnames(data)
  for (column_index in c(1:ncol(data))) {
    if (any(is.na(data[, column_index]) || is.null(data[, column_index]))) {
      stop("NA and NULL values are not supported: ",  column_names[column_index])
    }
    if (is.double(data[, column_index])) {
      preprocessed[, column_names[column_index]] <- data[, column_index]
    }
    else if (is.integer(data[, column_index])) {
      preprocessed[, column_names[column_index]] <- as.integer(data[, column_index])
    }
    else if (is.factor(data[, column_index])) {
      preprocessed[, column_names[column_index]] <- .Call("CatBoostHashStrings_R", as.character(data[, column_index]))
      cat_features <- c(cat_features, column_index - 1)
    }
    else {
      stop("Unsupported column type: ", column_names[column_index], typeof(data[, column_index]))
    }
  }
  preprocessed <- as.matrix(preprocessed)
  pool <- catboost.from_matrix(preprocessed, target, weight, baseline, cat_features)
  return(pool)
}

#' catboost.nrow
#'
#' Calculate the number of objects in the dataset.
#' @param pool The input dataset.
#' @return The number of objects.
#' @export
#' @seealso \url{https://tech.yandex.com/catboost/doc/dg/concepts/r-reference_catboost-nrow-docpage/}
catboost.nrow <- function(pool) {
  if (class(pool) != "catboost.Pool")
      stop("Expected catboost.Pool, got: ", class(pool))
  num_row <- .Call("CatBoostPoolNumRow_R", pool)
  return(num_row)
}

#' catboost.ncol
#'
#' Calculate the number of columns in the dataset.
#' @param pool The input dataset.
#' @return The number of columns.
#' @export
#' @seealso \url{https://tech.yandex.com/catboost/doc/dg/concepts/r-reference_catboost-ncol-docpage/}
catboost.ncol <- function(pool) {
  if (class(pool) != "catboost.Pool")
      stop("Expected catboost.Pool, got: ", class(pool))
  num_col <- .Call("CatBoostPoolNumCol_R", pool)
  return(num_col)
}

#' catboost.ntrees
#'
#' Return the number of trees in the model.
#' @param model The model obtained as the result of training.
#' @return The number of trees.
#' @export
#' @seealso \url{https://tech.yandex.com/catboost/doc/dg/concepts/r-reference_catboost-ntrees-docpage/}
catboost.ntrees <- function(model) {
  if (class(model) != "catboost.Model")
      stop("Expected catboost.Model, got: ", class(pool))
  num_trees <- .Call("CatBoostPoolNumTrees_R", model$handle)
  return(num_trees)
}

#' catboost.head
#'
#' Return a list with the first n objects of the dataset.
#'
#' Each line of this list contains the following information for each object:
#' \itemize{
#'   \item The target value.
#'   \item The weight of the object.
#'   \item The feature values for the object.
#' }
#' @param pool The input dataset.
#'
#' Default value: Required argument
#' @param n The quantity of the first objects in the dataset to be returned.
#'
#' Default value: 10
#' @export
#' @seealso \url{https://tech.yandex.com/catboost/doc/dg/concepts/r-reference_catboost-head-docpage/}
catboost.head <- function(pool, n = 10) {
  if (class(pool) != "catboost.Pool")
      stop("Expected catboost.Pool, got: ", class(pool))
  result <- .Call("CatBoostPoolHead_R", pool, n)
  return(result)
}

#' catboost.train
#'
#' Train the model using a CatBoost dataset.
#'
#' The list of parameters
#'
#' \itemize{
#'   \item Common parameters
#'   \itemize{
#'     \item fold_permutation_block_size
#'
#'       Objects in the dataset are grouped in blocks before the random permutations.
#'       This parameter defines the size of the blocks.
#'       The smaller is the value, the slower is the training.
#'       Large values may result in quality degradation.
#'
#'       Default value:
#'
#'       Default value differs depending on the dataset size and ranges from 1 to 256 inclusively
#'     \item ignored_features
#'
#'       Identifiers of features to exclude from training.
#'       The non-negative indices that do not match any features are successfully ignored.
#'       For example, if five features are defined for the objects in the dataset and this parameter
#'       is set to “42”, the corresponding non-existing feature is successfully ignored.
#'
#'       The identifier corresponds to the feature's index.
#'       Feature indices used in train and feature importance are numbered from 0 to featureCount – 1.
#'       If a file is used as input data then any non-feature column types are ignored when calculating these
#'       indices. For example, each row in the input file contains data in the following order:
#'       categorical feature<\code{\t}>target value<\code{\t}> numerical feature. So for the row rock<\code{\t}>0 <\code{\t}>42,
#'       the identifier for the “rock” feature is 0, and for the “42” feature it's 1.
#'
#'       The identifiers of features to exclude should be enumerated at vector.
#'
#'       For example, if training should exclude features with the identifiers
#'       1, 2, 7, 42, 43, 44, 45, the value of this parameter should be set to c(1,2,7,42,43,44,45).
#'
#'       Default value:
#'
#'       None (use all features)
#'     \item use_best_model
#'
#'       If this parameter is set, the number of trees that are saved in the resulting model is defined as follows:
#'
#'       Build the number of trees defined by the training parameters.
#'       \itemize{
#'         \item Identify the iteration with the optimal loss function value.
#'         \item No trees are saved after this iteration.
#'       }
#'
#'       This option requires a test dataset to be provided.
#'
#'       Default value:
#'
#'       FALSE (not used)
#'     \item loss_function
#'
#'       The loss function (see \url{https://tech.yandex.com/catboost/doc/dg/concepts/loss-functions-docpage/#loss-functions})
#'       to use in training. The specified value also determines the machine learning problem to solve.
#'
#'       Format:
#'
#'       <Loss function 1>[:<parameter 1>=<value>:..<parameter N>=<value>:]
#'
#'       Supported loss functions:
#'       \itemize{
#'         \item 'RMSE'
#'         \item 'Logloss'
#'         \item 'MAE'
#'         \item 'CrossEntropy'
#'         \item 'Quantile'
#'         \item 'LogLinQuantile'
#'         \item 'MultiClass'
#'         \item 'MAPE'
#'         \item 'Poisson'
#'       }
#'
#'       Supported parameters:
#'       \itemize{
#'         \item alpha - The coefficient used in quantile-based losses ('Quantile' and 'LogLinQuantile'). The default value is 0.5.
#'
#'
#'        For example, if you need to calculate the value of Quantile with the coefficient \eqn{\alpha = 0.1}, use the following construction:
#'
#'        'Quantile:alpha=0.1'
#'       }
#'
#'       Default value:
#'
#'       'RMSE'
#'     \item custom_loss
#'
#'       Loss function (see \url{https://tech.yandex.com/catboost/doc/dg/concepts/loss-functions-docpage/#loss-functions})
#'       values to output during training.
#'       These functions are not optimized and are displayed for informational purposes only.
#'
#'       Format:
#'
#'       c(<Loss function 1>[:<parameter>=<value>],<Loss function 2>[:<parameter>=<value>],...,<Loss function N>[:<parameter>=<value>])
#'
#'       Supported loss functions:
#'       \itemize{
#'         \item 'RMSE'
#'         \item 'Logloss'
#'         \item 'MAE'
#'         \item 'CrossEntropy'
#'         \item 'Quantile'
#'         \item 'LogLinQuantile'
#'         \item 'MultiClass'
#'         \item 'MAPE'
#'         \item 'Poisson'
#'         \item 'Recall'
#'         \item 'Precision'
#'         \item 'AUC'
#'         \item 'Accuracy'
#'         \item 'R2'
#'       }
#'
#'       Supported parameters:
#'       \itemize{
#'         \item alpha - The coefficient used in quantile-based losses ('Quantile' and 'LogLinQuantile'). The default value is 0.5.
#'       }
#'
#'       For example, if you need to calculate the value of CrossEntropy and Quantile with the coefficient \eqn{\alpha = 0.1}, use the following construction:
#'
#'       c('CrossEntropy') or simply 'CrossEntropy'.
#'
#'       Values of all custom loss functions for learning and test datasets are saved to the Loss function
#'       (see \url{https://tech.yandex.com/catboost/doc/dg/concepts/output-data_error-functions-docpage/#output-data_error-functions})
#'       output files (learn_error.tsv and test_error.tsv respectively). The catalog for these files is specified in the train-dir (train_dir) parameter.
#'
#'       Default value:
#'
#'       None (use one of the loss functions supported by the library)
#'     \item eval_metric
#'
#'       The loss function used for overfitting detection (if enabled) and best model selection (if enabled).
#'
#'       Supported loss functions:
#'       \itemize{
#'         \item 'RMSE'
#'         \item 'Logloss'
#'         \item 'MAE'
#'         \item 'CrossEntropy'
#'         \item 'Quantile'
#'         \item 'LogLinQuantile'
#'         \item 'MultiClass'
#'         \item 'MAPE'
#'         \item 'Poisson'
#'         \item 'Recall'
#'         \item 'Precision'
#'         \item 'AUC'
#'         \item 'Accuracy'
#'         \item 'R2'
#'       }
#'
#'       Format:
#'
#'       metric_name:param=Value
#'
#'       Examples:
#'
#'       \code{'R2'}
#'
#'       \code{'Quantile:alpha=0.3'}
#'
#'       Default value:
#'
#'       Optimized objective is used
#'
#'     \item iterations
#'
#'       The maximum number of trees that can be built when solving machine learning problems.
#'
#'       When using other parameters that limit the number of iterations, the final number of trees may be less
#'       than the number specified in this parameter.
#'
#'       Default value:
#'
#'       500
#'
#'     \item border
#'
#'       The target border. If the value is strictly greater than this threshold,
#'       it is considered a positive class. Otherwise it is considered a negative class.
#'
#'       The parameter is obligatory if the Logloss function is used, since it uses borders to transform
#'       any given target to a binary target.
#'
#'       Used in binary classification.
#'
#'       Default value:
#'
#'       0.5
#'
#'     \item gradient_iterations
#'
#'       The number of gradient steps when calculating the values in leaves.
#'
#'       Default value:
#'
#'       1
#'
#'     \item depth
#'
#'       Depth of the tree.

#'       The value can be any integer up to 32. It is recommended to use values in the range [1; 10].
#'
#'       Default value:
#'
#'       6
#'     \item learning_rate
#'
#'       The learning rate.
#'
#'       Used for reducing the gradient step.
#'
#'       Default value:
#'
#'       0.03
#'
#'     \item rsm
#'
#'       Random subspace method. The percentage of features to use at each iteration of building trees. At each iteration, features are selected over again at random.
#'
#'       The value must be in the range [0;1].
#'
#'       Default value:
#'
#'       1
#'     \item random_seed
#'
#'       The random seed used for training.
#'
#'       Default value:
#'
#'       A new random value is selected on each run
#'     \item od_pval
#'
#'       Use the Overfitting detector (see \url{https://tech.yandex.com/catboost/doc/dg/concepts/overfitting-detector-docpage/#overfitting-detector})
#'       to stop training when the threshold is reached.
#'       Requires that a test dataset was input.
#'
#'       For best results, it is recommended to set a value in the range [10^-10; 10^-2].
#'
#'       The larger the value, the earlier overfitting is detected.
#'
#'       Default value:
#'
#'       The overfitting detection is turned off
#'
#'     \item leaf_estimation_method
#'
#'       The method used to calculate the values in leaves.
#'
#'       Possible values:
#'       \itemize{
#'         \item Newton
#'         \item Gradient
#'       }
#'
#'       Default value:
#'
#'       Default value depends on the selected loss function
#'
#'     \item l2_leaf_reg
#'
#'       L2 regularization coefficient. Used for leaf value calculation.
#'
#'       Any positive values are allowed.
#'
#'       Default value:
#'
#'       3
#'
#'     \item has_time
#'
#'       Use the order of objects in the input data (do not perform random permutations during the
#'       Transforming categorical features to numerical features (see \url{https://tech.yandex.com/catboost/doc/dg/concepts/algorithm-main-stages_cat-to-numberic-docpage/#algorithm-main-stages_cat-to-numberic})
#'       and Choosing the tree structure (see \url{https://tech.yandex.com/catboost/doc/dg/concepts/algorithm-main-stages_choose-tree-structure-docpage/#algorithm-main-stages_choose-tree-structure}) stages).
#'
#'       Default value:
#'
#'       FALSE (not used; generate random permutations)
#'
#'     \item priors
#'
#'       Use the specified priors during training.
#'
#'       Format:
#'
#'       c(<prior 1>,<prior 2>,...,<prior N>)
#'
#'       Default value:
#'
#'       c(0,0.5,1)
#'
#'     \item feature_priors
#'
#'       Specify individual priors for categorical features (used at the Transforming categorical
#'
#'       Given in the form of a comma-separated list of prior descriptions for each specified feature.
#'       The description for each feature contains a colon-separated feature index and prior values.
#'
#'       Format:
#'
#'       \code{с('<ID of feature 1>:<prior 1.1>:<prior 1.2>:...:<prior 1.N1>',...,'<ID of feature M>:<prior M.1>:<prior M.2>:...:<prior M.NM>')}
#'
#'       Default value:
#'
#'       Each feature uses the default priors
#'
#'     \item name
#'
#'       The experiment name to display in visualization tools (see \url{https://tech.yandex.com/catboost/doc/dg/concepts/visualization-docpage/#visualization}).
#'
#'       Default value:
#'
#'       experiment
#'
#'     \item prediction_type
#'
#'       The format for displaying approximated values in output data.
#'
#'       Possible values:
#'       \itemize{
#'         \item 'Probability'
#'         \item 'Class'
#'         \item 'RawFormulaVal'
#'       }
#'
#'       Default value:
#'
#'       \code{'RawFormulaVal'}
#'
#'     \item fold_len_multiplier
#'
#'       Coefficient for changing the length of folds.
#'
#'       The value must be greater than 1. The best validation result is achieved with minimum values.
#'
#'       With values close to 1 (for example, \eqn{1 + \epsilon}), each iteration takes a quadratic amount of memory and time
#'       for the number of objects in the iteration. Thus, low values are possible only when there is a small number of objects.
#'
#'       Default value:
#'
#'       2
#'
#'     \item class_weights
#'
#'       Classes weights. The values are used as multipliers for the object weights.
#'
#'       Classes are indexed from 0 to classes count – 1. For example, in case of binary classification the classes are indexed 0 and 1.
#'
#'       For examples:
#'
#'       \code{c(0.85, 1.2, 1)}
#'
#'       Default value:
#'
#'       None (the weight for all classes is set to 1)
#'
#'     \item classes_count
#'
#'       The upper limit for the numeric class label. Defines the number of classes for multiclassification.
#'
#'       Only non-negative integers can be specified. The given integer should be greater than any of the target
#'       values.
#'
#'       If this parameter is specified the labels for all classes in the input dataset should be smaller
#'       than the given value.
#'
#'       Default value:
#'
#'       maximum class label + 1
#'
#'     \item one_hot_max_size
#'
#'       Convert the feature to float if the number of different values that it takes exceeds the specified value. Ctrs are not calculated for such features.
#'
#'       The one-vs.-all delimiter is used for the resulting float features.
#'
#'       Default value:
#'
#'       FALSE
#'
#'       Do not convert features to float based on the number of different values
#'
#'     \item random_strength
#'
#'       Score standard deviation multiplier.
#'
#'        Default value:
#'
#'        1
#'   }
#'   \item CTR settings
#'   \itemize{
#'     \item ctr
#'
#'       Binarization settings for categorical features (see \url{https://tech.yandex.com/catboost/doc/dg/concepts/algorithm-main-stages_cat-to-numberic-docpage/#algorithm-main-stages_cat-to-numberic}).
#'
#'
#'       Format:
#'
#'       \code{c(<CTR type 1>:[<number of borders 1>:<Binarization type 1>],...,<CTR type N>:[<number of borders N>:<Binarization type N>])}
#'
#'       Components:
#'       \itemize{
#'         \item CTR types:
#'         \itemize{
#'           \item \code{'Borders'}
#'           \item \code{'Buckets'}
#'           \item \code{'MeanValue'}
#'           \item \code{'CounterTotal'}
#'           \item \code{'CounterMax'}
#'         }
#'         \item The number of borders for target binarization. (see \url{https://tech.yandex.com/catboost/doc/dg/concepts/binarization-docpage/#binarization})
#'         Only used for regression problems. Allowed values are integers from 1 to 255 inclusively. The default value is 1.
#'         \item The binarization (see \url{https://tech.yandex.com/catboost/doc/dg/concepts/binarization-docpage/#binarization})
#'         type for the target. Only used for regression problems.
#'
#'         Possible values:
#'         \itemize{
#'           \item \code{'Median'}
#'           \item \code{'Uniform'}
#'           \item \code{'UniformAndQuantiles'}
#'           \item \code{'MaxSumLog'}
#'           \item \code{'MinEntropy'}
#'           \item \code{'GreedyLogSum'}
#'         }
#'         By default, \code{'MinEntropy'}
#'       }
#'
#'       Default value:
#'
#'     \item ctr_border_count
#'
#'       The number of splits for categorical features.
#'
#'       Allowed values are integers from 1 to 255 inclusively.
#'
#'       Default value:
#'
#'       50
#'
#'     \item max_ctr_complexity
#'
#'       The maximum number of categorical features that can be combined.
#'
#'       Default value:
#'
#'       4
#'
#'     \item ctr_leaf_count_limit
#'
#'       The maximum number of leafs with categorical features.
#'       If the quantity exceeds the specified value a part of leafs is discarded.
#'
#'       The leafs to be discarded are selected as follows:
#'       \enumerate{
#'         \item The leafs are sorted by the frequency of the values.
#'         \item The top N leafs are selected, where N is the value specified in the parameter.
#'         \item All leafs starting from N+1 are discarded.
#'       }
#'
#'       This option reduces the resulting model size and the amount of memory required for training.
#'       Note that the resulting quality of the model can be affected.
#'
#'       Default value:
#'
#'       None
#'
#'       The number of leafs with categorical features is not limited
#'
#'     \item store_all_simple_ctr
#'
#'       Ignore categorical features, which are not used in feature combinations,
#'       when choosing candidates for exclusion.
#'
#'       Use this parameter with ctr-leaf-count-limit only.
#'
#'       Default value:
#'
#'       False
#'
#'       Both simple features and feature combinations are taken in account when limiting the number
#'       of leafs with categorical features
#'   }
#'   \item Binarization settings
#'   \itemize{
#'     \item  border_count
#'
#'       The number of splits for numerical features. Allowed values are integers from 1 to 255 inclusively.
#'
#'       Default value:
#'
#'       32
#'     \item feature_border_type
#'
#'       The binarization mode (see \url{https://tech.yandex.com/catboost/doc/dg/concepts/binarization-docpage/#binarization})
#'       for numerical features.
#'
#'       Possible values:
#'       \itemize{
#'         \item \code{'Median'}
#'         \item \code{'Uniform'}
#'         \item \code{'UniformAndQuantiles'}
#'         \item \code{'MaxSumLog'}
#'         \item \code{'MinEntropy'}
#'         \item \code{'GreedyLogSum'}
#'       }
#'
#'       Default value:
#'
#'       \code{'MinEntropy'}
#'   }
#'   \item Performance settings
#'   \itemize{
#'     \item thread_count
#'
#'       The number of threads to use when applying the model.
#'
#'       Allows you to optimize the speed of execution. This parameter doesn't affect results.
#'
#'       Default value:
#'
#'       Min(number of processor cores, 8)
#'   }
#'   \item Output settings
#'   \itemize{
#'     \item verbose
#'
#'       Verbose output to stdout.
#'
#'       Default value:
#'
#'       FALSE (not used)
#'
#'     \item train_dir
#'
#'       The directory for storing the files generated during training.
#'
#'       Default value:
#'
#'       None (current catalog)
#'
#'     \item snapshot_file
#'
#'       Settings for recovering training after an interruption (see
#'       \url{https://tech.yandex.com/catboost/doc/dg/concepts/snapshots-docpage/#snapshots}).
#'
#'       Depending on whether the file specified exists in the file system:
#'       \itemize{
#'         \item Missing – write information about training progress to the specified file.
#'         \item Exists – load data from the specified file and continue training from where it left off.
#'       }
#'
#'       Default value:
#'
#'       File can't be generated or read. If the value is omitted, the file name is experiment.cbsnapshot.
#'   }
#' }
#'
#' @param learn_pool The dataset used for training the model.
#'
#' Default value:
#' Required argument
#' @param test_pool The dataset used for testing the quality of the model.
#'
#' Default value:
#' NULL (not used)
#' @param params The list of parameters to start training with.
#'
#' If omitted, default values are used (see The list of parameters).
#'
#' If set, the passed list of parameters overrides the default values.
#'
#' Default value:
#' Required argument
#' @param calc_importance
#' Calculate the feature importance.
#'
#' If set to “TRUE” the resulting feature importance are saved as the default value for the catboost.importance function.
#'
#' Default value:
#' FALSE
#' @examples
#' fit_params <- list(iterations = 100,
#'   thread_count = 10,
#'   loss_function = 'Logloss',
#'   ignored_features = c(4,9),
#'   border_count = 32,
#'   depth = 5,
#'   learning_rate = 0.03,
#'   l2_leaf_reg = 3.5,
#'   border = 0.5,
#'   train_dir = 'train_dir')
#' model <- catboost.train(pool, test_pool, fit_params)
#' @export
#' @seealso \url{https://tech.yandex.com/catboost/doc/dg/concepts/r-reference_catboost-train-docpage/}
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

#' catboost.load_model
#'
#' Load the model from a file.
#'
#' Note: Feature importance (see \url{https://tech.yandex.com/catboost/doc/dg/concepts/fstr-docpage/#fstr}) is not saved when using this function.
#' @param model_path The path to the model.
#'
#' Default value: Required argument
#' @export
#' @seealso \url{https://tech.yandex.com/catboost/doc/dg/concepts/r-reference_catboost-load_model-docpage/}
catboost.load_model <- function(model_path) {
  handle <- .Call("CatBoostReadModel_R", model_path)
  model <- list(handle = handle, var_imp = NULL)
  class(model) <- "catboost.Model"
  return(model)
}

#' catboost.save_model
#'
#' Save the model to a file.
#'
#' Note: Feature importance (see \url{https://tech.yandex.com/catboost/doc/dg/concepts/fstr-docpage/#fstr}) is not saved when using this function.
#' @param model The model to be saved.
#'
#' Default value: Required argument
#' @param model_path The path to the resulting binary file with the model description.
#' Used for solving other machine learning problems (for instance, applying a model).
#'
#' Default value: Required argument
#' @export
#' @seealso \url{https://tech.yandex.com/catboost/doc/dg/concepts/r-reference_catboost-save_model-docpage/}
catboost.save_model <- function(model, model_path) {
  status <- .Call("CatBoostOutputModel_R", model$handle, model_path)
  return(status)
}

#' catboost.predict
#'
#' Apply the model to the given dataset.
#'
#' Peculiarities: In case of multiclassification the prediction is returned in the form of a matrix.
#' Each line of this matrix contains the predictions for one object of the input dataset.
#' @param model The model obtained as the result of training.
#'
#' Default value: Required argument
#' @param pool The input dataset.
#'
#' Default value: Required argument
#' @param verbose Verbose output to stdout.
#'
#' Default value: FALSE (not used)
#' @param type The format for displaying approximated values in output data
#' (see \url{https://tech.yandex.com/catboost/doc/dg/concepts/output-data-docpage/#output-data}).
#'
#' Possible values:
#' \itemize{
#'   \item 'Probability'
#'   \item 'Class'
#'   \item 'RawFormulaVal'
#' }
#'
#' Default value: 'RawFormulaVal'
#' @param tree_count_limit The number of trees from the model to use when applying. If specified, the first <value> trees are used.
#'
#' Default value: 0 (if value equals to 0 this parameter is ignored and all trees from the model are used)
#' @param thread_count The number of threads to use when applying the model.
#'
#' Allows you to optimize the speed of execution. This parameter doesn't affect results.
#'
#' Default value: 1
#' @export
#' @seealso \url{https://tech.yandex.com/catboost/doc/dg/concepts/r-reference_catboost-predict-docpage/}
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

#' catboost.importance
#'
#' Calculate the feature importances (see \url{https://tech.yandex.com/catboost/doc/dg/concepts/fstr-docpage/#fstr})
#' (Regular feature importance
#' (see \url{https://tech.yandex.com/catboost/doc/dg/concepts/output-data_feature-importance-docpage/#per-feature-importance})
#' and Feature interaction strength
#' (see \url{https://tech.yandex.com/catboost/doc/dg/concepts/output-data_feature-interaction-strength-docpage/#output-data_feature-interaction-strength})
#' ).
#'
#' @param model The model obtained as the result of training.
#'
#' Default value: Required argument
#' @param pool The input dataset.
#'
#' The feature importance for the training dataset is calculated if this argument is not specified.
#'
#' Default value: NULL
#' @param thread_count The number of threads to use when applying the model.
#'
#' Allows you to optimize the speed of execution. This parameter doesn't affect results.
#'
#' Default value: 1
#' @export
#' @seealso \url{https://tech.yandex.com/catboost/doc/dg/concepts/r-reference_catboost-importance-docpage/}
catboost.importance <- function(model, pool = NULL, thread_count = 1) {
  if (class(model) != "catboost.Model")
      stop("Expected catboost.Model, got: ", class(model))
  if (!is.null(pool) && class(pool) != "catboost.Pool")
      stop("Expected catboost.Pool, got: ", class(pool))
  if (is.null(pool)) {
    return(model$var_imp)
  }
  importance <- .Call("CatBoostCalcRegularFeatureEffect_R", model$handle, pool, thread_count)
  names(importance) <- attr(pool, '.Names')
  return(importance)
}

#' catboost.get_model_params
#'
#' Return the model parameters.
#'
#' @param model
#' The model obtained as the result of training.
#'
#' Default value: Required argument
#' @export
#' @seealso \url{https://tech.yandex.com/catboost/doc/dg/concepts/r-reference_catboost-get_model_params-docpage/}
catboost.get_model_params <- function(model) {
  if (class(model) != "catboost.Model")
      stop("Expected catboost.Model, got: ", class(pool))
  params <- .Call("CatBoostGetModelParams_R", model$handle)
  params <- jsonlite::fromJSON(params)
  return(params)
}
