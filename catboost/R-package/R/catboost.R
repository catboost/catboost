#' @import jsonlite
#' @importFrom utils head
#' @importFrom utils tail
#' @importFrom utils write.table
#' @importFrom utils download.file
#' @useDynLib libcatboostr, .registration = TRUE
NULL


#' @name catboost.load_pool
#' @title Create a dataset
#'
#' @description Create a dataset from the given file, matrix or data.frame.
#'
#' @param data A file path, matrix or data.frame with features.
#' The following column types are supported:
#' \itemize{
#'     \item double
#'     \item factor.
#'     It is assumed that categorical features are given in this type of columns.
#'     A standard CatBoost processing procedure is applied to this type of columns:
#'     \describe{
#'         \item{1.}{The values are converted to strings.}
#'         \item{2.}{The ConvertCatFeatureToFloat function is applied to the resulting string.}
#'     }
#' }
#'
#' Default value: Required argument
#' @param label The label vector or label matrix
#' @param cat_features A vector of categorical features indices.
#' The indices are zero based and can differ from the given in the Column descriptions file.
#' If data parameter is data.frame don't use cat_features, categorical features are determined automatically
#' from data.frame column types.
#' @param column_description The path to the input file that contains the column descriptions.
#' @param pairs A file path, matrix or data.frame that contains the pairs descriptions. The shape should be Nx2, where N is the pairs' count.
#' The first element of pair is the index of winner document in training set. The second element of pair is the index of loser document in training set.
#' @param delimiter Delimiter character to use to separate features in a file.
#' @param has_header Read column names from first line, if this parameter is set to True.
#' @param weight The weights of the objects.
#' @param group_id The group ids of the objects.
#' @param group_weight The group weight of the objects.
#' @param subgroup_id The subgroup ids of the objects.
#' @param pairs_weight The weights of the pairs.
#' @param baseline Vector of initial (raw) values of the objective function.
#' Used in the calculation of final values of trees.
#' @param feature_names A list of names for each feature in the dataset.
#' @param thread_count The number of threads to use while reading the data. Optimizes reading time. This parameter doesn't affect results.
#' If -1, then the number of threads is set to the number of CPU cores.
#'
#' @examples
#' \dontrun{
#' # From file
#' pool_path <- system.file("extdata", "adult_train.1000", package = "catboost")
#' cd_path <- system.file("extdata", "adult.cd", package = "catboost")
#' pool <- catboost.load_pool(pool_path, column_description = cd_path)
#' print(pool)
#'
#' # From matrix
#' target <- 1
#' data_matrix <-matrix(runif(18), 6, 3)
#' pool <- catboost.load_pool(data_matrix[, -target], label = data_matrix[, target])
#' print(pool)
#'
#' # From data.frame
#' nonsense <- factor(c('A', 'B', 'C'))
#' data_frame <- data.frame(value = runif(10), category = nonsense[(1:10) %% 3 + 1])
#' label = (1:10) %% 2
#' pool <- catboost.load_pool(data_frame, label = label)
#' print(pool)
#' }
#' @return catboost.Pool
#' @export
catboost.load_pool <- function(data, label = NULL, cat_features = NULL, column_description = NULL,
                               pairs = NULL, delimiter = "\t", has_header = FALSE, weight = NULL,
                               group_id = NULL, group_weight = NULL, subgroup_id = NULL, pairs_weight = NULL,
                               baseline = NULL, feature_names = NULL, thread_count = -1) {
    if (!is.null(pairs) && (is.character(data) != is.character(pairs))) {
        stop("Data and pairs should be the same types.")
    }

    if (is.character(data) && length(data) == 1) {
        for (arg in list("label", "cat_features", "weight", "group_id",
                         "group_weight", "subgroup_id", "pairs_weight",
                         "baseline")) {
            if (!is.null(get(arg))) {
                stop("parameter '", arg, "' should be NULL when the pool is read from file")
            }
        }
        pool <- catboost.from_file(data, column_description, pairs, delimiter, has_header, thread_count, FALSE, feature_names)
    } else if (is.matrix(data)) {
        pool <- catboost.from_matrix(data, label, cat_features, NULL, NULL, pairs, weight, group_id, group_weight, subgroup_id, pairs_weight,
                                     baseline, feature_names)
    } else if (is.data.frame(data)) {
        for (arg in list("column_description")) {
            if (!is.null(get(arg))) {
                stop("Parameter '", arg, "' should be NULL when the pool is constructed from data.frame")
            }
        }
        if (!is.null(get("cat_features"))) {
            cat("Parameter 'cat_features' is meaningless because column types are taken from data.frame.",
                "Please, convert categorical columns to factors manually.", sep = "\n")
        }
        pool <- catboost.from_data_frame(data, label, pairs, weight, group_id, group_weight, subgroup_id, pairs_weight,
                                         baseline, feature_names)
    } else {
        stop("Unsupported data type, expecting string, matrix or dafa.frame, got: ", class(data))
    }
    return(pool)
}


catboost.from_file <- function(pool_path, cd_path = "", pairs_path = "", delimiter = "\t", has_header = FALSE,
                               thread_count = -1, verbose = FALSE, feature_names_path = "",
                               num_vector_delimiter = ';') {
    if (missing(pool_path))
        stop("Need to specify pool path.")
    if (is.null(pairs_path))
        pairs_path <- ""
    if (is.null(feature_names_path))
        feature_names_path <- ""
    if (!is.character(pool_path) || !is.character(cd_path) || !is.character(pairs_path) || !is.character(feature_names_path))
        stop("Path must be a string.")

    pool_path <- path.expand(pool_path)
    cd_path <- path.expand(cd_path)

    pool <- .Call("CatBoostCreateFromFile_R", pool_path, cd_path, pairs_path, feature_names_path, delimiter, num_vector_delimiter, has_header, thread_count, verbose)
    attributes(pool) <- list(.Dimnames = list(NULL, NULL), class = "catboost.Pool")
    return(pool)
}


catboost.from_matrix <- function(float_and_cat_features_data, label = NULL, cat_features_indices = NULL, text_features_data = NULL,
                                 text_features_indices = NULL, pairs = NULL, weight = NULL, group_id = NULL, group_weight = NULL,
                                 subgroup_id = NULL, pairs_weight = NULL, baseline = NULL, feature_names = NULL) {
  if (!is.matrix(float_and_cat_features_data))
      stop("Unsupported data type, expecting matrix, got: ", class(float_and_cat_features_data))

  float_and_cat_columns <- if (is.null(float_and_cat_features_data)) 0 else ncol(float_and_cat_features_data)
  text_columns <- if (is.null(text_features_data)) 0 else ncol(text_features_data)
  data_columns <- float_and_cat_columns + text_columns
  if (text_columns == 0 && float_and_cat_columns == 0)
      stop("Data has no columns")

  if (!is.null(label) && !is.matrix(label))
      label <- as.matrix(label)
  if (!is.double(label) && !is.integer(label) && !is.null(label))
      stop("Unsupported label type, expecting double or int, got: ", typeof(label))
  if (!is.null(label) && !is.double(label))
      label <- matrix(as.double(as.double(label)), nrow=nrow(label), ncol=ncol(label))
  if (!is.null(label) && nrow(label) != nrow(float_and_cat_features_data))
      stop("Data has ", nrow(float_and_cat_features_data), " rows, label has ", nrow(label), " rows.")

  if (!all(cat_features_indices == as.integer(cat_features_indices)) && !is.null(cat_features_indices))
      stop("Unsupported cat_features type, expecting integer, got: ", typeof(cat_features_indices))

  if (!is.null(text_features_data) && !is.matrix(text_features_data))
     stop("Unsupported text data type, expecting matrix, got: ", class(text_features_data))
  if (!all(text_features_indices == as.integer(text_features_indices)) && !is.null(text_features_indices))
     stop("Unsupported text_features_indices type, expecting integer, got: ", typeof(text_features_indices))

  if (!is.matrix(pairs) && !is.null(pairs))
      stop("Unsupported pairs class, expecting matrix, got: ", class(pairs))
  if (!is.null(pairs) && dim(pairs)[2] != 2)
      stop("Unsupported pairs dim, expecting 2 columns, got: ", dim(pairs)[2])
  if (!all(pairs == as.integer(pairs)) && !is.null(pairs))
      stop("Unsupported pair type, expecting integer, got: ", typeof(pairs))

  if (!is.double(weight) && !is.null(weight))
      stop("Unsupported weight type, expecting double, got: ", typeof(weight))
  if (length(weight) != nrow(float_and_cat_features_data) && !is.null(weight))
      stop("Data has ", nrow(float_and_cat_features_data), " rows, weight vector has ", length(weight), " rows.")

  if (!is.integer(group_id) && !is.null(group_id))
      stop("Unsupported group_id type, expecting int, got: ", typeof(group_id))
  if (length(group_id) != nrow(float_and_cat_features_data) && !is.null(group_id))
      stop("Data has ", nrow(float_and_cat_features_data), " rows, group_id vector has ", length(group_id), " rows.")

  if (!is.double(group_weight) && !is.null(group_weight))
      stop("Unsupported group_weight type, expecting double, got: ", typeof(group_weight))
  if (length(group_weight) != nrow(float_and_cat_features_data) && !is.null(group_weight))
      stop("Data has ", nrow(float_and_cat_features_data), " rows, group_weight vector has ", length(group_weight), " rows.")

  if (!is.integer(subgroup_id) && !is.null(subgroup_id))
      stop("Unsupported subgroup_id type, expecting int, got: ", typeof(subgroup_id))
  if (length(subgroup_id) != nrow(float_and_cat_features_data) && !is.null(subgroup_id))
      stop("Data has ", nrow(float_and_cat_features_data), " rows, subgroup_id vector has ", length(subgroup_id), " rows.")

  if (!is.double(pairs_weight) && !is.null(pairs_weight))
      stop("Unsupported pairs_weight type, expecting double, got: ", typeof(pairs_weight))
  if (length(pairs_weight) != nrow(pairs) && !is.null(pairs_weight) && !is.null(pairs))
      stop("Pairs has ", nrow(pairs), " rows, pairs_weight vector has ", length(pairs_weight), " rows.")

  if (!is.matrix(baseline) && !is.null(baseline))
      stop("Baseline should be matrix, got: ", class(baseline))
  if (!is.double(baseline) && !is.null(baseline))
      stop("Unsupported baseline type, expecting double, got: ", typeof(baseline))
  if (nrow(baseline) != nrow(float_and_cat_features_data) && !is.null(baseline))
      stop("Baseline must be matrix of size n_objects*n_classes. Data has ", nrow(float_and_cat_features_data), " objects, baseline has ", nrow(baseline), " rows.")

  if (!is.list(feature_names) && !is.null(feature_names))
      stop("Unsupported feature_names type, expecting list, got: ", typeof(feature_names))
  if (!is.null(feature_names) && (length(feature_names) != data_columns))
      stop("Data has ", data_columns, " columns, feature_names has ", length(feature_names), " columns.")

  if (float_and_cat_columns == 0)
      float_and_cat_features_data <- NULL
  if (text_columns == 0)
      text_features_data <- NULL
  pool <- .Call("CatBoostCreateFromMatrix_R",
                float_and_cat_features_data, label, cat_features_indices, text_features_data, text_features_indices, pairs, weight,
                group_id, group_weight, subgroup_id, pairs_weight, baseline, feature_names)
  attributes(pool) <- list(.Dimnames = list(NULL, as.character(feature_names)), class = "catboost.Pool")
  return(pool)
}


catboost.from_data_frame <- function(data, label = NULL, pairs = NULL, weight = NULL, group_id = NULL, group_weight = NULL,
                                     subgroup_id = NULL, pairs_weight = NULL, baseline = NULL, feature_names = NULL) {
    if (!is.data.frame(data)) {
        stop("Unsupported data type, expecting data.frame, got: ", class(data))
    }
    if (is.null(feature_names)) {
        feature_names <- as.list(colnames(data))
    }
    factor_columns <- vapply(data, is.factor, logical(1))
    num_columns <-
      vapply(data, is.double, logical(1)) |
      vapply(data, is.integer, logical(1)) |
      vapply(data, is.logical, logical(1))
    text_columns <- vapply(data, is.character, logical(1))
    bad_columns <- !(factor_columns | num_columns | text_columns)

    if (sum(bad_columns) > 0) {
        stop("Unsupported column type: ", paste(c(unique(vapply(data[, bad_columns], class, character(1)))), collapse = ", "))
    }

    text_features_data <- data[text_columns]
    text_features_indices <- c()
    for (column_index in which(text_columns)){
        text_features_indices <- c(text_features_indices, column_index - 1)
    }

    float_and_cat_features_data <- data
    cat_features_indices <- c()
    for (column_index in which(factor_columns)) {
        float_and_cat_features_data[, column_index] <- .Call("CatBoostHashStrings_R", as.character(float_and_cat_features_data[[column_index]]))
        cat_features_indices <- c(cat_features_indices, column_index - 1)
    }
    float_and_cat_features_data <- float_and_cat_features_data[!text_columns]

    if (!is.null(pairs)) {
        pairs <- as.matrix(pairs)
    }
    pool <- catboost.from_matrix(as.matrix(float_and_cat_features_data), label, cat_features_indices, as.matrix(text_features_data),
                                 text_features_indices, pairs, weight, group_id, group_weight, subgroup_id, pairs_weight, baseline, feature_names)
    return(pool)
}


#' @name catboost.save_pool
#' @title Save the dataset
#'
#' @description Save the dataset to the CatBoost format.
#'              Files with the following data are created:
#'              \itemize{
#'                  \item Dataset description
#'                  \item Column descriptions
#'              }
#'              Use the catboost.load_pool function to read the resulting files.
#'              These files can also be used in the
#'              \href{https://catboost.ai/docs/concepts/cli-installation.html}{Command-line version}
#'              and the \href{https://catboost.ai/docs/concepts/python-installation.html}{Python library}.
#'
#' @param data A data.frame with features.
#' The following column types are supported:
#'     \itemize{
#'     \item double
#'     \item factor.
#'     It is assumed that categorical features are given in this type of columns.
#'     A standard CatBoost processing procedure is applied to this type of columns:
#'     \describe{
#'         \item{1.}{The values are converted to strings.}
#'         \item{2.}{The ConvertCatFeatureToFloat function is applied to the resulting string.}
#'     }
#' }
#'
#' Default value: Required argument
#' @param label The label vector.
#' @param weight The weights of the label vector.
#' @param baseline Vector of initial (raw) values of the label function for the object.
#' Used in the calculation of final values of trees.
#' @param pool_path The path to the output file that contains the dataset description.
#' @param cd_path The path to the output file that contains the column descriptions.
#' @return Nothing. This method writes a dataset to disk.
#' @export
catboost.save_pool <- function(data, label = NULL, weight = NULL, baseline = NULL,
                               pool_path = "data.pool", cd_path = "cd.pool") {
    if (missing(pool_path) || missing(cd_path))
        stop("Need to specify pool_path and cd_path.")
    if (!is.character(pool_path) || !is.character(cd_path))
        stop("Path must be a string.")


    pool <- label
    if (!is.null(pool)) {
        column_description <- c("Label")
    }
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
    factors <- which(vapply(data, class, character(1)) == "factor")
    if (length(factors) != 0) {
        column_description <- rbind(column_description, data.frame(index = nrow(column_description) + factors - 1,
                                                                   type = rep("Categ", length(factors))))
    }
    pool_path <- path.expand(pool_path)
    cd_path <- path.expand(cd_path)
    write.table(pool, file = pool_path, sep = "\t", row.names = FALSE, col.names = FALSE, quote = FALSE)
    write.table(column_description, file = cd_path, sep = "\t", row.names = FALSE, col.names = FALSE, quote = FALSE)
}


#' @name dim.catboost.Pool
#' @title Dimensions of catboost.Pool
#' @description Get dimensions of a Pool.
#'
#' @param x The input dataset.
#'
#' Default value: Required argument
#' @return Returns a vector of row numbers and column numbers in an catboost.Pool.
#' @export
dim.catboost.Pool <- function(x) {
    return(c(.Call("CatBoostPoolNumRow_R", x), .Call("CatBoostPoolNumCol_R", x)))
}


#' @name dimnames.catboost.Pool
#' @title Dimension names of catboost.Pool
#'
#' @description Get dimension names of a Pool.
#' @param x The input dataset.
#'
#' Default value: Required argument
#' @return A list with the two elements. The second element contains the column names.
#' @export
dimnames.catboost.Pool <- function(x) {
    return(attr(x, ".Dimnames"))
}


#' @name head.catboost.Pool
#' @title Head of catboost.Pool
#'
#' @description Return a list with the first n objects of the dataset.
#'
#'              Each line of this list contains the following information for each object:
#'              \itemize{
#'                  \item The label value.
#'                  \item The weight value.
#'                  \item The feature values.
#'              }
#' @param x The input dataset.
#'
#' Default value: Required argument
#' @param n The quantity of the first objects in the dataset to be returned.
#'
#' Default value: 10
#' @param ... not currently used
#' @return A matrix containing the first \code{n} objects of the dataset.
#' @export
head.catboost.Pool <- function(x, n = 10, ...) {
    if (is.null.handle(x))
        stop("Pool object is invalid.")
    if (n < 0) {
        n <- max(0, dim(x)[1] + n)
    } else {
        n <- min(n, dim(x)[1])
    }
    result <- .Call("CatBoostPoolSlice_R", x, n, 0)
    result <- matrix(unlist(result), nrow = n, byrow = TRUE)
    return(result)
}

#' @name tail.catboost.Pool
#' @title Tail of catboost.Pool
#'
#' @description Return a list with the last n objects of the dataset.
#'
#'              Each line of this list contains the following information for each object:
#'              \itemize{
#'                  \item The target value.
#'                  \item The weight value.
#'                  \item The feature values.
#'              }
#' @param x The input dataset.
#'
#' Default value: Required argument
#' @param n The quantity of the last objects in the dataset to be returned.
#'
#' Default value: 10
#' @param ... not currently used
#' @return A matrix containing the last \code{n} objects of the dataset.
#' @export
tail.catboost.Pool <- function(x, n = 10, ...) {
    if (is.null.handle(x))
        stop("Pool object is invalid.")
    if (n < 0) {
        n <- max(0, dim(x)[1] + n)
    } else {
        n <- min(n, dim(x)[1])
    }
    result <- .Call("CatBoostPoolSlice_R", x, n, dim(x)[1] - n)
    result <- matrix(unlist(result), nrow = n, byrow = TRUE)
    return(result)
}

#' @name print.catboost.Pool
#' @title Print catboost.Pool
#'
#' @description Print dimensions of catboost.Pool.
#'
#' @param x a catboost.Pool object
#'
#' Default value: Required argument
#' @param ... not currently used
#' @return Nothing. This method prints pool dimensions.
#' @export
print.catboost.Pool <- function(x, ...) {
    if (is.null.handle(x))
        cat("Warning: pool object is invalid.")
    cat("catboost.Pool\n", nrow(x), " rows, ", ncol(x), " columns", sep = "")
}


#' @title Print basic information about model
#' @description Displays the most general characteristics of a CatBoost model.
#' @param x The model obtained as the result of training.
#' @param ... Not used
#' @return The same model that was passed as input.
#' @export
print.catboost.Model <- function(x, ...) {
    cat(sprintf("CatBoost model (%d trees)\n", x$tree_count))
    cat(sprintf("Loss function: %s\n", catboost.get_plain_params(x)$loss_function))
    cat(sprintf("Fit to %d feature(s)\n", x$feature_count))
    if (is.null.handle(x$cpp_obj$handle))
        cat("(Handle is incomplete)\n")
    return(invisible(x))
}


#' @title Print basic information about model
#' @description Displays the most general characteristics of a CatBoost model
#' (same as 'print').
#' @param object The model obtained as the result of training.
#' @param ... Not used
#' @return The same model that was passed as input.
#' @export
summary.catboost.Model <- function(object, ...) {
    print.catboost.Model(object)
}


#' @name catboost.train
#' @title Train the model
#'
#' @description Train the model using a CatBoost dataset.
#'
#' The list of parameters
#'
#' \itemize{
#'   \item Common parameters
#'   \itemize{
#'     \item fold_permutation_block
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
#'       is set to "42", the corresponding non-existing feature is successfully ignored.
#'
#'       The identifier corresponds to the feature's index.
#'       Feature indices used in train and feature importance are numbered from 0 to featureCount-1.
#'       If a file is used as input data then any non-feature column types are ignored when calculating these
#'       indices. For example, each row in the input file contains data in the following order:
#'       "categorical feature<\verb{\t}>label<\verb{\t}>numerical feature". So for the row "rock<\verb{\t}>0<\verb{\t}>42",
#'       the identifier for the "rock" feature is 0, and for the "42" feature it is 1.
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
#'       The loss function (see \url{https://catboost.ai/docs/concepts/loss-functions.html#loss-functions})
#'       to use in training. The specified value also determines the machine learning problem to solve.
#'
#'       Format:
#'
#'       <Loss function 1>[:<parameter 1>=<value>:..<parameter N>=<value>:]
#'
#'       Supported loss functions:
#'       \itemize{
#'         \item 'Logloss'
#'         \item 'CrossEntropy'
#'         \item 'MultiClass'
#'         \item 'MultiClassOneVsAll'
#'         \item 'RMSE'
#'         \item 'MAE'
#'         \item 'Quantile'
#'         \item 'LogLinQuantile'
#'         \item 'MAPE'
#'         \item 'Poisson'
#'         \item 'Lq'
#'         \item 'PairLogit'
#'         \item 'PairLogitPairwise'
#'         \item 'YetiRank'
#'         \item 'YetiRankPairwise'
#'         \item 'QueryCrossEntropy'
#'         \item 'QueryRMSE'
#'         \item 'QuerySoftMax'
#'       }
#'
#'       Supported parameters:
#'       \itemize{
#'         \item alpha - The coefficient used in quantile-based losses ('Quantile' and 'LogLinQuantile'). The default value is 0.5.
#'
#'         For example, if you need to calculate the value of Quantile with the coefficient \eqn{\alpha = 0.1}, use the following construction:
#'
#'         'Quantile:alpha=0.1'
#'       }
#'
#'       Default value:
#'
#'       'RMSE'
#'     \item custom_loss
#'
#'       Loss function (see \url{https://catboost.ai/docs/concepts/loss-functions.html#loss-functions})
#'       values to output during training.
#'       These functions are not used for optimization and are displayed for informational purposes only.
#'
#'       Format:
#'
#'       c(<Loss function 1>[:<parameter>=<value>],<Loss function 2>[:<parameter>=<value>],...,<Loss function N>[:<parameter>=<value>])
#'
#'       Supported loss functions:
#'       \itemize{
#'         \item 'Logloss'
#'         \item 'CrossEntropy'
#'         \item 'Precision'
#'         \item 'Recall'
#'         \item 'F1'
#'         \item 'F'
#'         \item 'BalancedAccuracy'
#'         \item 'BalancedErrorRate'
#'         \item 'MCC'
#'         \item 'Accuracy'
#'         \item 'CtrFactor'
#'         \item 'AUC'
#'         \item 'BrierScore'
#'         \item 'HingeLoss'
#'         \item 'HammingLoss'
#'         \item 'ZeroOneLoss'
#'         \item 'Kappa'
#'         \item 'WKappa'
#'         \item 'LogLikelihoodOfPrediction'
#'         \item 'MultiClass'
#'         \item 'MultiClassOneVsAll'
#'         \item 'TotalF1'
#'         \item 'MAE'
#'         \item 'MAPE'
#'         \item 'Poisson'
#'         \item 'Quantile'
#'         \item 'RMSE'
#'         \item 'LogLinQuantile'
#'         \item 'Lq'
#'         \item 'NumErrors'
#'         \item 'SMAPE'
#'         \item 'R2'
#'         \item 'MSLE'
#'         \item 'MedianAbsoluteError'
#'         \item 'PairLogit'
#'         \item 'PairLogitPairwise'
#'         \item 'PairAccuracy'
#'         \item 'QueryCrossEntropy'
#'         \item 'QueryRMSE'
#'         \item 'QuerySoftMax'
#'         \item 'PFound'
#'         \item 'NDCG'
#'         \item 'AverageGain'
#'         \item 'PrecisionAt'
#'         \item 'RecallAt'
#'         \item 'MAP'
#'         \item 'MRR'
#'         \item 'ERR'
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
#'       (see \url{https://catboost.ai/docs/concepts/output-data_loss-function.html#output-data_loss-function})
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
#'         \item 'Logloss'
#'         \item 'CrossEntropy'
#'         \item 'Precision'
#'         \item 'Recall'
#'         \item 'F1'
#'         \item 'F'
#'         \item 'BalancedAccuracy'
#'         \item 'BalancedErrorRate'
#'         \item 'MCC'
#'         \item 'Accuracy'
#'         \item 'CtrFactor'
#'         \item 'AUC'
#'         \item 'BrierScore'
#'         \item 'HingeLoss'
#'         \item 'HammingLoss'
#'         \item 'ZeroOneLoss'
#'         \item 'Kappa'
#'         \item 'WKappa'
#'         \item 'LogLikelihoodOfPrediction'
#'         \item 'MultiClass'
#'         \item 'MultiClassOneVsAll'
#'         \item 'TotalF1'
#'         \item 'MAE'
#'         \item 'MAPE'
#'         \item 'Poisson'
#'         \item 'Quantile'
#'         \item 'RMSE'
#'         \item 'LogLinQuantile'
#'         \item 'Lq'
#'         \item 'NumErrors'
#'         \item 'SMAPE'
#'         \item 'R2'
#'         \item 'MSLE'
#'         \item 'MedianAbsoluteError'
#'         \item 'PairLogit'
#'         \item 'PairLogitPairwise'
#'         \item 'PairAccuracy'
#'         \item 'QueryCrossEntropy'
#'         \item 'QueryRMSE'
#'         \item 'QuerySoftMax'
#'         \item 'PFound'
#'         \item 'NDCG'
#'         \item 'AverageGain'
#'         \item 'PrecisionAt'
#'         \item 'RecallAt'
#'         \item 'MAP'
#'         \item 'MRR'
#'         \item 'ERR'
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
#'       1000
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
#'     \item leaf_estimation_iterations
#'
#'       The number of gradient steps when calculating the values in leaves.
#'
#'       Default value:
#'
#'       1
#'
#'     \item depth
#'
#'       Depth of the trees.
#'
#'       The value can be any integer up to 16. It is recommended to use values in the range [1; 10].
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
#'
#'     \item random_seed
#'
#'       The random seed used for training.
#'
#'       Default value:
#'
#'       0
#'
#'    \item nan_mode
#'
#'       Way to process missing values.
#'
#'       Possible values:
#'       \itemize{
#'         \item \code{'Min'}
#'         \item \code{'Max'}
#'         \item \code{'Forbidden'}
#'       }
#'
#'       Default value:
#'
#'       \code{'Min'}
#'
#'     \item od_pval
#'
#'       Use the Overfitting detector (see \url{https://catboost.ai/docs/concepts/overfitting-detector.html#overfitting-detector})
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
#'     \item od_type
#'
#'       The method used to calculate the values in leaves.
#'
#'       Possible values:
#'       \itemize{
#'         \item IncToDec
#'         \item Iter
#'       }
#'
#'       Restriction.
#'       Do not specify the overfitting detector threshold when using the Iter type.
#'
#'       Default value:
#'
#'       'IncToDec'
#'
#'     \item od_wait
#'
#'       The number of iterations to continue the training after the iteration with the optimal loss function value.
#'       The purpose of this parameter differs depending on the selected overfitting detector type:
#'       \itemize{
#'         \item IncToDec - Ignore the overfitting detector when the threshold is reached and continue learning for the specified number of iterations after the iteration with the optimal loss function value.
#'         \item Iter - Consider the model overfitted and stop training after the specified number of iterations since the iteration with the optimal loss function value.
#'       }
#'
#'       Default value:
#'
#'       20
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
#'     \item grow_policy
#'
#'       GPU only. The tree growing policy. It describes how to perform greedy tree construction.
#'
#'       Possible values:
#'       \itemize{
#'         \item SymmetricTree
#'         \item Lossguide
#'         \item Depthwise
#'       }
#'
#'       Default value:
#'
#'       SymmetricTree
#'
#'     \item min_data_in_leaf
#'
#'       GPU only.
#'       The minimum training samples count in leaf.
#'       CatBoost will not search for new splits in leaves with samples count less than min_data_in_leaf.
#'       This parameter is used only for Depthwise and Lossguide growing policies.
#'
#'       Default value:
#'
#'       1
#'
#'     \item max_leaves
#'
#'       GPU only. The maximum leaf count in resulting tree. Used only for Lossguide growing policy.
#'       This parameter is used only for Lossguide growing policy.
#'
#'       Default value:
#'
#'       31
#'
#'     \item score_function
#'       GPU only. Score that is used during tree construction to select the next tree split.
#'
#'       Possible values:
#'       \itemize{
#'         \item L2
#'         \item Cosine
#'         \item NewtonL2
#'         \item NewtonCosine
#'       }
#'
#'       Default value:
#'
#'       Cosine
#'
#'       For growing policy Lossguide default is NewtonL2.
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
#'     \item model_size_reg
#'
#'       Model size regularization coefficient. The influence coefficient of the model size for choosing tree structure.
#'       To get a smaller model size - increase this coefficient.
#'
#'       Any positive values are allowed.
#'
#'       Default value:
#'
#'       0.5
#'
#'     \item has_time
#'
#'       Use the order of objects in the input data
#'       (do not perform a random permutation of the dataset at the preprocessing stage)
#'
#'       Default value:
#'
#'       FALSE (not used; permute input dataset)
#'
#'     \item allow_const_label
#'
#'       To allow the constant label value in the dataset.
#'
#'       Default value:
#'
#'       FALSE
#'
#'     \item name
#'
#'       The experiment name to display in visualization tools (see \url{https://catboost.ai/docs/features/visualization.html#visualization}).
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
#'       For example, for 3 class classification you could use:
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
#'
#'     \item bootstrap_type
#'
#'       Bootstrap type. Defines the method for sampling the weights of documents.
#'
#'       Possible values:
#'       \itemize{
#'         \item 'Bayesian'
#'         \item 'Bernoulli'
#'         \item 'Poisson'
#'         \item 'MVS'
#'         \item 'No'
#'       }
#'
#'       Poisson bootstrap is supported only on GPU.
#'
#'       Default value:
#'
#'       \code{'Bayesian'}
#'
#'     \item bagging_temperature
#'
#'        Controls intensity of Bayesian bagging. The higher the temperature the more aggressive bagging is.
#'
#'        Typical values are in the range \eqn{[0, 1]} (0 is for no bagging).
#'
#'        Possible values are in the range \eqn{[0, +\infty)}.
#'
#'        Default value:
#'
#'        1
#'
#'     \item subsample
#'
#'       Sample rate for bagging. This parameter can be used if one of the following bootstrap types is defined:
#'       \itemize{
#'         \item 'Bernoulli'
#'       }
#'
#'       Default value:
#'
#'       0.66
#'
#'     \item sampling_unit
#'
#'       The parameter allows to specify the sampling scheme: sample weights for each object individually or for an entire group of objects together.
#'
#'       Possible values:
#'       \itemize{
#'         \item 'Object'
#'         \item 'Group'
#'       }
#'
#'       Default value:
#'
#'       \code{'Object'}
#'
#'     \item sampling_frequency
#'
#'       Frequency to sample weights and objects when building trees.
#'
#'       Possible values:
#'       \itemize{
#'         \item 'PerTree'
#'         \item 'PerTreeLevel'
#'       }
#'
#'       Default value:
#'
#'       \code{'PerTreeLevel'}
#'
#'     \item model_shrink_rate
#'
#'       For i > 0 at the start of i-th iteration multiplies model by (1 - model_shrink_rate / i).
#'
#'       Possible values: [0, 1).
#'
#'       Default value: 0
#'   }
#'   \item CTR settings
#'   \itemize{
#'     \item simple_ctr
#'
#'       Binarization settings for categorical features (see \url{https://catboost.ai/docs/concepts/algorithm-main-stages_cat-to-numberic.html}).
#'
#'       Format:
#'
#'       \code{c(CtrType[:TargetBorderCount=BorderCount][:TargetBorderType=BorderType][:CtrBorderCount=Count][:CtrBorderType=Type][:Prior=num_1/denum_1]..[:Prior=num_N/denum_N])}
#'
#'       Components:
#'       \itemize{
#'         \item CTR types for training on CPU:
#'         \itemize{
#'           \item \code{'Borders'}
#'           \item \code{'Buckets'}
#'           \item \code{'BinarizedTargetMeanValue'}
#'           \item \code{'Counter'}
#'         }
#'         \item CTR types for training on GPU:
#'         \itemize{
#'           \item \code{'Borders'}
#'           \item \code{'Buckets'}
#'           \item \code{'FeatureFreq'}
#'           \item \code{'FloatTargetMeanValue'}
#'         }
#'         \item The number of borders for label value binarization. (see \url{https://catboost.ai/docs/concepts/quantization.html})
#'         Only used for regression problems. Allowed values are integers from 1 to 255 inclusively. The default value is 1.
#'         This option is available for training on CPU only.
#'         \item The binarization (see \url{https://catboost.ai/docs/concepts/quantization.html})
#'         type for the label value. Only used for regression problems.
#'
#'         Possible values:
#'         \itemize{
#'           \item \code{'Median'}
#'           \item \code{'Uniform'}
#'           \item \code{'UniformAndQuantiles'}
#'           \item \code{'MaxLogSum'}
#'           \item \code{'MinEntropy'}
#'           \item \code{'GreedyLogSum'}
#'         }
#'         By default, \code{'MinEntropy'}
#'         This option is available for training on CPU only.
#'         \item The number of splits for categorical features. Allowed values are integers from 1 to 255 inclusively.
#'         \item The binarization type for categorical features.
#'         Supported values for training on CPU:
#'         \itemize{
#'           \item \code{'Uniform'}
#'         }
#
#'         Supported values for training on GPU:
#'         \itemize{
#'           \item \code{'Median'}
#'           \item \code{'Uniform'}
#'           \item \code{'UniformAndQuantiles'}
#'           \item \code{'MaxLogSum'}
#'           \item \code{'MinEntropy'}
#'           \item \code{'GreedyLogSum'}
#'         }
#'         \item Priors to use during training (several values can be specified)
#
#'         Possible formats:
#'         \itemize{
#'           \item \code{'One number - Adds the value to the numerator.'}
#'           \item \code{'Two slash-delimited numbers (for GPU only) - Use this format to set a fraction. The number is added to the numerator and the second is added to the denominator.'}
#'         }
#'       }
#'
#'     \item combinations_ctr
#'
#'       Binarization settings for combinations of categorical features (see \url{https://catboost.ai/docs/concepts/algorithm-main-stages_cat-to-numberic.html}).
#'
#'       Format:
#'
#'       \code{c(CtrType[:TargetBorderCount=BorderCount][:TargetBorderType=BorderType][:CtrBorderCount=Count][:CtrBorderType=Type][:Prior=num_1/denum_1]..[:Prior=num_N/denum_N])}
#'
#'       Components:
#'       \itemize{
#'         \item CTR types for training on CPU:
#'         \itemize{
#'           \item \code{'Borders'}
#'           \item \code{'Buckets'}
#'           \item \code{'BinarizedTargetMeanValue'}
#'           \item \code{'Counter'}
#'         }
#'         \item CTR types for training on GPU:
#'         \itemize{
#'           \item \code{'Borders'}
#'           \item \code{'Buckets'}
#'           \item \code{'FeatureFreq'}
#'           \item \code{'FloatTargetMeanValue'}
#'         }
#'         \item The number of borders for target binarization. (see \url{https://catboost.ai/docs/concepts/quantization.html})
#'         Only used for regression problems. Allowed values are integers from 1 to 255 inclusively. The default value is 1.
#'         This option is available for training on CPU only.
#'         \item The binarization (see \url{https://catboost.ai/docs/concepts/quantization.html})
#'         type for the target. Only used for regression problems.
#'
#'         Possible values:
#'         \itemize{
#'           \item \code{'Median'}
#'           \item \code{'Uniform'}
#'           \item \code{'UniformAndQuantiles'}
#'           \item \code{'MaxLogSum'}
#'           \item \code{'MinEntropy'}
#'           \item \code{'GreedyLogSum'}
#'         }
#'         By default, \code{'MinEntropy'}
#'         This option is available for training on CPU only.
#'         \item The number of splits for categorical features. Allowed values are integers from 1 to 255 inclusively.
#'         \item The binarization type for categorical features.
#'         Supported values for training on CPU:
#'         \itemize{
#'           \item \code{'Uniform'}
#'         }
#
#'         Supported values for training on GPU:
#'         \itemize{
#'           \item \code{'Median'}
#'           \item \code{'Uniform'}
#'           \item \code{'UniformAndQuantiles'}
#'           \item \code{'MaxLogSum'}
#'           \item \code{'MinEntropy'}
#'           \item \code{'GreedyLogSum'}
#'         }
#'         \item Priors to use during training (several values can be specified)
#
#'         Possible formats:
#'         \itemize{
#'           \item \code{'One number - Adds the value to the numerator.'}
#'           \item \code{'Two slash-delimited numbers (for GPU only) - Use this format to set a fraction. The number is added to the numerator and the second is added to the denominator.'}
#'         }
#'       }
#'
#'     \item ctr_target_border_count
#'
#'       Maximum number of borders used in target binarization for categorical features that need it.
#'       If TargetBorderCount is specified in 'simple_ctr', 'combinations_ctr' or 'per_feature_ctr' option it overrides this value.
#'
#'       Default value:
#'
#'       1
#'
#'     \item counter_calc_method
#'
#'       The method for calculating the Counter CTR type for the test dataset.
#'
#'       Possible values:
#'         \itemize{
#'           \item \code{'Full'}
#'           \item \code{'FullTest'}
#'           \item \code{'PrefixTest'}
#'           \item \code{'SkipTest'}
#'         }
#'
#'         Default value: \code{'PrefixTest'}
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
#'       The maximum number of leaves with categorical features.
#'       If the number of leaves exceeds the specified limit, some leaves are discarded.
#'       The value must be positive (for zero limit use \code{ignored_features} parameter).
#'
#'       The leaves to be discarded are selected as follows:
#'       \enumerate{
#'         \item The leaves are sorted by the frequency of the values.
#'         \item The top N leaves are selected, where N is the value specified in the parameter.
#'         \item All leaves starting from N+1 are discarded.
#'       }
#'
#'       This option reduces the resulting model size and the amount of memory required for training.
#'       Note that the resulting quality of the model can be affected.
#'
#'       Default value:
#'
#'       None (The number of leaves with categorical features is not limited)
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
#'       FALSE (Both simple features and feature combinations are taken in account when limiting the number of leaves with categorical features)
#'
#'
#'   }
#'   \item Binarization settings
#'   \itemize{
#'     \item  border_count
#'
#'       The number of splits for numerical features. Allowed values are integers from 1 to 255 inclusively.
#'
#'       Default value:
#'
#'       254 for training on CPU or 128 for training on GPU
#'
#'     \item feature_border_type
#'
#'       The binarization mode (see \url{https://catboost.ai/docs/concepts/quantization.html})
#'       for numerical features.
#'
#'       Possible values:
#'       \itemize{
#'         \item \code{'Median'}
#'         \item \code{'Uniform'}
#'         \item \code{'UniformAndQuantiles'}
#'         \item \code{'MaxLogSum'}
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
#'       The number of CPU cores.
#'   }
#'   \item Output settings
#'   \itemize{
#'     \item logging_level
#'
#'       Possible values:
#'       \itemize{
#'         \item \code{'Silent'}
#'         \item \code{'Verbose'}
#'         \item \code{'Info'}
#'         \item \code{'Debug'}
#'       }
#'
#'       Default value:
#'
#'       'Silent'
#'
#'     \item metric_period
#'
#'       The frequency of iterations to print the information to stdout. The value should be a positive integer.
#'
#'       Default value:
#'
#'       1
#'
#'     \item train_dir
#'
#'       The directory for storing the files generated during training.
#'
#'       Default value:
#'
#'       None (current catalog)
#'
#'     \item save_snapshot
#'
#'       Enable snapshotting for restoring the training progress after an interruption.
#'
#'       Default value:
#'
#'       None
#'
#'     \item snapshot_file
#'
#'       Settings for recovering training after an interruption (see
#'       \url{https://catboost.ai/docs/features/snapshots.html}).
#'
#'       Depending on whether the file specified exists in the file system:
#'       \itemize{
#'         \item Missing - write information about training progress to the specified file.
#'         \item Exists - load data from the specified file and continue training from where it left off.
#'       }
#'
#'       Default value:
#'
#'       File can't be generated or read. If the value is omitted, the file name is experiment.cbsnapshot.
#'
#'   \item snapshot_interval
#'
#'       Interval between saving snapshots (seconds)
#'
#'       Default value:
#'
#'       600
#'
#'   \item allow_writing_files
#'
#'       If this flag is set to FALSE, no files with different diagnostic info will be created during training.
#'       With this flag set to FALSE no snapshotting can be done. Plus visualisation will not
#'       work, because visualisation uses files that are created and updated during training.
#'
#'       Default value:
#'
#'       TRUE
#'
#'   \item approx_on_full_history
#'
#'       If this flag is set to TRUE, each approximated value is calculated using all the preceding rows in the fold (slower, more accurate).
#'       If this flag is set to FALSE, each approximated value is calculated using only the beginning 1/fold_len_multiplier fraction of the fold (faster, slightly less accurate).
#'
#'       Default value:
#'
#'       FALSE
#'
#'   \item boosting_type
#'
#'       Boosting scheme.
#'      Possible values:
#'          - 'Ordered' - Gives better quality, but may slow down the training.
#'          - 'Plain' - The classic gradient boosting scheme. May result in quality degradation, but does not slow down the training.
#'
#'       Default value:
#'
#'       Depends on object count and feature count in train dataset and on learning mode.
#'
#'   \item dev_score_calc_obj_block_size
#'
#'       CPU only. Size of block of samples in score calculation. Should be > 0
#'       Used only for learning speed tuning.
#'       Changing this parameter can affect results in pairwise scoring mode due to numerical accuracy differences
#'
#'       Default value:
#'
#'       5000000
#'
#'   \item dev_efb_max_buckets
#'
#'       CPU only. Maximum bucket count in exclusive features bundle. Should be in an integer between 0 and 65536.
#'       Used only for learning speed tuning.
#'
#'       Default value:
#'
#'       1024
#'
#'   \item sparse_features_conflict_fraction
#'
#'      CPU only. Maximum allowed fraction of conflicting non-default values for features in exclusive features bundle.
#'      Should be a real value in [0, 1) interval.
#'
#'      Default value:
#'
#'      0.0
#'
#'    \item leaf_estimation_backtracking
#'
#'        Type of backtracking during gradient descent.
#'        Possible values:
#'            - 'No' - never backtrack; supported on CPU and GPU
#'            - 'AnyImprovement' - reduce the descent step until the value of loss function is less than before the step; supported on CPU and GPU
#'            - 'Armijo' - reduce the descent step until Armijo condition is satisfied; supported on GPU only
#'
#'        Default value:
#'
#'        'AnyImprovement'
#'
#'   }
#' }
#'
#' @param learn_pool The dataset used for training the model.
#'
#' Default value: Required argument
#' @param test_pool The dataset used for testing the quality of the model.
#'
#' Default value: NULL (not used)
#' @param params The list of parameters to start training with.
#'
#' If omitted, default values are used (see The list of parameters).
#'
#' If set, the passed list of parameters overrides the default values.
#'
#' Default value: Required argument
#' @examples
#' \dontrun{
#' train_pool_path <- system.file("extdata", "adult_train.1000", package = "catboost")
#' test_pool_path <- system.file("extdata", "adult_test.1000", package = "catboost")
#' cd_path <- system.file("extdata", "adult.cd", package = "catboost")
#' train_pool <- catboost.load_pool(train_pool_path, column_description = cd_path)
#' test_pool <- catboost.load_pool(test_pool_path, column_description = cd_path)
#' fit_params <- list(
#'     iterations = 100,
#'     loss_function = 'Logloss',
#'     ignored_features = c(4, 9),
#'     border_count = 32,
#'     depth = 5,
#'     learning_rate = 0.03,
#'     l2_leaf_reg = 3.5,
#'     train_dir = 'train_dir')
#' model <- catboost.train(train_pool, test_pool, fit_params)
#' }
#' @return Model object.
#' @export
#' @seealso \url{https://catboost.ai/docs/concepts/r-reference_catboost-train.html}
catboost.train <- function(learn_pool, test_pool = NULL, params = list()) {
    if (!inherits(learn_pool, "catboost.Pool"))
        stop("Expected catboost.Pool, got: ", class(learn_pool))
    if (is.null.handle(learn_pool))
        stop("'learn_pool' object is invalid.")
    if (!is.null(test_pool) && !inherits(test_pool, "catboost.Pool"))
        stop("Expected catboost.Pool, got: ", class(test_pool))
    if (!is.null(test_pool) && is.null.handle(test_pool))
        stop("'test_pool' object is invalid.")
    if (length(params) == 0)
        message("Training catboost with default parameters! See help(catboost.train).")

    params <- process_synonyms(params)
    json_params <- prepare_train_export_parameters(params)
    handle <- .Call("CatBoostFit_R", learn_pool, test_pool, json_params)
    raw <- .Call("CatBoostSerializeModel_R", handle)
    model <- create.model.base(handle, raw)

    if (catboost._is_oblivious(model)) {
        if (catboost._is_groupwise_metric(model)) {
            # too expensive
        } else {
            model$feature_importances <- catboost.get_feature_importance(model)
            rownames(model$feature_importances) <- colnames(learn_pool)
        }
    }

    plain_params <- catboost.get_plain_params(model)
    model$tree_count <- catboost.ntrees(model)
    model$learning_rate <- plain_params$learning_rate
    model$feature_count <- ncol(learn_pool) - length(plain_params$ignored_features)
    return(model)
}

process_synonyms <- function(params) {
    params <- process_synonyms_in_one_group(c('loss_function', 'objective'), params)
    params <- process_synonyms_in_one_group(c('iterations', 'n_estimators', 'num_boost_round', 'num_trees'), params)
    params <- process_synonyms_in_one_group(c('learning_rate', 'eta'), params)
    params <- process_synonyms_in_one_group(c('random_seed', 'random_state'), params)
    params <- process_synonyms_in_one_group(c('l2_leaf_reg', 'reg_lambda'), params)
    params <- process_synonyms_in_one_group(c('depth', 'max_depth'), params)
    params <- process_synonyms_in_one_group(c('min_data_in_leaf', 'min_child_samples'), params)
    params <- process_synonyms_in_one_group(c('max_leaves', 'num_leaves'), params)
    params <- process_synonyms_in_one_group(c('rsm', 'colsample_bylevel'), params)
    params <- process_synonyms_in_one_group(c('border_count', 'max_bin'), params)
    params <- process_synonyms_in_one_group(c('verbose', 'verbose_eval'), params)

    return(params)
}

process_synonyms_in_one_group <- function(synonyms, params) {
    value <- NULL
    for (synonym in synonyms) {
        if (synonym %in% names(params)) {
            if (!is.null(value)) {
                message <- paste("Only one of the parameters [",
                                paste(synonyms, collapse=', '),
                                "] should be initialized.", sep="")
                stop(message)
            }
            value = params[[synonym]]
            params[[synonym]] <- NULL
        }
   }

  if (!is.null(value)) {
      params[[synonyms[[1]]]] <- value
  }

  return(params)
}

prepare_train_export_parameters <- function(params) {

    if (length(params) == 0) {
        return ("{}")
    }

    if (!is.null(params$early_stopping_rounds)) {
        params$od_type <- "Iter"
        params$od_pval <- NULL
        params$od_wait <- params$early_stopping_rounds
        params$early_stopping_rounds <- NULL
    }

    if (!is.null(params$per_float_feature_quantization)) {
        params$per_float_feature_quantization <- as.list(params$per_float_feature_quantization)
    }

    if (!is.null(params$ignored_features)) {
        # treat parameter AsIs to avoid conversion from 1-element array to atomic value
        params$ignored_features <- I(as.character(params$ignored_features))
    }

    return(jsonlite::toJSON(params, auto_unbox = TRUE, digits = 10))
}

#' @name catboost.cv
#' @title Cross-validate model.
#' @description Estimate model performance using cross-validation.
#' @param pool Data to cross-validate on
#' @param params Parameters for catboost.train
#' @param fold_count Folds count.
#' @param type is type of cross-validation.
#' @param partition_random_seed The random seed used for splitting pool into folds.
#' @param shuffle Shuffle the dataset objects before splitting into folds.
#' @param stratified Perform stratified sampling.
#' @param early_stopping_rounds Activates Iter overfitting detector with od_wait set to early_stopping_rounds.
#' @return A data.frame of evaluation results from cross-validation.
#' @export
catboost.cv <- function(pool,
                        params = list(),
                        fold_count = 3,
                        type = "Classical",
                        partition_random_seed = 0,
                        shuffle = TRUE,
                        stratified = FALSE,
                        early_stopping_rounds = NULL) {

    if (!inherits(pool, "catboost.Pool"))
        stop("Expected catboost.Pool, got: ", class(pool))
    if (is.null.handle(pool))
        stop("Pool object is invalid.")
    if (length(params) == 0)
        message("Training catboost with default parameters! See help(catboost.train).")

    if (!is.null(early_stopping_rounds)) {
        params$od_type <- "Iter"
        params$od_pval <- NULL
        params$od_wait <- early_stopping_rounds
    }

    json_params <- prepare_train_export_parameters(params)
    result <- .Call("CatBoostCV_R", json_params, pool, fold_count, type, partition_random_seed, shuffle, stratified)

    return(data.frame(result))
}

#' @name catboost.sum_models
#' @title Sum models.
#' @description Blend trees and counters of two or more trained CatBoost models into a new model.
#'              Leaf values can be individually weighted for each input model. For example, it may
#'              be useful to blend models trained on different validation datasets.
#'
#' @param models Models for the summation.
#'
#' Default value: Required argument
#' @param weights The weights of the models.
#'
#' Default value: NULL (use weight 1 for every model)
#' @param ctr_merge_policy The counters merging policy.
#' Possible values:
#' \itemize{
#'   \item 'FailIfCtrIntersects'
#'     Ensure that the models have zero intersecting counters
#'   \item 'LeaveMostDiversifiedTable'
#'     Use the most diversified counters by the count of unique hash values
#'   \item 'IntersectingCountersAverage'
#'     Use the average ctr counter values in the intersecting bins
#' }
#'
#' Default value: 'IntersectingCountersAverage'
#' @return Model object.
#' @export
catboost.sum_models <- function(models, weights = NULL, ctr_merge_policy = 'IntersectingCountersAverage') {
    if (is.null(weights)) {
        weights = rep(1.0, length(models));
    } else if (length(models) != length(weights)) {
        stop("The length of this list must be equal to the number of blended models.");
    }

    i <- 1L
    modelsVector <- list()
    for (model in models) {
        catboost.restore_handle(model)
        modelsVector[[i]] <- model$cpp_obj$handle
        i <- i + 1L
    }
    handle <- .Call("CatBoostSumModels_R", modelsVector, weights, ctr_merge_policy)
    raw <- .Call("CatBoostSerializeModel_R", handle)
    model <- create.model.base(handle, raw)

    model$random_seed <- 0
    model$learning_rate <- 0

    return(model)
}

#' @name catboost.load_model
#' @title Load the model
#'
#' @description Load the model from a file.
#'
#'              Note: Feature importance (see \url{https://catboost.ai/docs/concepts/fstr.html#fstr})
#'              is not saved when using this function.
#' @param model_path The path to the model.
#'
#' Default value: Required argument
#' @param file_format Format of the model file.
#'
#' Default value: 'cbm'
#' @return A model object.
#' @export
#' @seealso \url{https://catboost.ai/docs/concepts/r-reference_catboost-load_model.html}
catboost.load_model <- function(model_path, file_format = "cbm") {
    model_path <- path.expand(model_path)
    handle <- .Call("CatBoostReadModel_R", model_path, file_format)
    raw <- .Call("CatBoostSerializeModel_R", handle)
    model <- create.model.base(handle, raw)
    model$tree_count <- catboost.ntrees(model)
    model$learning_rate <- catboost.get_plain_params(model)[['learning_rate']]
    return(model)
}


#' @name catboost.save_model
#' @title Save the model
#'
#' @description Save the model to a file.
#'
#'              Note: Feature importance (see \url{https://catboost.ai/docs/concepts/fstr.html#fstr})
#'              is not saved when using this function.
#' @param model The model to be saved.
#'
#' Default value: Required argument
#' @param model_path The path to the resulting binary file with the model description.
#' Used for solving other machine learning problems (for instance, applying a model).
#'
#' Default value: Required argument
#' @param file_format specified format model from a file.
#' Possible values:
#' \itemize{
#'   \item 'cbm'
#'     For catboost binary format
#'   \item 'coreml'
#'     To export into Apple CoreML format
#'   \item 'onnx'
#'     To export into ONNX-ML format
#'   \item 'pmml'
#'     To export into PMML format
#'   \item 'cpp'
#'     To export as C++ code
#'   \item 'python'
#'     To export as Python code.
#' }
#'
#' Default value: 'cbm'
#' @param export_parameters are a parameters for CoreML or PMML export.
#' @param pool is training pool.
#' @return Status, the result of model shrinking. TRUE if shrinking succeeded, FALSE otherwise.
#' @export
#' @seealso \url{https://catboost.ai/docs/features/export-model-to-core-ml.html}
catboost.save_model <- function(model, model_path,
                                file_format = "cbm",
                                export_parameters = NULL,
                                pool = NULL) {
    if (!is.null(pool) && !inherits(pool, "catboost.Pool"))
        stop("Expected catboost.Pool, got: ", class(pool))
    if (!is.null(pool) && is.null.handle(pool))
        stop("Pool object is invalid.")
    params_string <- ""
    if (!is.null(export_parameters))
        params_string <- jsonlite::toJSON(export_parameters, auto_unbox = TRUE)

    catboost.restore_handle(model)
    model_path <- path.expand(model_path)
    status <- .Call("CatBoostOutputModel_R", model$cpp_obj$handle, model_path, file_format, params_string, pool)
    return(status)
}


#' @name catboost.predict
#' @title Apply the model
#'
#' @description Apply the model to the given dataset.
#'
#'              Peculiarities: In case of multiclassification the prediction is returned in the form of a matrix.
#'              Each line of this matrix contains the predictions for one object of the input dataset.
#' @param model The model obtained as the result of training.
#'
#' Default value: Required argument
#' @param pool The input dataset.
#'
#' Default value: Required argument
#' @param verbose Verbose output to stdout.
#'
#' Default value: FALSE (not used)
#' @param prediction_type The format for displaying approximated values in output data
#' (see \url{https://catboost.ai/docs/concepts/output-data.html}).
#'
#' Possible values:
#' \itemize{
#'   \item 'Probability'
#'   \item 'LogProbability'
#'   \item 'Class'
#'   \item 'RawFormulaVal'
#'   \item 'Exponent'
#'   \item 'RMSEWithUncertainty'
#' }
#'
#' Default value: 'RawFormulaVal'
#' @param ntree_start Model is applied on the interval [ntree_start, ntree_end) (zero-based indexing).
#'
#' Default value: 0
#' @param ntree_end Model is applied on the interval [ntree_start, ntree_end) (zero-based indexing).
#'
#' Default value: 0 (if value equals to 0 this parameter is ignored and ntree_end equal to tree_count)
#' @param thread_count The number of threads to use when applying the model. If -1, then the number of threads is set to the number of CPU cores.
#'
#' Allows you to optimize the speed of execution. This parameter doesn't affect results.
#'
#' Default value: 1
#' @return Vector of predictions (matrix for multi-class classification).
#' @export
#' @seealso \url{https://catboost.ai/docs/concepts/r-reference_catboost-predict.html}
catboost.predict <- function(model, pool,
                             verbose = FALSE, prediction_type = "RawFormulaVal",
                             ntree_start = 0, ntree_end = 0, thread_count = -1) {
    if (!inherits(pool, "catboost.Pool"))
        stop("Expected catboost.Pool, got: ", class(pool))
    if (is.null.handle(pool))
        stop("Pool object is invalid.")

    catboost.restore_handle(model)
    prediction <- .Call("CatBoostPredictMulti_R", model$cpp_obj$handle, pool,
                        verbose, prediction_type, ntree_start, ntree_end, thread_count)
    if (length(prediction) != nrow(pool)) {
        prediction <- matrix(prediction, nrow = nrow(pool), byrow = TRUE)
    }
    return(prediction)
}


#' @name catboost.staged_predict
#' @title Apply the model for each tree
#'
#' @description Apply the model to the given dataset and calculate the results for each i-th tree of the model
#'              taking into consideration only the trees in the range [1;i].
#'
#'              Peculiarities: In case of multiclassification the prediction is returned in the form of a matrix.
#'              Each line of this matrix contains the predictions for one object of the input dataset.
#' @param model The model obtained as the result of training.
#'
#' Default value: Required argument
#' @param pool The input dataset.
#'
#' Default value: Required argument
#' @param verbose Verbose output to stdout.
#'
#' Default value: FALSE (not used)
#' @param prediction_type The format for displaying approximated values in output data
#' (see \url{https://catboost.ai/docs/concepts/output-data.html}).
#'
#' Possible values:
#' \itemize{
#'   \item 'Probability'
#'   \item 'Class'
#'   \item 'RawFormulaVal'
#' }
#'
#' Default value: 'RawFormulaVal'
#' @param ntree_start Model is applied on the interval [ntree_start, ntree_end) with the step eval_period (zero-based indexing).
#'
#' Default value: 0
#' @param ntree_end Model is applied on the interval [ntree_start, ntree_end) with the step eval_period (zero-based indexing).
#'
#' Default value: 0 (if value equals to 0 this parameter is ignored and ntree_end equal to tree_count)
#' @param eval_period Model is applied on the interval [ntree_start, ntree_end) with the step eval_period (zero-based indexing).
#'
#' Default value: 1
#' @param thread_count The number of threads to use when applying the model. If -1, then the number of threads is set to the number of CPU cores.
#'
#' Allows you to optimize the speed of execution. This parameter doesn't affect results.
#'
#' Default value: 1
#' @return List object with predictions from one iteration.
#' @export
#' @seealso \url{https://catboost.ai/docs/concepts/r-reference_catboost-staged_predict.html}
catboost.staged_predict <- function(model, pool, verbose = FALSE, prediction_type = "RawFormulaVal",
                                    ntree_start = 0L, ntree_end = 0L, eval_period = 1, thread_count = -1) {
    if (!inherits(model, "catboost.Model"))
        stop("Expected catboost.Model, got: ", class(model))
    if (!inherits(pool, "catboost.Pool"))
        stop("Expected catboost.Pool, got: ", class(pool))
    if (is.null.handle(pool))
        stop("Pool object is invalid.")
    if (ntree_end == 0L)
        ntree_end <- model$tree_count

    current_tree_count <- ntree_start
    approx <- 0
    preds <- function() {
        current_tree_count <<- current_tree_count + eval_period
        if (current_tree_count - eval_period >= ntree_end)
            stop("StopIteration")
        catboost.restore_handle(model)
        current_approx <- as.array(.Call("CatBoostPredictMulti_R", model$cpp_obj$handle, pool,
                                         verbose, "RawFormulaVal",
                                         current_tree_count - eval_period,
                                         min(current_tree_count, ntree_end), thread_count))
        approx <<- approx + current_approx
        prediction_columns <- length(approx) / nrow(pool)
        loss_function <- catboost.get_model_params(model)$loss_function$type
        prediction <- .Call("CatBoostPrepareEval_R", approx, prediction_type, loss_function, prediction_columns, thread_count)
        if (prediction_columns != 1) {
            prediction <- matrix(prediction, ncol = prediction_columns, byrow = TRUE)
        }
        return(prediction)
    }

    obj <- list(nextElem = preds)
    class(obj) <- c("catboost.staged_predict", "abstractiter", "iter")
    return(obj)
}

#' @name catboost.virtual_ensembles_predict
#' @title Apply the model with several virtual ensembles
#'
#' @description Apply the model to the given dataset using several independent truncated models - virtual ensembles. Each tree in
#'              ensemble predicts its own value for each document from pool.
#'
#'              Peculiarities: Return value varies on prediction_type: array for 'VirtEnsembles' and matrix for 'TotalUncertainty'
#' @param model The model obtained as the result of training.
#'
#' Default value: Required argument
#' @param pool The input dataset.
#'
#' Default value: Required argument
#' @param verbose Verbose output to stdout.
#'
#' Default value: FALSE (not used)
#' @param prediction_type The format for displaying approximated values in output data
#' (see \url{https://catboost.ai/docs/concepts/python-reference_virtual_ensembles_predict.html#python-reference_catboostclassifier_predict__output-format}).
#'
#' Possible values:
#' \itemize{
#'   \item 'VirtEnsembles'
#'   \item 'TotalUncertainty'
#' }
#'
#' Default value: 'VirtEnsembles'
#' @param ntree_end Index of the first tree not to be used when applying the model or calculating the metrics (zero-based indexing).
#'
#' Default value: 0 (the index of the last tree to use equals to the number of trees in the model minus one)
#' @param virtual_ensembles_count Number of tree ensembles to use. Each virtual ensemble can be considered as truncated model.
#'
#' Default value: 10
#' @param thread_count The number of threads to use when applying the model. If -1, then the number of threads is set to the number of CPU cores.
#'
#' Allows you to optimize the speed of execution. This parameter doesn't affect results.
#'
#' Default value: -1
#' @return Matrix or Array of predictions (for 'TotalUncertainty' and 'VirtEnsembles' prediction_type correspondingly)
#' @export
#' @seealso \url{https://catboost.ai/docs/concepts/python-reference_virtual_ensembles_predict.html?lang=en}
catboost.virtual_ensembles_predict <- function(model, pool, verbose = FALSE, prediction_type = "VirtEnsembles",
                                    ntree_end = 0L, virtual_ensembles_count = 10, thread_count = -1) {
    catboost.restore_handle(model)
    if (!inherits(pool, "catboost.Pool"))
        stop("Expected catboost.Pool, got: ", class(pool))
    if (is.null.handle(pool))
        stop("Pool object is invalid.")

    prediction <- .Call("CatBoostPredictVirtualEnsembles_R", model$cpp_obj$handle, pool,
                        verbose, prediction_type, ntree_end, virtual_ensembles_count, thread_count)
    objects_count <- nrow(pool)
    if (prediction_type == "VirtEnsembles") {
        document_predict_size <- length(prediction) / virtual_ensembles_count / objects_count
        prediction <- aperm(
                            array(prediction,
                                    dim = c(document_predict_size, virtual_ensembles_count, objects_count)),
                            perm = c(2, 1, 3))
    } else {
        # prediction_type == "TotalUncertainty"
        prediction <- matrix(prediction, nrow = objects_count, byrow = TRUE)
    }
    return(prediction)
}


#' @name catboost.get_feature_importance
#' @title Calculate the feature importances
#'
#' @description Calculate the feature importances (see \url{https://catboost.ai/docs/concepts/fstr.html#fstr})
#'              (Regular feature importance, ShapValues, and Feature interaction strength).
#'
#' @param model The model obtained as the result of training.
#'
#' Default value: Required argument
#' @param pool The input dataset.
#'
#' The feature importance for the training dataset is calculated if this argument is not specified.
#' Models with ranking metrics require pool argument to calculate feature importance.
#'
#' Default value: NULL
#' @param type The feature importance type.
#'
#' Possible values:
#' \itemize{
#'   \item 'PredictionValuesChange'
#'
#'     Calculate score for every feature.
#'
#'   \item 'LossFunctionChange'
#'
#'     Calculate score for every feature for groupwise model.
#'
#'   \item 'FeatureImportance'
#'
#'     'LossFunctionChange' in case of groupwise model and 'PredictionValuesChange' otherwise.
#'
#'   \item 'Interaction'
#'
#'     Calculate pairwise score between every feature.
#'
#'   \item 'ShapValues'
#'
#'     Calculate SHAP Values for every object.
#'
#' }
#'
#' Default value: 'FeatureImportance'
#' @param thread_count The number of threads to use when applying the model. If -1, then the number of threads is set to the number of CPU cores.
#'
#' Allows you to optimize the speed of execution. This parameter doesn't affect results.
#'
#' Default value: -1
#' @param fstr_type Deprecated parameter, use 'type' instead.
#' @return Feature importances
#' @export
#' @seealso \url{https://catboost.ai/docs/features/feature-importances-calculation.html}
catboost.get_feature_importance <- function(model, pool = NULL, type = "FeatureImportance", thread_count = -1, fstr_type = NULL) {
    if (!is.null(fstr_type)) {
        type <- fstr_type
    }
    if (!inherits(model, "catboost.Model"))
        stop("Expected catboost.Model, got: ", class(model))
    if (!is.null(pool) && class(pool) != "catboost.Pool")
        stop("Expected catboost.Pool, got: ", class(pool))
    if (!is.null(pool) && is.null.handle(pool))
        stop("Pool object is invalid.")
    if ( (type == "ShapValues" || type == "LossFunctionChange") && length(pool) == 0)
        stop("For `", type, "` type of feature importance, the pool is required")
    if ( (type == "PredictionValuesChange" || type == "FeatureImportance") && is.null(pool) && !is.null(model$feature_importances))
        return(model$feature_importances)

    catboost.restore_handle(model)
    importances <- .Call("CatBoostCalcRegularFeatureEffect_R", model$cpp_obj$handle, pool, type, thread_count)

    if (type == "Interaction") {
        colnames(importances) <- c("feature1_index", "feature2_index", "score")
    } else if (type == "ShapValues") {
        if (is.list(colnames(importances))) {
            dimnames(importances)[[length(dim(importances))]] <- c(colnames(pool), "<base>")
        }
    } else if (type == "PredictionValuesChange" || type == "FeatureImportance" || type == "LossFunctionChange") {
        # TODO: incorrect pool and ignored_features lead to incorrect column names; testing length is not enough
        if (!is.null(pool) && dim(importances)[1] == length(colnames(pool))) {
            rownames(importances) <- colnames(pool)
        }
    } else {
        stop("Unknown type: ", type)
    }
    return(importances)
}


#' @name catboost.get_object_importance
#' @title Calculate the object importances
#'
#' @description Calculate the object importances (see \url{https://catboost.ai/docs/concepts/ostr.html}).
#'              This is the implementation of the LeafInfluence algorithm from the following paper:
#'               https://arxiv.org/pdf/1802.06640.pdf
#'
#' @param model The model obtained as the result of training.
#'
#' Default value: Required argument
#' @param pool The pool for which you want to evaluate the object importances.
#'
#' Default value: Required argument
#' @param train_pool The pool on which the model has been trained.
#'
#' Default value: Required argument
#' @param top_size Method returns the result of the top_size most important train objects. If -1, then the top size is not limited.
#'
#' Default value: -1
#' @param type
#'
#' Possible values:
#' \itemize{
#'   \item 'Average'
#'
#'     Method returns the mean train objects scores for all input objects.
#'
#'   \item 'PerObject'
#'
#'     Method returns the train objects scores for every input object.
#' }
#'
#' Default value: 'Average'
#' @param update_method Description of the update set methods are given in section 3.1.3 of the paper.
#'
#' Possible values:
#' \itemize{
#'   \item 'SinglePoint'
#'   \item 'TopKLeaves'
#'     It is posible to set top size : TopKLeaves:top=2.
#'   \item 'AllPoints'
#' }
#'
#' Default value: 'SinglePoint'
#' @param thread_count The number of threads to use when applying the model. If -1, then the number of threads is set to the number of CPU cores.
#'
#' Allows you to optimize the speed of execution. This parameter doesn't affect results.
#'
#' Default value: -1
#' @param ostr_type Deprecated parameter, use 'type' instead.
#' @return List with elements \code{"indices"} and \code{"scores"}.
#' @export
#' @seealso \url{https://catboost.ai/docs/concepts/r-reference_catboost-get_object_importance.html}
catboost.get_object_importance <- function(
    model,
    pool,
    train_pool,
    top_size = -1,
    type = "Average",
    update_method = "SinglePoint",
    thread_count = -1,
    ostr_type = NULL
) {
    if (!inherits(pool, "catboost.Pool"))
        stop("Expected catboost.Pool, got: ", class(pool))
    if (!inherits(train_pool, "catboost.Pool"))
        stop("Expected catboost.Pool, got: ", class(train_pool))
    if (is.null.handle(pool))
        stop("'pool' object is invalid.")
    if (is.null.handle(train_pool))
        stop("'train_pool' object is invalid.")
    if (top_size < 0 && top_size != -1)
        stop("top_size should be positive integer or -1.")
    catboost.restore_handle(model)
    if (!is.null(ostr_type)) {
        type <- ostr_type
        warning("ostr_type option is deprecated, use type instead")
    }
    importances <- .Call("CatBoostEvaluateObjectImportances_R", model$cpp_obj$handle,
                         pool, train_pool, top_size, type, update_method, thread_count)
    indices <- head(importances, length(importances) / 2)
    scores <- tail(importances, length(importances) / 2)
    column_count <- nrow(train_pool)
    if (top_size != -1) {
        column_count <- min(column_count, top_size)
    }
    indices <- matrix(as.integer(indices), ncol = column_count, byrow = TRUE)
    scores <- matrix(scores, ncol = column_count, byrow = TRUE)

    return(list(indices = indices, scores = scores))
}


#' @name catboost.shrink
#' @title Shrink the model
#'
#' @param model The model obtained as the result of training.
#' @param ntree_end Leave the trees with indices from the interval [ntree_start, ntree_end) (zero-based indexing).
#' @param ntree_start Leave the trees with indices from the interval [ntree_start, ntree_end) (zero-based indexing).
#'
#' @return Status, the result of model shrinking. TRUE if shrinking succeeded, FALSE otherwise.
#' @export
#' @seealso \url{https://catboost.ai/docs/concepts/r-reference_catboost-shrink.html}
catboost.shrink <- function(model, ntree_end, ntree_start = 0) {
    if (ntree_start > ntree_end)
        stop("ntree_start should be less than ntree_end.")

    catboost.restore_handle(model)
    status <- .Call("CatBoostShrinkModel_R", model$cpp_obj$handle, ntree_start, ntree_end)
    model$cpp_obj$raw <- .Call("CatBoostSerializeModel_R", model$cpp_obj$handle)
    return(status)
}


#' @name catboost.drop_unused_features
#' @title Drop unused features information from model
#'
#' @param model The model obtained as the result of training.
#' @param ntree_end Leave the trees with indices from the interval [ntree_start, ntree_end) (zero-based indexing).
#' @param ntree_start Leave the trees with indices from the interval [ntree_start, ntree_end) (zero-based indexing).
#'
#' @return Status, the result of dropping feature. TRUE if this succeeded, FALSE otherwise.
#' @export
catboost.drop_unused_features <- function(model, ntree_end, ntree_start = 0) {
    catboost.restore_handle(model)
    status <- .Call("CatBoostDropUnusedFeaturesFromModel_R", model$cpp_obj$handle)
    model$cpp_obj$raw <- .Call("CatBoostSerializeModel_R", model$cpp_obj$handle)
    return(status)
}


catboost.ntrees <- function(model) {
    catboost.restore_handle(model)
    num_trees <- .Call("CatBoostGetNumTrees_R", model$cpp_obj$handle)
    return(num_trees)
}

catboost._is_groupwise_metric <- function(model) {
    catboost.restore_handle(model)
    is_groupwise_metric <- .Call("CatBoostIsGroupwiseMetric_R", model$cpp_obj$handle)
    return(is_groupwise_metric)
}


catboost._is_oblivious <- function(model) {
    catboost.restore_handle(model)
    is_oblivious <- .Call("CatBoostIsOblivious_R", model$cpp_obj$handle)
    return(is_oblivious)
}


#' @name catboost.get_model_params
#' @title Model parameters
#'
#' @description Return the model parameters.
#'
#' @param model The model obtained as the result of training.
#'
#' @return A list object with model parameters.
#' @export
#' @seealso \url{https://catboost.ai/docs/concepts/r-reference_catboost-get_model_params.html}
catboost.get_model_params <- function(model) {
    catboost.restore_handle(model)
    params <- .Call("CatBoostGetModelParams_R", model$cpp_obj$handle)
    params <- jsonlite::fromJSON(params)
    return(params)
}

#' @name catboost.get_plain_params
#' @title Plain Model parameters
#'
#' @description Return the plain model parameters.
#'
#' @param model he model obtained as the result of training.
#' @return A list object with model parameters.
#' @export
catboost.get_plain_params <- function(model) {
    catboost.restore_handle(model)
    params <- .Call("CatBoostGetPlainParams_R", model$cpp_obj$handle)
    params <- jsonlite::fromJSON(params)
    return(params)
}


#' @name catboost.eval_metrics
#' @title Calculate metrics.
#'
#' @description Calculate the specified metrics for the specified dataset.
#'
#' @param model The model obtained as the result of training.
#'
#' Default value: Required argument
#' @param pool The pool for which you want to evaluate the metrics.
#'
#' Default value: Required argument
#' @param metrics The list of metrics to be calculated.
#' (Supported metrics https://catboost.ai/docs/references/custom-metric__supported-metrics.html)
#'
#' Default value: Required argument
#' @param ntree_start Model is applied on the interval [ntree_start, ntree_end) with the step eval_period (zero-based indexing).
#'
#' Default value: 0
#' @param ntree_end Model is applied on the interval [ntree_start, ntree_end) with the step eval_period (zero-based indexing).
#'
#' Default value: 0 (if value equals to 0 this parameter is ignored and ntree_end equal to tree_count)
#' @param eval_period Model is applied on the interval [ntree_start, ntree_end) with the step eval_period (zero-based indexing).
#'
#' Default value: 1
#' @param thread_count The number of threads to use when applying the model.
#' If -1, then the number of threads is set to the number of CPU cores.
#'
#' Allows you to optimize the speed of execution. This parameter doesn't affect results.
#'
#' Default value: -1
#' @param tmp_dir  The name of the temporary directory for intermediate results.
#' If NULL, then the name will be generated.
#'
#' Default value: NULL
#' @return dict: metric -> array of shape [(ntree_end - ntree_start) / eval_period].
#' @export
#' @seealso \url{https://catboost.ai/docs/concepts/python-reference_catboost_eval-metrics.html}
catboost.eval_metrics <- function(model, pool, metrics, ntree_start = 0L, ntree_end = 0L,
                                  eval_period = 1, thread_count = -1, tmp_dir = NULL) {
  catboost.restore_handle(model)
  if (!inherits(pool, "catboost.Pool"))
    stop("Expected catboost.Pool, got: ", class(pool))
  if (is.null.handle(pool))
    stop("Pool object is invalid.")
  if (ntree_start < 0)
    stop("ntree_start should be greater or equal zero.")
  if (ntree_end == 0L) {
    ntree_end <- model$tree_count
  } else {
    ntree_end <- min(c(ntree_end, model$tree_count))
  }
  if (ntree_start >= ntree_end)
    stop("ntree_start should be less than ntree_end.")
  if (eval_period <= 0)
    stop("eval_period should be greater than zero.")
  if (eval_period > (ntree_end - ntree_start))
    eval_period <- ntree_end - ntree_start
  if (!is.list(metrics) && !(is.character(metrics) && length(metrics) == 1))
    stop("Unsupported metrics type, expecting list or string, got: ", typeof(metrics))
  if (is.character(metrics) && length(metrics) == 1)
    metrics <- list(metrics)
  if (length(metrics) == 0)
    stop("No metrics found.")
  if (is.null(tmp_dir))
    tmp_dir <- tempdir()
  tmp_dir <- path.expand(tmp_dir)

  params <- catboost.get_plain_params(model)
  train_dir <- params[['train_dir']]
  if (is.null(params[['train_dir']]))
    train_dir <- 'catboost_info'
  result <- .Call("CatBoostEvalMetrics_R", model$cpp_obj$handle, pool, metrics,
                  ntree_start, ntree_end, eval_period,
                  thread_count, tmp_dir, train_dir)

  return(result)
}


#' @name catboost.restore_handle
#' @title Restore or complete model handle after de-serializing
#'
#' @description After de-serializing a model object through R base's functions (`readRDS`, `load`),
#'              its underlying object will not exist in the computer's memory anymore, and needs
#'              to be restored from the raw bytes that the model stores.
#'
#'              This is automatically done internally when calling functions such as \link{catboost.predict},
#'              but the process is repeated at each call, which makes them slower than if using a
#'              fresh model object and increases memory usage inbetween calls to the garbage collector.
#'              This function allows restoring the internal object beforehand so as to avoid
#'              restoring the object multiple times.
#'
#'              Note that the model object needs to be re-assigned as the output of this function,
#'              as the modifications are not done in-place.
#'
#' @param model The model obtained as the result of training which has been serialized and is
#'              now de-serialized.
#'
#' @return The model object with its handle pointing to a valid object in memory.
#' @export
catboost.restore_handle <- function(model) {
    if (!inherits(model, "catboost.Model"))
        stop("Expected catboost.Model, got: ", class(model))
    if (is.null.handle(model$cpp_obj$handle))
        model$cpp_obj$handle <- .Call("CatBoostDeserializeModel_R", model$cpp_obj$raw)
    return(model)
}

is.null.handle <- function(handle) {
  stopifnot(typeof(handle) == "externalptr")
  .Call("CatBoostIsNullHandle_R", handle)
}

create.model.base <- function(handle, raw) {
    model <- list(cpp_obj = as.environment(list(handle = handle, raw = raw)))
    class(model) <- "catboost.Model"
    return(model)
}
