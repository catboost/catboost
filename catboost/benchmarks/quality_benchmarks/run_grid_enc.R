library(h2o)
library(KScorrect)
library(reshape2)
Sys.unsetenv("http_proxy")

###
# Load data, classification
###

cd <- read.table(file="train_full3.cd", head=F, sep="\t")
colnames(cd) <- c("index", "type")
cat_ind <- 1 + cd[cd$type == "Categ", 1]

train <- read.table(file="train_full3", head=F, sep="\t")
train[,1] <- as.factor(train[,1])
train[,cat_ind] <- lapply(train[,cat_ind], factor)

valid <- read.table(file="test3", head=F, sep="\t")
valid[,1] <- as.factor(valid[,1])
valid[,cat_ind] <- lapply(valid[,cat_ind], factor)

y <- "V1"
x <- setdiff(names(train), c(y))

port = sample(0:65536, 1)
localH2O = h2o.init(nthreads=-1, max_mem_size='60G', port=port)
h2_train <- as.h2o(train)
h2_valid <- as.h2o(valid)

###
# Grid search params
###

seed = 12345
seeds = c(12, 23, 34, 45, 56)
set.seed(seed)
n_models = 50
n_sample = 10
n_trees = 5000
metric = 'logloss'
histogram_type_options = c('uniform_adaptive', 'random', 'quantiles_global', 'round_robin')
categorical_encoding_options = c('Enum', 'OneHotExplicit', 'Binary', 'Eigen', 'LabelEncoder', 'SortByResponse')

gbm_params_default <- list(learn_rate = 0.1,
                           max_depth = 5,
                           sample_rate = 1.0,
                           col_sample_rate = 1.0,
                           col_sample_rate_change_per_level = 1,
                           col_sample_rate_per_tree = 1,
                           min_split_improvement = 1e-5,
                           min_rows = 10,
                           histogram_type = 'auto',
                           categorical_encoding = 'AUTO')

gbm_params_random <- list(learn_rate = rlunif(n_sample, exp(-7), exp(0)),
                          max_depth = round(runif(n_sample, 2, 10)/1)*1,
                          sample_rate = runif(n_sample, 0.5, 1),
                          col_sample_rate = runif(n_sample, 0.5, 1),
                          col_sample_rate_change_per_level = runif(n_sample, 0, 2),
                          col_sample_rate_per_tree = runif(n_sample, 0, 1),
                          min_split_improvement = rlunif(n_sample, exp(-16), exp(0)),
                          min_rows = rlunif(n_sample, exp(0), exp(5)),
                          histogram_type = histogram_type_options,
                          categorical_encoding = categorical_encoding_options)

gbm_search_criteria <- list(strategy = "RandomDiscrete",
                             max_models = n_models,
                             max_runtime_secs = 1200000)

###
# Functions
###

### Extract param list from a model

get_best_params <- function(model){
    params <- list(learn_rate = model@allparameters$learn_rate,
                   max_depth = model@allparameters$max_depth,
                   sample_rate = model@allparameters$sample_rate,
                   col_sample_rate = model@allparameters$col_sample_rate,
                   col_sample_rate_change_per_level = model@allparameters$col_sample_rate_change_per_level,
                   col_sample_rate_per_tree = model@allparameters$col_sample_rate_per_tree,
                   min_split_improvement = model@allparameters$min_split_improvement,
                   min_rows = model@allparameters$min_rows,
                   histogram_type = model@allparameters$histogram_type,
                   categorical_encoding = model@allparameters$categorical_encoding)
    return(params)
}

### Choose the best number of trees using all CV results and the target metric

get_best_cv <- function(model, nfolds = 5){
    scoring_history_stats <- data.frame(matrix(NA,0,0))
    for (i in 1:nfolds){
        cv_model <- h2o.getModel(model@model$cross_validation_models[[i]]$name)
        scoring_history <- data.frame(cv_model@model$scoring_history)[,c('number_of_trees', 'validation_logloss')]
        scoring_history_stats <- rbind(scoring_history_stats, scoring_history)
    }
    scoring_history_avg <- dcast(scoring_history_stats, number_of_trees ~ ., mean, value.var = "validation_logloss")
    colnames(scoring_history_avg) <- c('n_trees', 'logloss')
    min_logloss <- min(scoring_history_avg$logloss)
    best_n_trees <- scoring_history_avg[scoring_history_avg$logloss==min_logloss,]$n_trees
    return(list(logloss_cv = min_logloss,
                n_trees = best_n_trees))
}

### Iterate over all grid models, choose the best model using the target metric

get_best_model <- function(gbm_cv_perf){
    best_logloss = Inf
    for (i in 1:length(gbm_cv_perf@model_ids)){
        model <- h2o.getModel(gbm_cv_perf@model_ids[[i]])
        model_cv <- get_best_cv(model)
        model_params <- get_best_params(model)
        if (model_cv$logloss_cv < best_logloss){
            best_logloss <- model_cv$logloss_cv
            best_cv <- model_cv
            best_params <- model_params
        }
    }
    return(list(best_params = best_params,
                best_cv = best_cv))
}

### Load model params from result_(default|tuned).tsv

load_params <- function(file){
    tsv_default <- read.table(file=file, head=F, sep="\t", fill=T, colClasses = "character")
    params <- tsv_default[2:11,]
    param_list <- vector("list", nrow(params))
    for (i in 1:nrow(params)) {
        param_list[[i]] <- params[i,2]
    }
    names(param_list) <- params[,1]
    n_trees = as.numeric(tsv_default[11,2])
    return(list(param_list = param_list,
                n_trees = n_trees))
}

### Train 1 model using grid method, use validation frame to calculate metrics

validate_model <- function(x, y, train, validate, params, n_trees, seed){
    grid_id = paste0("gbm_validate", seed)
    gbm_validate <- h2o.grid("gbm", x = x, y = y,
                            grid_id = grid_id,
                            training_frame = train,
                            validation_frame = validate,
                            ntrees = n_trees,
                            seed = seed,
                            hyper_params = params)
    gbm_validate_perf <- h2o.getGrid(grid_id = grid_id,
                                    sort_by = metric,
                                    decreasing = FALSE)
    model <- h2o.getModel(gbm_validate_perf@model_ids[[1]])
    model_metrics <- model@model$validation_metrics
    logloss_val <- model_metrics@metrics$logloss
    return(logloss_val)
}



###
# Train with default parameters
###

### CV to get the best ntrees

gbm_default_cv <- h2o.grid("gbm", x = x, y = y,
                            grid_id = "gbm_default_cv",
                            training_frame = h2_train,
                            nfolds = 5,
                            keep_cross_validation_predictions = T,
                            fold_assignment = "AUTO",
                            ntrees = n_trees,
                            seed = seed,
                            hyper_params = gbm_params_default)

gbm_default_cv_perf <- h2o.getGrid(grid_id = "gbm_default_cv",
                                   sort_by = metric,
                                   decreasing = FALSE)

model <- h2o.getModel(gbm_default_cv_perf@model_ids[[1]])
best_cv <- get_best_cv(model)
best_params <- get_best_params(model)

### Train model with the best parameters and ntrees using different random seeds

seeds_default <- c()
for (seed in seeds) {
    set.seed(seed)
    default_logloss_val <- validate_model(x, y, h2_train, h2_valid, best_params, best_cv$n_trees, seed)
    seeds_default <- c(seeds_default, default_logloss_val)
}
seeds_default <- c(seeds_default, mean(seeds_default), sd(seeds_default))
seeds_default <- as.data.frame(seeds_default, row.names = c(sapply(seeds, as.character), "mean", "sd"))
logloss_val_mean <- seeds_default["mean",]

### Save results

result <- data.frame(unlist(best_params), stringsAsFactors = FALSE)
result <- rbind(result,
                n_trees = best_cv$n_trees,
                logloss_cv = best_cv$logloss_cv,
                logloss_val = logloss_val_mean)
colnames(result) <- "Default"

write.table(result, file='result_default_enc.tsv', quote=FALSE, sep='\t')
write.table(seeds_default, file='result_default_enc_seeds.tsv', quote=FALSE, sep='\t')
cat("Default finished\n")

###
# Random hyperparameter search, random subspace
###

### CV to get the best ntrees

gbm_random_cv <- h2o.grid("gbm", x = x, y = y,
                            grid_id = "gbm_random_cv",
                            training_frame = h2_train,
                            search_criteria = gbm_search_criteria,
                            ntrees = n_trees,
                            nfolds = 5,
                            keep_cross_validation_predictions = T,
                            fold_assignment = "AUTO",
                            seed = seed,
                            hyper_params = gbm_params_random)

gbm_random_cv_perf <- h2o.getGrid(grid_id = "gbm_random_cv",
                                  sort_by = metric,
                                  decreasing = FALSE)

best_model <- get_best_model(gbm_random_cv_perf)

### Train model with the best parameters and ntrees using different random seeds

seeds_tuned <- c()
for (seed in seeds) {
    set.seed(seed)
    tuned_logloss_val <- validate_model(x, y, h2_train, h2_valid, best_model$best_params, best_model$best_cv$n_trees, seed)
    seeds_tuned <- c(seeds_tuned, tuned_logloss_val)
}
seeds_tuned <- c(seeds_tuned, mean(seeds_tuned), sd(seeds_tuned))
seeds_tuned <- as.data.frame(seeds_tuned, row.names = c(sapply(seeds, as.character), "mean", "sd"))
logloss_val_mean <- seeds_tuned["mean",]

### Save results

result <- data.frame(unlist(best_model$best_params), stringsAsFactors = FALSE)
result <- rbind(result,
                n_trees = best_model$best_cv$n_trees,
                logloss_cv = best_model$best_cv$logloss_cv,
                logloss_val = logloss_val_mean)
colnames(result) <- "Tuned"

write.table(result, file='result_tuned_enc.tsv', quote=FALSE, sep='\t')
write.table(seeds_tuned, file='result_tuned_enc_seeds.tsv', quote=FALSE, sep='\t')
cat("Tuned finished\n")
