library(caret)
require(catboost)

set.seed(12345)

target_ <- sample(c(1, 2, 3), size = 1000, replace = TRUE)

data <- data.frame(f_numeric = target_ + rnorm(length(target_), mean = 0, sd = 1),
                   f_logical = (target_ + rnorm(length(target_), mean = 0, sd = 1)) > 0,
                   f_factor = as.factor(round(10 * (target_ + rnorm(length(target_), mean = 0, sd = 1)))),
                   f_character = as.character(round(10 * (target_ + rnorm(length(target_), mean = 0, sd = 1)))))

data$f_logical = as.factor(data$f_logical)
data$f_character = as.factor(data$f_character)

data$target <- as.factor(make.names(target_))

fit_control <- trainControl(method = "cv",
                            number = 4,
                            classProbs = TRUE)

grid <- expand.grid(depth = c(4, 6, 8),
                    learning_rate = c(0.01, 0.1, 0.2),
                    iterations = 10)

report <- train(target ~ f_numeric + f_logical + f_factor,
                data = data,
                method = catboost.caret,
                verbose = FALSE,
                preProc = NULL,
                tuneGrid = grid,
                trControl = fit_control)

print(report)

importance <- varImp(report, scale = FALSE)
print(importance)
