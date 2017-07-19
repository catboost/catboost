library(caret)
library(titanic)
library(catboost)

set.seed(12345)

data <- as.data.frame(as.matrix(titanic_train), stringsAsFactors=TRUE)

drop_columns = c("PassengerId", "Survived", "Name", "Ticket", "Cabin")
x <- data[,!(names(data) %in% drop_columns)]
y <- data[,c("Survived")]

fit_control <- trainControl(method = "cv",
                            number = 4,
                            classProbs = TRUE)

grid <- expand.grid(depth = c(4, 6, 8),
                    learning_rate = 0.1,
                    iterations = 100,
                    l2_leaf_reg = 1e-3,
                    rsm = 0.95,
                    border_count = 64)

report <- train(x, as.factor(make.names(y)),
                method = catboost.caret,
                verbose = TRUE, preProc = NULL,
                tuneGrid = grid, trControl = fit_control)

print(report)

importance <- varImp(report, scale = FALSE)
print(importance)
