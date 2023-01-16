
Calculate the object strength:
```r
library(catboost)

train_dataset = matrix(c(1900,7,1,
                         1896,1,1),
                        nrow=2, 
                        ncol=3, 
                        byrow = TRUE)

label_values = c(0, 1)

train_pool = catboost.load_pool(train_dataset, 
                                label_values)

input_dataset = matrix(c(1900,47,1,
                         1904,27,1),
                 nrow=2, 
                 ncol=3, 
                 byrow = TRUE)

input_pool = catboost.load_pool(input_dataset, 
                                label_values)

trained_model <- catboost.train(train_pool,
                               params = list(iterations = 10))

object_importance <- catboost.get_object_importance(trained_model,
                                                    input_pool,
                                                    train_pool)
```
