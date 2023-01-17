
Calculate the effect of objects from the train dataset on the optimized metric values for the objects from the input dataset:
- Positive values reflect that the optimized metric increases.
- Negative values reflect that the optimized metric decreases.

The higher the deviation from 0, the bigger the impact that an object has on the optimized metric.

The method is an implementation of the approach described in the [Finding Influential Training Samples for Gradient Boosted Decision Trees](https://arxiv.org/abs/1802.06640) paper .

Currently, object importance is supported only for the following loss functions.

{{ error-function--Logit }}

{{ error-function--CrossEntropy }}

{{ error-function--RMSE }}

{{ error-function--MAE }}

{{ error-function--Quantile }}

{{ error-function__Expectile }}

{{ error-function--LogLinQuantile }}

{{ error-function--MAPE }}

{{ error-function--Poisson }}
