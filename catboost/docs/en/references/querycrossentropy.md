# {{ error-function__QueryCrossEntropy }}

Let's assume that it is required to solve a classification problem on a dataset with grouped objects. For example, it may be required to predict user clicks on a search engine results page.

Generally, this task can be solved by the {{ error-function--Logit }} function:
$Logloss = \displaystyle\frac{1}{\sum\limits_{i = 1}^{N} w_{i}} \sum_{group} \left( \sum_{obj\_in\_group}  w_{i} \left(t_{i} \cdot log(p_{i}) + (1 - t_{i}) \cdot log(1 - p_{i}) \right) \right)$
- $t_{i}$ is the label value for the i-th object (from the input data for training). Possible values are in the range $[0;1]$.
- $a_{i}$ is the {{ error-function--Logit }} raw formula prediction.
- $p_{i}$ is the predicted probability that the object belongs to the positive class. $p_i = \sigma(a_{i})$ (refer to the [Logistic function, odds, odds ratio, and logit](https://en.wikipedia.org/wiki/Logistic_regression#Logistic_function,_odds,_odds_ratio,_and_logit) section of the [Logistic regression](https://en.wikipedia.org/wiki/Logistic_regression) article in Wikipedia for details).

Since the internal structure of the data is known, it can be assumed that the predictions in various groups are different. This can be modeled by adding a $shift\_group$ to each formula prediction for a group:
$\bar p_{i} = \sigma(a_{i} + group\_shift)$
The $shift\_group$ parameter is jointly optimized for each group during the training.

In this case, the {{ error-function--Logit }} formula for grouped objects takes the following form:
$Logloss_{group} =  \displaystyle\frac{1}{\sum\limits_{i = 1}^{N} w_{i}} \sum_{group} \left( \sum_{obj\_in\_group}  w_{i} \left( t_{i} \cdot log({{\bar p_{i}}} ) + (1 - t_{i}) \cdot log(1 - {{\bar p_i}} ) \right) \right)$
The {{ error-function__QueryCrossEntropy }} metric is calculated as follows:
$QueryCrossEntropy(\alpha) = (1 - \alpha) \cdot LogLoss + \alpha \cdot LogLoss_{group}$

## {{ title__loss-functions__text__user-defined-params }} {#user-defined-parameters}
**Parameter:** {{ loss-functions__params__alpha }}

#### Description

 The coefficient used in quantile-based losses. Defines the rules for mixing the
{% cut "regular" %}

$Logloss = \displaystyle\frac{1}{\sum\limits_{i = 1}^{N} w_{i}} \sum_{group} \left( \sum_{obj\_in\_group}  w_{i} \left(t_{i} \cdot log(p_{i}) + (1 - t_{i}) \cdot log(1 - p_{i}) \right) \right)$

{% endcut %}

 and
{% cut "shifted" %}

$Logloss_{group} =  \displaystyle\frac{1}{\sum\limits_{i = 1}^{N} w_{i}} \sum_{group} \left( \sum_{obj\_in\_group}  w_{i} \left( t_{i} \cdot log({{\bar p_{i}}} ) + (1 - t_{i}) \cdot log(1 - {{\bar p_i}} ) \right) \right)$

{% endcut %}

 versions of the {{ error-function--Logit }} function.

