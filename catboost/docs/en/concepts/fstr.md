# Feature importance

## {{ title__regular-feature-importance-PredictionValuesChange }} {#regular-feature-importance}

{% include [reusage-formats-regular-feature-importance-desc](../_includes/work_src/reusage-formats/regular-feature-importance-desc.md) %}


For each feature, {{ title__regular-feature-importance-PredictionValuesChange }} shows how much on average the prediction changes if the feature value changes. The bigger the value of the importance the bigger on average is the change to the prediction value, if this feature is changed.

{% include [reusage-formats-see-featureimportance-file-format](../_includes/work_src/reusage-formats/see-featureimportance-file-format.md) %}


{% cut "{{ title__fstr__calculation-principle }}" %}


Leaf pairs that are compared have different split values in the node on the path to these leaves. If the split condition is met (this condition depends on the feature F), the object goes to the left subtree; otherwise it goes to the right one.

$feature\_importance_{F} = \displaystyle\sum\limits_{trees, leafs_{F}} \left(v_{1} - avr \right)^{2} \cdot c_{1} + \left( v_{2} - avr \right)^{2} \cdot c_{2} { , }$

$avr = \displaystyle\frac{v_{1} \cdot c_{1} + v_{2} \cdot c_{2}}{c_{1} + c_{2}} { , where}$

- $c_1, c_2$ represent the total weight of objects in the left and right leaves respectively. This weight is equal to the number of objects in each leaf if weights are not specified for the dataset.
- $v_1, v_2$ represent the formula value in the left and right leaves respectively.

If the model uses a combination of some of the input features instead of using them individually, an average feature importance for these features is calculated and output. For example, the model uses a combination of features `f54`, `c56` and `f77`. First, the feature importance is calculated for the combination of these features. Then the resulting value is divided by three and is assigned to each of the features.

If the model uses a feature both individually and in a combination with other features, the total importance value of this feature is defined using the following formula:

$feature\_total\_importance_{j} = feature\_importance + \sum\limits_{i=1}^{N}average\_feature\_importance_{i} { , where}$

- $feature\_importance_{j}$ is the individual feature importance of the j-th feature.
- $average\_feature\_importance_{i}$ is the average feature importance of the j-th feature in the i-th combinational feature.


{% endcut %}


{% cut "{{ title__fstr__complexity }}" %}



$O(trees\_count \cdot depth \cdot 2 ^ {depth} \cdot dimension)$

{% endcut %}



#### {{ input_data__title__peculiarities }}

- Feature importance values are normalized so that the sum of importances of all features is equal to 100. This is possible because the values of these importances are always non-negative.

- Formula values inside different groups may vary significantly in ranking modes. This might lead to high importance values for some groupwise features, even though these features don't have a large impact on the resulting metric value.

## {{ title__regular-feature-importance-LossFunctionChange }} {#regular-feature-importances__lossfunctionchange}

{% include [reusage-formats-regular-feature-importance__lossfunctionchange__-desc](../_includes/work_src/reusage-formats/regular-feature-importance__lossfunctionchange__-desc.md) %}


For each feature the value represents the difference between the loss value of the model with this feature and without it. The model without this feature is equivalent to the one that would have been trained if this feature was excluded from the dataset. Since it is computationally expensive to retrain the model without one of the features, this model is built approximately using the original model with this feature removed from all the trees in the ensemble. The calculation of this feature importance requires a dataset and, therefore, the calculated value is dataset-dependent.

{% include [reusage-formats-see-featureimportance-file-format](../_includes/work_src/reusage-formats/see-featureimportance-file-format.md) %}


{% cut "{{ title__fstr__calculation-principle }}" %}


Depend on the conditions for achieving the best metric value to ensure that the more important is a feature, the higher is the corresponding importance value.
- Minimum/maximum best value metric:

    $feature\_importance_{i} = \pm (metric E_{i}v) - metric(v))$

- Exact best value metric:

    $feature\_importance_{i} = abs(metric(E_{i}v) - best\_value) - abs(metric(v) - best\_value)$

In general, the value of {{ title__regular-feature-importance-LossFunctionChange }} can be negative.

Variables description:
- $E_{i}v$ is the mathematical expectation of the formula value without the $i$-th feature. If the feature $i$ is on the path to a leaf, the new leaf value is set to the weighted average of values of leaves that have different paths by feature value. Weights represent the total weight of objects in the corresponding leaf. This weight is equal to the number of objects in each leaf if weights are not specified for the dataset.

    For feature combinations ($F = (f_{1}, ..., f_{n})$) the average value on a leaf is calculated as follows:

    $E_fv = \displaystyle\left(\frac{(n - 1) v + E_{f}v}{n}\right)$

- $v$ is the vector with formula values for the dataset. The values of the training dataset are used if both training and validation datasets are provided.
- metric is the loss function specified in the training parameters.

The pool random subset size used for calculation is determined as follows:

$subset\_size = min(documentCount, max(2e5, \frac{2e9}{featureCount}))$

{% endcut %}


{% cut "{{ title__fstr__complexity }}" %}


$O(trees\_count \cdot (depth + sub\_samples\_count) \cdot 2 ^ {depth} +$

$+ Eval\_metric\_complexity(model, sub\_samples\_count) \cdot features\_count) { ,}$

$sub\_samples\_count = Min(samples\_count, Max(10^5, 10^9 / features\_count))$

{% endcut %}


This feature importance approximates the difference between metric values calculated on the following models:

- The model with the $i$-th feature excluded.
- The original model with all features.


## {{ title__internal-feature-importance }} {#internal-feature-importance}

{% include [reusage-formats-internal-feature-importance-desc](../_includes/work_src/reusage-formats/internal-feature-importance-desc.md) %}


See the [{{ title__internal-feature-importance }}](output-data_feature-analysis_feature-importance.md#internal-feature-importance) file format.


{% cut "{{ title__fstr__calculation-principle }}" %}


Leaf pairs that are compared have different split values in the node on the path to these leaves. If the split condition is met (this condition depends on the feature F), the object goes to the left subtree; otherwise it goes to the right one.

$feature\_importance_{F} = \displaystyle\sum\limits_{trees, leafs_{F}} \left(v_{1} - avr \right)^{2} \cdot c_{1} + \left( v_{2} - avr \right)^{2} \cdot c_{2} { , }$

$avr = \displaystyle\frac{v_{1} \cdot c_{1} + v_{2} \cdot c_{2}}{c_{1} + c_{2}} { , where}$

- $c_{1}, c_{2}$ represent the total weight of objects in the left and right leaves respectively. This weight is equal to the number of objects in each leaf if weights are not specified for the dataset.
- $v_{1}, v_{2}$ represent the formula value in the left and right leaves respectively.

If the model uses a combination of some of the input features instead of using them individually, an average feature importance for these features is calculated and output. For example, the model uses a combination of features `f54`, `c56` and `f77`. First, the feature importance is calculated for the combination of these features. Then the resulting value is divided by three and is assigned to each of the features.

If the model uses a feature both individually and in a combination with other features, the total importance value of this feature is defined using the following formula:

$feature\_total\_importance_{j} = feature\_importance + \sum\limits_{i=1}^{N}average\_feature\_importance_{i} { , where}$

- $feature\_importance_{j}$ is the individual feature importance of the j-th feature.
- $average\_feature\_importance_{i}$ is the average feature importance of the j-th feature in the i-th combinational feature.

{% endcut %}


{% cut "{{ title__fstr__complexity }}" %}


$O(trees\_count \cdot depth \cdot 2 ^ {depth} \cdot dimension)$

{% endcut %}


## {{ title__predictiondiff }} {#prediction-diff}

The impact of a feature on the prediction results for a pair of objects. This type of feature importance is designed for analyzing the reasons for wrong ranking in a pair of documents, but it also can be used for any one-dimensional model.

For each feature {{ title__predictiondiff }} reflects the maximum possible change in the predictions difference if the value of the feature is changed for both objects. The change is considered only if there is an improvement in the direction of changing the order of documents.

{% include [plot_predictions-plot_predictions__restriction](../_includes/work_src/reusage-python/plot_predictions__restriction.md) %}


#### Related information
[Detailed information regarding usage specifics for different Catboost implementations.](../features/feature-importances-calculation.md#feature-importances-calculation)
