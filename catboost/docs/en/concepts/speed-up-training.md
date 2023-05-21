# Speeding up the training

{{ product }} provides several settings that can speed up the training.

{% note info %}

Certain changes to these parameters can decrease the quality of the resulting model.

{% endnote %}

## Training on {{ calcer_type__gpu }} {#training-on-gpu}

If the dataset is large enough (starting from tens of thousands of objects), training on {{ calcer_type__gpu }} gives a significant speedup compared to training on {{ calcer_type__cpu }}. The larger the dataset, the more significant is the speedup. For example, the speedup for training on datasets with millions of objects on Volta GPUs is around 40-50 times.


## Iterations and learning rate {#iterations-and-learning-rate}

By default, {{ product }} builds {{ fit--iterations }} trees. The number of iterations can be decreased to speed up the training.

When the number of iterations decreases, the learning rate needs to be increased. By default, the value of the learning rate is defined automatically depending on the number of iterations and the input dataset. Changing the number of iterations to a smaller value is a good starting point for optimization.

The default learning rate is close to optimal one, but it can be tuned to get the best possible quality. Look at evaluation metric values on each iteration to tune the learning rate:
- Decrease the learning rate if overfitting is observed.
- Increase the learning rate if there is no overfitting and the error on the evaluation dataset still reduces on the last iteration.


{% cut "{{ dl--parameters }}" %}

Command-line version parameters|     Python parameters                                    | R parameters                |
-----------------------------|------------------------------------------------------------|-----------------------------|
`-i`<br/><br/>`--iterations` | `iterations`<br/><br/>_Aliases:_<br/>- `num_boost_round`<br/>- `n_estimators`<br/>- `num_trees` | `iterations`
`-w`<br/><br/>`--learning-rate` | `learning_rate`<br/><br/>_Alias:_`eta` | `learning_rate`

{% endcut %}

## Boosting type {#boosting-type}

By default, the boosting type is set to  for small datasets. This prevents overfitting but it is expensive in terms of computation. Try to set the value of this parameter to  to speed up the training.

{% cut "{{ dl--parameters }}" %}

Command-line version parameters|Python parameters|R parameters|
------------------|-----------------|---------------|
`--boosting-type` | `boosting_type` | `boosting_type`

{% endcut %}


## Bootstrap type {#bootstrap-type}

By default, the method for sampling the weights of objects is set to . The training is performed faster if the  method is set and the value for the sample rate for bagging is smaller than 1.

{% cut "{{ dl--parameters }}" %}

Command-line version parameters|Python parameters|R parameters|
------------------|-----------------|---------------|
`--bootstrap-type` | `bootstrap_type` | `bootstrap_type`
`--subsample` | `subsample` | `subsample`

{% endcut %}


## One-hot encoding {#one-hot-encoding}

{% include [one-hot-encoding-one-hot-encoding__default-intro](../_includes/work_src/reusage-common-phrases/one-hot-encoding__default-intro.md) %}


- N/A if training is performed on CPU in Pairwise scoring  mode


{% cut "Read more about  Pairwise scoring" %}


{% include [reusage-default-values-metrics_parwise_scoring](../_includes/work_src/reusage-default-values/metrics_parwise_scoring.md) %}


{% endcut %}


- 255 if training is performed on GPU and the selected Ctr types require target data that is not available during the training
- 10 if training is performed in [Ranking](../concepts/loss-functions-ranking.md) mode
- 2 if none of the conditions above is met

Statistics are calculated for all other categorical features. This is more time consuming than using one-hot encoding.

Set a larger value for this parameter to speed up the training.


{% cut "{{ dl--parameters }}" %}

Command-line version parameters|Python parameters|R parameters|
------------------|-----------------|---------------|
`--one-hot-max-size` | `one_hot_max_size` | `one_hot_max_size`

{% endcut %}


## Random subspace method {#rsm}

For datasets with hundreds of features this parameter speeds up the training and usually does not affect the quality. It is not recommended to change the default value of this parameter for datasets with few (10-20) features.

For example, set the parameter to <q>0.1</q>. In this case, the training requires roughly 20% more iterations to converge. But each iteration is performed roughly ten times faster. Therefore, the training time is much shorter even though the resulting model contains more trees.

{% cut "{{ dl--parameters }}" %}

Command-line version parameters|Python parameters|R parameters|
------------------|-----------------|---------------|
`--rsm` | `rsm`<br/><br/>_Alias:_`colsample_bylevel` | `rsm`

{% endcut %}


## Leaf estimation iterations {#leaf-estimation-iteration}

This parameter defines the rules for calculating leaf values after selecting the tree structures. The default value depends on the training objective and can slow down the training for datasets with a small number of features (for example, 10 features).

Try setting the value to <q>1</q> or <q>5</q> to speed up the training on datasets with a small number of features.

{% cut "{{ dl--parameters }}" %}

Command-line version parameters|Python parameters|R parameters|
------------------|-----------------|---------------|
 `--leaf-estimation-iterations` | `leaf_estimation_iterations` | `leaf_estimation_iterations`

{% endcut %}

## Number of categorical features to combine {#max-ctr}

By default, the combinations of categorical features are generated in a greedy way. This slows down the training.

Try turning off the generation of categorical feature combinations or limiting the number of categorical features that can be combined to two to speed up the training.

This parameter can affect the training time only if the dataset contains categorical features.


{% cut "{{ dl--parameters }}" %}

Command-line version parameters|Python parameters|R parameters|
------------------|-----------------|---------------|
`--max-ctr-complexity` | `max_ctr_complexity` | `max_ctr_complexity`

{% endcut %}


## Number of splits for numerical features {#splis-numerical-features}

This parameter defines the number of splits considered for each feature.

{% include [reusage-default-values-border_count](../_includes/work_src/reusage-default-values/border_count.md) %}


{% include [parameter-tuning-border-count__how-affects-the-speed](../_includes/work_src/reusage-common-phrases/border-count__how-affects-the-speed.md) %}


Try to set the value of this parameter to 32 if training is performed on GPU. In many cases, this does not affect the quality of the model but significantly speeds up the training.

{% include [parameter-tuning-border-count__how-affects-the-speed-cpu](../_includes/work_src/reusage-common-phrases/border-count__how-affects-the-speed-cpu.md) %}


{% cut "{{ dl--parameters }}" %}

Command-line version parameters|Python parameters|R parameters|
------------------|-----------------|---------------|
`-x`<br/><br/>`--border-count` | `border_count`<br/><br/>_Alias:_`max_bin` | `border_count`

{% endcut %}


## Reusing quantized datasets in Python {#reusing-quantized-datasets}

By default, the train and test datasets are quantized each time that the boosting is run.

If the dataset and quantization parameters are the same across multiple runs, the total wall clock time can be reduced by saving and reusing the quantized dataset. This optimization is applicable only for datasets without categorical features.

Example:
```python
import numpy as np
from catboost import Pool, CatBoostRegressor


train_data = np.random.randint(1, 100, size=(10000, 10))
train_labels = np.random.randint(2, size=(10000))
quantized_dataset_path = 'quantized_dataset.bin'

# save quantized dataset
train_dataset = Pool(train_data, train_labels)
train_dataset.quantize()
train_dataset.save(quantized_dataset_path)

# fit multiple models w/o dataset quantization
quantized_train_dataset = Pool(data='quantized://' + quantized_dataset_path)

model_depth_four = CatBoostRegressor(depth=4)
model_depth_four.fit(quantized_train_dataset)

model_depth_eight = CatBoostRegressor(depth=8)
model_depth_eight.fit(quantized_train_dataset)

```


## Using pandas.Categorical type instead of object {#pandas-instead-of-objects}

Use the pandas.Categorical type instead of the object type to speed up the preprocessing for datasets with categorical features up to 200 times.

## Using numpy arrays instead pandas dataframes {#numpy-instead-of-pandas}
Use numpy arrays instead of pandas dataframes to speed up the preprocessing for large datasets.
