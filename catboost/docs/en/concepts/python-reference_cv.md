# cv

```python
cv(pool=None,
   params=None,
   dtrain=None,
   iterations=None,
   num_boost_round=None,
   fold_count=3,
   nfold=None,
   inverted=False,
   partition_random_seed=0,
   seed=None,
   shuffle=True,
   logging_level=None,
   stratified=None,
   as_pandas=True,
   metric_period=None,
   verbose=None,
   verbose_eval=None,
   plot=False,
   early_stopping_rounds=None,
   folds=None,
   type='Classical',
   return_models=False)
```

## {{ dl--purpose }} {#purpose}

{% include [cv-cv__purpose](../_includes/work_src/reusage-python/cv__purpose.md) %}


The dataset is split into _N_ folds. _N–1_ folds are used for training, and one fold is used for model performance estimation. _N_ models are updated on each iteration _K_. Each model is evaluated on its' own validation dataset on each iteration. This produces _N_ metric values on each iteration _K_.

The `cv` function calculates the average of these _N_ values and the standard deviation. Thus, these two values are returned on each iteration.

If the dataset contains group identifiers, all objects from one group are added to the same fold when partitioning is performed.

## {{ dl--parameters }} {#parameters}

### pool

_Alias_: `dtrain`

#### Description

The input dataset to cross-validate.

**Possible types**

[Pool](python-reference_pool.md)

**Default value**

{{ python--required }}

### params

#### Description

{% include [python__cv-python__cv__params__description__div](../_includes/work_src/reusage/python__cv__params__description__div.md) %}


{% note info %}

- The following parameters are not supported in cross-validation mode: `save_snapshot`,
    `--snapshot-file`
    , `snapshot_interval`.
- The behavior of the [overfitting detector](overfitting-detector.md) is slightly different from the training mode. Only one metric value is calculated at each iteration in the training mode, while [`fold_count`](#python__cv__fold_count__parmname) metric values are calculated in the cross-validation mode. Therefore, all `fold_count` values are averaged and the best iteration is chosen based on the average metric value at each iteration.

{% endnote %}

**Possible types**

{{ python-type--dict }}

**Default value**

{{ python--required }}

### iterations

_Aliases_: `num_boost_round`, `n_estimators`, `num_trees`

#### Description

The maximum number of trees that can be built when solving machine learning problems.

When using other parameters that limit the number of iterations, the final number of trees may be less than the number specified in this parameter.

**Possible types**

{{ python-type--int }}

**Default value**

{{ fit--iterations }}

### fold_count

_Alias_: `nfold`

#### Description

The number of folds to split the dataset into.

**Possible types**

{{ python-type--int }}

**Default value**

`3`

### inverted

#### Description

Train on the test fold and evaluate the model on the training folds.

**Possible types**

{{ python-type--bool }}

**Default value**

False

### partition_random_seed

_Alias_: `seed`

#### Description

{% include [reusage-cv-rand__desc_intro](../_includes/work_src/reusage/cv-rand__desc_intro.md) %}


{% include [reusage-cv-rand__permutation-is-performed](../_includes/work_src/reusage/cv-rand__permutation-is-performed.md) %}


{% include [reusage-cv-rand__unique-data-splits](../_includes/work_src/reusage/cv-rand__unique-data-splits.md) %}

**Possible types**

{{ python-type--int }}

**Default value**

0

### shuffle

#### Description

Shuffle the dataset objects before splitting into folds.

**Possible types**

{{ python-type--bool }}

**Default value**

True

### logging_level

#### Description

The logging level to output to stdout.

Possible values:
- Silent — Do not output any logging information to stdout.

- Verbose — Output the following data to stdout:

    - optimized metric
    - elapsed time of training
    - remaining time of training

- Info — Output additional information and the number of trees.

- Debug — Output debugging information.

**Possible types**

{{ python-type--string }}

**Default value**

None (corresponds to the {{ fit--verbose }} logging level)

### stratified

#### Description

Perform stratified sampling.

It is turned on (True) by default if one of the following loss functions is selected: {{ error-function--Logit }}, {{ error-function--MultiClass }}, {{ error-function--MultiClassOneVsAll }}.

It is turned off (False) for all other loss functions by default.

**Possible types**

{{ python-type--bool }}

**Default value**

None


### as_pandas

#### Description

Sets the type of return value to {{ python-type--pandasDataFrame }}.

The type of return value is {{ python-type--dict }} if this parameter is set to False or the `pandas`{{ python-package }} is not installed.

**Possible types**

{{ python-type--bool }}

**Default value**

True

### metric_period

#### Description

{% include [reusage-cli__metric-period__desc__start](../_includes/work_src/reusage/cli__metric-period__desc__start.md) %}


{% include [reusage-cli__metric-period__desc__end](../_includes/work_src/reusage/cli__metric-period__desc__end.md) %}

**Possible types**

{{ python-type--int }}

**Default value**

{{ fit__metric-period }}

### verbose

_Alias_: `verbose_eval`

#### Description


{% include [sections-with-methods-desc-python__feature-importances__verbose__short-description__list-intro](../_includes/work_src/reusage/python__feature-importances__verbose__short-description__list-intro.md) %}


- {{ python-type--bool }} — Defines the logging level:
    - <q>True</q>  corresponds to the Verbose logging level
    - <q>False</q> corresponds to the Silent logging level

- {{ python-type--int }} — Use the Verbose logging level and set the logging period to the value of this parameter.

**Possible types**

- {{ python-type--bool }}
- {{ python-type--int }}

**Default value**

False

### plot

#### Description

Plot the following information during training:
- the metric values;
- the custom loss values;
- the loss function change during feature selection;
- the time has passed since training started;
- the remaining time until the end of training.
This [option can be used](../features/visualization_jupyter-notebook.md) if training is performed in Jupyter notebook.

**Possible types**

{{ python-type--bool }}

**Default value**

{{ fit--plot }}

### early_stopping_rounds

#### Description

Sets the overfitting detector type to {{ fit--od-type-iter }} and stops the training after the specified number of iterations since the iteration with the optimal metric value.

**Possible types**

{{ python-type--int }}

**Default value**

False

### folds

#### Description

Custom splitting indices.

The format of the input data depends on the type of the parameter:
- {{ python-type__generator }} or {{ python-type__iterator }} — Train and test indices for each fold.
- {% include [reusage-python-object-scikitlearn](../_includes/work_src/reusage-python/object-scikitlearn.md) %}

This parameter has the highest priority among other data split parameters.

**Possible types**

- {{ python-type__generator }}
- {{ python-type__iterator }}
- scikit-learn splitter {{ python-type__object }}

**Default value**

None

### type

#### Description

The method to split the dataset into folds.

Possible values:
- {{ cv__type__Classical }} — The dataset is split into `fold_count` folds, `fold_count` trainings are performed. Each test set consists of a single fold, and the corresponding train set consists of the remaining k–1 folds.

- {{ cv__type__Inverted }} — The dataset is split into `fold_count` folds, `fold_count` trainings are performed. Each test set consists of the first k–1 folds, and the corresponding train set consists of the remaining fold.

- {{ cv__type__TimeSeries }} — The dataset is split into (`fold_count` + 1) consecutive parts without shuffling the data, `fold_count` trainings are performed. The k-th train set consists of the first k folds, and the corresponding test set consists of the (k+1)-th fold.

**Possible types**

{{ python-type--string }}

**Default value**

{{ cv__type__default }}

### return_models

#### Description

If `return_models` is `True`, returns a list of models fitted for each CV fold. By default, False.

**Possible types**

{{ python-type--bool }}

**Default value**

False


## {{ dl--output-format }} {#output-format}

Depends on `return_models`, `as_pandas`, and the availability of the `pandas`{{ python-package }}:
- If `return_models` is `False`, `cv` returns `cv_results` which is a dict or a pandas frame (see a table below).
- If `return_models` is `True`, `cv` returns a tuple (`cv_results`, `fitted_models`) containing, in addition to regular `cv_results`, a list of models fitted for each fold.

`as_pandas` value | `pandas`{{ python-package }} availability | Type of return value
----- | ----- | -----
True | Installed | {{ python-type--pandasDataFrame }}
True | Not installed | {{ python-type--dict }}
False | Unimportant | {{ python-type--dict }}

The first key (if the output type is {{ python-type--dict }}) or column name (if the output type is {{ python-type--pandasDataFrame }}) contains the iteration of the calculated metrics values on the corresponding line. Each following key or column name is formed from the evaluation dataset type (train or test), metric name, and computed characteristic (std, mean, etc.). Each value is a list of corresponding computed values.

For example, if only the {{ error-function--RMSE }} metric is specified in the parameters, then the return value is:

```
    iterations  test-Logloss-mean  test-Logloss-std  train-Logloss-mean  train-Logloss-std
0            0           0.693219          0.000101            0.684767           0.011851
1            1           0.682687          0.014995            0.674235           0.003043
2            2           0.672758          0.029630            0.655983           0.005906
3            3           0.668589          0.023734            0.648127           0.005204
```

Each key or column value contains the same number of calculated values as the number of training iterations (or less, if the [overfitting detection](overfitting-detector.md) is turned on and the threshold is reached earlier).

## {{ dl--example }} {#usage-examples}

{% include [cv-usage-example-cv__usage-example](../_includes/work_src/reusage-code-examples/cv__usage-example.md) %}


{% include [cv-usage-example-cv_with_roc_curve__example](../_includes/work_src/reusage-code-examples/cv_with_roc_curve__example.md) %}
