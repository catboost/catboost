# fit

{% include [sections-with-methods-desc-fit--purpose-desc](../_includes/work_src/reusage/fit--purpose-desc.md) %}


{% include [reusage-python-how-to-train-on-gpu](../_includes/work_src/reusage-python/how-to-train-on-gpu.md) %}


## {{ dl--invoke-format }} {#call-format}

```python
fit(X,
    y=None,
    cat_features=None,
    text_features=None,
    embedding_features=None,
    sample_weight=None,
    baseline=None,
    use_best_model=None,
    eval_set=None,
    verbose=None,
    logging_level=None
    plot=False,
    plot_file=None,
    column_description=None,
    verbose_eval=None,
    metric_period=None,
    silent=None,
    early_stopping_rounds=None,
    save_snapshot=None,
    snapshot_file=None,
    snapshot_interval=None,
    init_model=None,
    log_cout=sys.stdout,
    log_cerr=sys.stderr)
```

## {{ dl--parameters }} {#parameters}

{% include [precedence-python--classifier--precedence-p](../_includes/work_src/reusage/python--classifier--precedence-p.md) %}

### X

#### Description

The description is different for each group of possible types.

**Possible types**

{% cut "{{ python-type--pool }}" %}

The input training dataset.

{% note info %}

If a nontrivial value of the `cat_features` parameter is specified in the constructor of this class, {{ product }} checks the equivalence of categorical features indices specification from the constructor parameters and in this Pool class.

{% endnote %}

{% endcut %}

{% cut "{{ python-type--list }}, {{ python-type--numpy-ndarray }}, {{ python-type--pandasDataFrame }}, {{ python-type--pandasSeries }}" %}

The input training dataset in the form of a two-dimensional feature matrix.

{% endcut %}

{% cut "{{ python_type__pandas-SparseDataFrame }}, {{ python_type__scipy-sparse-spmatrix }} (all subclasses except dia_matrix)" %}

{% include [libsvm-libsvm__desc](../_includes/work_src/reusage-formats/libsvm__desc.md) %}

{% endcut %}

**Default value**

{{ python--required }}

**Supported processing units**

 {{ cpu-gpu }}

### y

#### Description

{% include [methods-param-desc-label--short-desc1](../_includes/work_src/reusage/label--short-desc1.md) %}


{% include [methods-param-desc-label--short-desc2](../_includes/work_src/reusage/label--short-desc2.md) %}


{% note info %}

Do not use this parameter if the input training dataset (specified in the `X` parameter) type is {{ python-type--pool }}.

{% endnote %}

**Possible types**

- {{ python-type--list }}
- {{ python-type--numpy-ndarray }}
- {{ python-type--pandasDataFrame }}
- {{ python-type--pandasSeries }}

**Default value**

None

**Supported processing units**

{{ cpu-gpu }}


### cat_features

#### Description

A one-dimensional array of categorical columns indices.

Use it only if the `X` parameter is a two-dimensional feature matrix (has one of the following types: {{ python-type--list }}, {{ python-type__np_ndarray }}, {{ python-type--pandasDataFrame }}, {{ python-type--pandasSeries }}).

{% note info %}

The `cat_features` parameter can also be specified in the constructor of the class. If it is, {{ product }} checks the equivalence of the `cat_features` parameter specified in this method and in the constructor of the class.

{% endnote %}


**Possible types**

- {{ python-type--list }}
- {{ python-type--numpy-ndarray }}

**Default value**

{% include [default-value-none](../_includes/concepts/default-value-none.md) %}

**Supported processing units**

{{ cpu-gpu }}


### text_features

#### Description

A one-dimensional array of text columns indices (specified as integers) or names (specified as strings).

{% include [reusage-python__cat_features__description__non-catfeatures-text](../_includes/work_src/reusage/python__cat_features__description__non-catfeatures-text.md) %}

**Possible types**

- {{ python-type--list }}
- {{ python-type--numpy-ndarray }}

**Default value**

{% include [default-value-none](../_includes/concepts/default-value-none.md) %}

**Supported processing units**

{{ cpu-gpu }}


### embedding_features

#### Description

A one-dimensional array of embedding columns indices (specified as integers) or names (specified as strings).

{% include [reusage-python__cat_features__description__non-catfeatures-text](../_includes/work_src/reusage/python__cat_features__description__non-catfeatures-text.md) %}

**Possible types**

- {{ python-type--list }}
- {{ python-type--numpy-ndarray }}

**Default value**

{% include [default-value-none](../_includes/concepts/default-value-none.md) %}

**Supported processing units**

{{ cpu-gpu }}


### sample_weight

#### Description

{% include [methods-param-desc-python__weight__short-desc-intro](../_includes/work_src/reusage/python__weight__short-desc-intro.md) %}


{% include [methods-param-desc-python__weight__short-desc__outro](../_includes/work_src/reusage/python__weight__short-desc__outro.md) %}

**Possible types**

- {{ python-type--list }}
- {{ python-type--numpy-ndarray }}
- {{ python-type--pandasDataFrame }}
- {{ python-type--pandasSeries }}

**Default value**

None

**Supported processing units**

{{ cpu-gpu }}


### baseline

#### Description

{% include [methods-param-desc-baseline--short-desc1](../_includes/work_src/reusage/baseline--short-desc1.md) %}


{% note info %}

Do not use this parameter if the input training dataset (specified in the `X` parameter) type is {{ python-type--pool }}.

{% endnote %}

**Possible types**

- {{ python-type--list }}
- {{ python-type--numpy-ndarray }}

**Default value**

None

**Supported processing units**

{{ cpu-gpu }}



### use_best_model

#### Description

If this parameter is set, the number of trees that are saved in the resulting model is defined as follows:
1. Build the number of trees defined by the training parameters.
1. Use the validation dataset to identify the iteration with the optimal value of the metric specified in  `--eval-metric` (`--eval-metric`).

No trees are saved after this iteration.

This option requires a validation dataset to be provided.

**Possible types**

{{ python-type--bool }}

**Default value**

True if a validation set is input (the `eval_set` parameter is defined) and at least one of the label values of objects in this set differs from the others. False otherwise.

**Supported processing units**

{{ cpu-gpu }}


### eval_set

#### Description

The validation dataset or datasets used for the following processes:
- [overfitting detector](../concepts/overfitting-detector.md)
- best iteration selection
- monitoring metrics' changes


**Possible types**

- {{ python-type--pool }}
- {{ python-type--list }} of {{ python-type--pool }}
- {{ python-type--tuple }} (X, y)
- {{ python-type--list }} of {{ python-type--tuple }}s (X, y)
- {{ python-type--string }} (path to the dataset file)
- {{ python-type--list }} of {{ python-type--string }}s (paths to dataset files)

**Default value**

None

**Supported processing units**

{{ cpu-gpu }}

{% note info %}

Only a single validation dataset can be input if the training is performed on GPU

{% endnote %}

### verbose

_Alias:_`verbose_eval`


#### Description

{% include [sections-with-methods-desc-python__feature-importances__verbose__short-description__list-intro](../_includes/work_src/reusage/python__feature-importances__verbose__short-description__list-intro.md) %}


- {{ python-type--bool }} — Defines the logging level:
    - <q>True</q>  corresponds to the Verbose logging level
    - <q>False</q> corresponds to the Silent logging level

- {{ python-type--int }} — Use the Verbose logging level and set the logging period to the value of this parameter.

{% note alert %}

Do not use this parameter with the `logging_level` parameter.

{% endnote %}

**Possible types**

- {{ python-type--bool }}
- {{ python-type--int }}

**Default value**

{{ train_verbose_fr-of-iterations-to-output__default }}

**Supported processing units**

{{ cpu-gpu }}



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

**Supported processing units**

{{ cpu-gpu }}


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

**Supported processing units**

{{ calcer_type__cpu }}

### plot_file

#### Description

Save a plot with the training progress information (metric values, custom loss values) to the file specified by this parameter.

**Possible types**

File-like object or {{ python-type--string }}

**Default value**

None

**Supported processing units**

{% include [reusage-python-cpu](../_includes/work_src/reusage-python/cpu-and-gpu.md) %}

### column_description

#### Description

{% include [reusage-cd-short-desct](../_includes/work_src/reusage/cd-short-desct.md) %}

The given file is used to build pools from the train and/or validation datasets, which are input from files.

**Default value**

None

**Supported processing units**

{{ cpu-gpu }}

### silent

#### Description

Defines the logging level:
- <q>True</q> — corresponds to the Silent logging level
- <q>False</q> — corresponds to the Verbose logging level

**Possible types**

{{ python-type--bool }}

**Default value**

False

**Supported processing units**

{{ cpu-gpu }}


### early_stopping_rounds

#### Description

Sets the overfitting detector type to {{ fit--od-type-iter }} and stops the training after the specified number of iterations since the iteration with the optimal metric value.

**Possible types**

{{ python-type--int }}

**Default value**

False

**Supported processing units**

{{ cpu-gpu }}

### save_snapshot

#### Description

Enable snapshotting for [restoring the training progress after an interruption](../features/snapshots.md). If enabled, the default period for making snapshots is {{ fit__snapshot-interval__default }} seconds. Use the `snapshot_interval` parameter to change this period.

{% include [reusage-common-phrases-snapshot-not-working-for-cv](../_includes/work_src/reusage-common-phrases/snapshot-not-working-for-cv.md) %}

**Possible types**

{{ python-type--bool }}

**Default value**

{{ fit--save_snapshot }}

**Supported processing units**

{{ cpu-gpu }}

### snapshot_file

#### Description

The name of the file to save the training progress information in. This file is used for [recovering training after an interruption](../features/snapshots.md).

{% include [reusage-snapshot-filename-desc](../_includes/work_src/reusage/snapshot-filename-desc.md) %}


{% include [reusage-common-phrases-snapshot-not-working-for-cv](../_includes/work_src/reusage-common-phrases/snapshot-not-working-for-cv.md) %}

**Possible types**

{{ python-type--string }}

**Default value**

{% cut "experiment..." %}

{{ fit--snapshot-file-python }}

{% endcut %}

**Supported processing units**

{{ cpu-gpu }}


### snapshot_interval

#### Description

The interval between saving snapshots in seconds.
The first snapshot is taken after the specified number of seconds since the start of training. Every subsequent snapshot is taken after the specified number of seconds since the previous one. The last snapshot is taken at the end of the training.

{% include [reusage-common-phrases-snapshot-not-working-for-cv](../_includes/work_src/reusage-common-phrases/snapshot-not-working-for-cv.md) %}

**Possible types**

{{ python-type--int }}

**Default value**

{{ fit__snapshot-interval__default }}

**Supported processing units**

{{ cpu-gpu }}

### init_model

#### Description

The description is different for each group of possible types.


**Possible types**

The model to continue learning from.

{% note info %}

The initial model must have the same problem type as the one being solved in the current training (binary classification, multiclassification or regression/ranking).

{% endnote %}

None (incremental learning is not used){{ calcer_type__cpu }}

{% cut "{{ [catboost.CatBoost](../concepts/python-reference_catboost.md), catboost.CatBoostClassifier](../concepts/python-reference_catboostclassifier.md) }}" %}

The initial model object.

{% endcut %}

{% cut "{{ python-type--string }}" %}

The path to the input file that contains the initial model.

{% endcut %}

**Default value**

None (incremental learning is not used)

**Supported processing units**

{{ calcer_type__cpu }}

{% include [python__log-params](../_includes/work_src/reusage-python/python__log-params.md) %}

## {{ dl__usage-examples }} {#usage-examples}

#### Train a model using a matrix

```python
from catboost import CatBoostClassifier

cat_features = [0,1,2]

train_data = [["a", "b", 1, 4, 5, 6],
              ["a", "b", 4, 5, 6, 7],
              ["c", "d", 30, 40, 50, 60]]

train_labels = [1,1,0]

model = CatBoostClassifier(iterations=20)

model.fit(train_data, train_labels, cat_features)
predictions = model.predict(train_data)
```

#### Load the dataset using [Pool](../concepts/python-reference_pool.md), train it with [CatBoostClassifier](../concepts/python-reference_catboostclassifier.md) and make a prediction

```python
from catboost import CatBoostClassifier, Pool

train_data = Pool(data=[[1, 4, 5, 6],
                        [4, 5, 6, 7],
                        [30, 40, 50, 60]],
                  label=[1, 1, -1],
                  weight=[0.1, 0.2, 0.3])

model = CatBoostClassifier(iterations=10)

model.fit(train_data)
preds_class = model.predict(train_data)

```
