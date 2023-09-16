# train

```python
train(pool=None,
      params=None,
      dtrain=None,
      logging_level=None,
      verbose=None,
      iterations=None,
      num_boost_round=None,
      evals=None,
      eval_set=None,
      plot=None,
      verbose_eval=None,
      metric_period=None,
      early_stopping_rounds=None,
      save_snapshot=None,
      snapshot_file=None,
      snapshot_interval=None,
      init_model=None)
```

## {{ dl--purpose }} {#purpose}

Train a model.

{% note info %}

{% include [installation-nvidia-driver-reqs](../_includes/work_src/reusage-code-examples/nvidia-driver-reqs.md) %}

{% endnote %}


## {{ dl--parameters }} {#parameters}

### pool

_Alias_: `dtrain`

#### Description

{% include [sections-with-methods-desc-python__x__desc__paragraph-no-note](../_includes/work_src/reusage/python__x__desc__paragraph-no-note.md) %}

**Possible types**

{{ python-type--pool }}

**Default value**

{{ python--required }}

**Supported processing units**

{{ cpu-gpu }}

### params

#### Description

The list of [parameters](../references/training-parameters/index.md) to start training with.

**Possible types**

{{ python-type--dict }}

**Default value**

{{ python--required }}

**Supported processing units**

{{ cpu-gpu }}

### logging_level

#### Description

{% include [reusage-cli__logging-level__intro](../_includes/work_src/reusage/cli__logging-level__intro.md) %}


{% include [reusage-cli__logging-level__outro](../_includes/work_src/reusage/cli__logging-level__outro.md) %}


{% note alert %}

Should not be used with the `verbose` parameter.

{% endnote %}

**Possible types**

{{ python-type--string }}

**Default value**

None (corresponds to the {{ fit--verbose }} logging level)

**Supported processing units**

{{ cpu-gpu }}

### verbose

_Alias_: `verbose_eval`

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


### iterations

_Alias_: `num_boost_round`

#### Description

The maximum number of trees that can be built when solving machine learning problems.

When using other parameters that limit the number of iterations, the final number of trees may be less than the number specified in this parameter.

**Possible types**

{{ python-type--int }}

**Default value**

{{ fit--iterations }}

**Supported processing units**

{{ cpu-gpu }}

### eval_set

_Alias_: `evals`

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

### metric_period

#### Description

The frequency of iterations to calculate the values of [objectives and metrics](../concepts/loss-functions.md). The value should be a positive integer.

The usage of this parameter speeds up the training.

{% note info %}

It is recommended to increase the value of this parameter to maintain training speed if a GPU processing unit type is used.

{% endnote %}

**Possible types**

{{ python-type--int }}

**Default value**

{{ fit__metric-period }}

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

The model to continue learning from.

{% note info %}

The initial model must have the same problem type as the one being solved in the current training (binary classification, multiclassification or regression/ranking).

{% endnote %}

**Possible types**

{% cut "catboost.CatBoost, catboost.CatBoostClassifier, catboost.CatBoostRegressor" %}

The initial model object:
  - [catboost.CatBoost](../concepts/python-reference_catboost.md)
  - [catboost.CatBoostClassifier](../concepts/python-reference_catboostclassifier.md)
  - [catboost.CatBoostRegressor](../concepts/python-reference_catboostregressor.md)

{% endcut %}

{% cut "{{ python-type--string }}" %}

The path to the input file that contains the initial model.

{% endcut %}

**Default value**

None (incremental learning is not used)

**Supported processing units**

{{ calcer_type__cpu }}
