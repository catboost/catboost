# Overfitting detection settings

## early_stopping_rounds {#early_stopping_rounds}

#### Description

Sets the overfitting detector type to {{ fit--od-type-iter }} and stops the training after the specified number of iterations since the iteration with the optimal metric value.

**Type**

{{ python-type--int }}

**Default value**

False

**Supported processing units**

{{ cpu-gpu }}

## od_type {#od_type}

Command-line: `--od-type`

#### Description

The type of the overfitting detector to use.

Possible values:
- {{ fit--od-type-inctodec }}
- {{ fit--od-type-iter }}

**Type**
 {{ python-type--string }}

**Default value**

{{ fit--od-type-inctodec }}

**Supported processing units**

{{ cpu-gpu }}

## od_pval {#od_pval}

Command-line: `--od-pval`

#### Description

The threshold for the {{ fit--od-type-inctodec }} [overfitting detector](../../concepts/overfitting-detector.md) type. The training is stopped when the specified value is reached. Requires that a validation dataset was input.

For best results, it is recommended to set a value in the range $[10^{–10}; 10^{-2}]$.

The larger the value, the earlier overfitting is detected.

{% note alert %}

Do not use this parameter with the {{ fit--od-type-iter }} overfitting detector type.

{% endnote %}

**Type**

{{ python-type--float }}

**Default value**

{{ fit--auto_stop_pval }}

**Supported processing units**

{{ cpu-gpu }}

## od_wait {#od_wait}

Command-line: `--od-wait`

#### Description

The number of iterations to continue the training after the iteration with the optimal metric value.
The purpose of this parameter differs depending on the selected overfitting detector type:
- {{ fit--od-type-inctodec }} — Ignore the overfitting detector when the threshold is reached and continue learning for the specified number of iterations after the iteration with the optimal metric value.
- {{ fit--od-type-iter }} — Consider the model overfitted and stop training after the specified number of iterations since the iteration with the optimal metric value.

**Type**

{{ python-type--int }}

**Default value**

{{ fit--od-wait }}

**Supported processing units**

{{ cpu-gpu }}

