
### -m, --model-file, --model-path

#### Description

The name of the input file with the description of the model obtained as the result of training.

**{{ cli__params-table__title__default }}**

{{ calc--model-path }}

### --model-format

#### Description

The format of the input model.
Possible values:
- CatboostBinary.
- AppleCoreML (only datasets without categorical features are currently supported).
- json (multiclassification models are not currently supported). Refer to the CatBoost [JSON model tutorial](https://github.com/catboost/tutorials/blob/master/model_analysis/model_export_as_json_tutorial.ipynb) for format details.
**{{ cli__params-table__title__default }}**

{{ calc--model-format }}

### -f, --learn-set

#### Description

The path to the input file that contains the dataset description.

Format:

```
[scheme://]<path>
```

- `scheme` (optional) defines the type of the input dataset. Possible values:
    - `quantized://` — catboost. Pool [quantized](../../../concepts/python-reference_pool_quantized.md) pool.
    - `libsvm://` — dataset in the [extended libsvm format](../../../concepts/input-data_libsvm.md).
If omitted, a dataset in the [Native CatBoost Delimiter-separated values format](../../../concepts/input-data_values-file.md) is expected.
- `path` defines the path to the dataset description.

**{{ cli__params-table__title__default }}**

Required parameter (the path must be specified).

### -t, --test-set

#### Description

The path to the input file that contains the validation dataset description (the format must be the same as used in the training dataset).

**{{ cli__params-table__title__default }}**

Required parameter

### --column-description, --cd

#### Description

The path to the input file that contains the [columns description](../../../concepts/input-data_column-descfile.md#input-data_column-descfile).

**{{ cli__params-table__title__default }}**

If omitted, it is assumed that the first column in the file with the dataset description defines the label value, and the other columns are the values of numerical features.

### -o, --output-path

#### Description

The path to the output file with [calculated metrics](../../../concepts/output-data_loss-function.md).

**{{ cli__params-table__title__default }}**

{{ calc--output-path }}


### -T, --thread-count

#### Description

The number of threads to use during the training.

Optimizes the speed of execution. This parameter doesn't affect results.

**{{ cli__params-table__title__default }}**

The number of processor cores

### --delimiter

#### Description

The delimiter character used to separate the data in the dataset description input file.
Only single char delimiters are supported. If the specified value contains more than one character, only the first one is used.

{% note info %}

Used only if the dataset is given in the [Delimiter-separated values format](../../../concepts/input-data_values-file.md).

{% endnote %}

**{{ cli__params-table__title__default }}**

The input data is assumed to be tab-separated


### --has-header

#### Description

False (the first line is supposed to have the same data as the rest of them)

**{{ cli__params-table__title__default }}**

False (the first line is supposed to have the same data as the rest of them)

### --update-method

#### Description

The algorithm accuracy method.

Possible values:
- SinglePoint — The fastest and least accurate method.
- TopKLeaves — Specify the number of leaves. The higher the value, the more accurate and the slower the calculation.
- AllPoints — The slowest and most accurate method.
Supported parameters:
- `top` — Defines the number of leaves to use for the TopKLeaves update method. See the [Finding Influential Training Samples for Gradient Boosted Decision Trees](https://arxiv.org/abs/1802.06640) for more details.
For example, the following value sets the method to TopKLeaves and limits the number of leaves to 3:

```
TopKLeaves:top=3
```

**{{ cli__params-table__title__default }}**

SinglePoint
