
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


### --input-path

#### Description

The name of the input file with the [dataset description](../../../concepts/input-data_values-file.md).

**{{ cli__params-table__title__default }}**

{{ calc--input-path }}


### --column-description, --cd

#### Description

The path to the input file that contains the [columns description](../../../concepts/input-data_column-descfile.md).

**{{ cli__params-table__title__default }}**

If omitted, it is assumed that the first column in the file with the dataset description defines the label value, and the other columns are the values of numerical features.

### --input-pairs

#### Description

The path to the input file that contains the pairs description for the dataset.

This information is used for the calculation of Pairwise metrics.

**{{ cli__params-table__title__default }}**

Omitted

Pairwise metrics require pairs of data. If this data is not provided explicitly by specifying this parameter, pairs are generated automatically in each group using object label values

### -o, --output-path

#### Description

The path to the output file with [calculated metrics](../../../concepts/output-data_loss-function.md).

**{{ cli__params-table__title__default }}**

{{ calc--output-path }}


### -T, --thread-count

#### Description

The number of threads to calculate metrics.

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


### --ntree-start

#### Description

To reduce the number of trees to use when the model is applied or the metrics are calculated, set the range of the tree indices to `[--ntree-start; --ntree-end)` and the step of the trees to use to `--eval-period`.

This parameter defines the index of the first tree to be used when applying the model or calculating the metrics (the inclusive left border of the range). Indices are zero-based.

**{{ cli__params-table__title__default }}**

0


### --ntree-end

#### Description

To reduce the number of trees to use when the model is applied or the metrics are calculated, set the range of the tree indices to `[--ntree-start; --ntree-end)` and the step of the trees to use to `--eval-period`.

This parameter defines the index of the first tree not to be used when applying the model or calculating the metrics (the exclusive right border of the range). Indices are zero-based.

**{{ cli__params-table__title__default }}**

0 (the index of the last tree to use equals to the number of trees in the model minus one)


### --eval-period

#### Description

To reduce the number of trees to use when the model is applied or the metrics are calculated, set the range of the tree indices to `[--ntree-start; --ntree-end)` and the step of the trees to use to `--eval-period`.

This parameter defines the step to iterate over the range `[--ntree-start; --ntree-end)`. For example, let's assume that the following parameter values are set:

- `--ntree-start` is set 0
- `--ntree-end` is set to N (the total tree count)
- `--eval-period` is set to 2
In this case, the results are returned for the following tree ranges: `[0, 2), [0, 4), ... , [0, N)`.

**{{ cli__params-table__title__default }}**

0 (the index of the last tree to use equals to the number of trees in the model minus one)


### --metrics

#### Description

A comma-separated list of metrics to be calculated.

[Possible values](../../../references/custom-metric__supported-metrics.md)

For example, if the AUC and Logloss metrics should be calculated, use the following construction:

```
--metrics AUC,Logloss
```

**{{ cli__params-table__title__default }}**

Required parameter


### --result-dir

#### Description

The directory for storing the files generated during metric calculation.

**{{ cli__params-table__title__default }}**

None (current directory)


### --tmp-dir

#### Description

The directory for storing temporary files that are generated if non-additive metrics are calculated.

By default, the directory is generated inside the current one at the start of calculation, and it is removed when the calculation is complete. Otherwise the specified value is used.

**{{ cli__params-table__title__default }}**

(the directory is generated)

### --verbose

#### Description

Verbose output to stdout.

**{{ cli__params-table__title__default }}**

False
