# Scale and bias

## {{ dl--purpose }} {#purpose}

Set and/or print the model scale and bias.

## {{ dl__cli__execution-format }} {#execution-format}

```bash
catboost normalize-model [optional parameters]
```

## {{ common-text__title__reference__parameters }} {#options}

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
- json (multiclassification models are not currently supported). Refer to the [CatBoost JSON model tutorial](https://github.com/catboost/tutorials/blob/master/model_analysis/model_export_as_json_tutorial.ipynb) for format details.
- onnx — ONNX-ML format (only datasets without categorical features are currently supported). Refer to [https://onnx.ai](https://onnx.ai) for details. See the [ONNX](apply-onnx-ml.md) section for details on applying the resulting model.
- pmml — [PMML version 4.3](http://dmg.org/pmml/pmml-v4-3.html) format. Categorical features must be interpreted as one-hot encoded during the training if present in the training dataset. This can be accomplished by setting the --one-hot-max-size/one_hot_max_size parameter to a value that is greater than the maximum number of unique categorical feature values among all categorical features in the dataset. See the [PMML](apply-pmml.html) section for details on applying the resulting model.

**{{ cli__params-table__title__default }}**

 {{ calc--model-format }}

### --column-description, --cd

#### Description

The path to the input file that contains the [columns description](input-data_column-descfile.md).

**{{ cli__params-table__title__default }}**

 If omitted, it is assumed that the first column in the file with the dataset description defines the label value, and the other columns are the values of numerical features.

### --delimiter

#### Description

The delimiter character used to separate the data in the dataset description input file.

Only single char delimiters are supported. If the specified value contains more than one character, only the first one is used.

{% note info %}

Used only if the dataset is given in the [Delimiter-separated values format](input-data_values-file.md).

{% endnote %}

**{{ cli__params-table__title__default }}**

 The input data is assumed to be tab-separated

### --has-header

#### Description

Read the column names from the first line of the dataset description file if this parameter is set.

{% note info %}

Used only if the dataset is given in the [Delimiter-separated values format](input-data_values-file.md).

{% endnote %}


**{{ cli__params-table__title__default }}**

 False (the first line is supposed to have the same data as the rest of them)

### --set-scale

#### Description

The model scale.

**{{ cli__params-table__title__default }}**

 1

### --set-bias

#### Description

The model bias.

The model prediction results are calculated as follows:

The value of this parameters affects the prediction by changing the default value of the bias.

**{{ cli__params-table__title__default }}**

 Depends on the value of the `--boost-from-average` for the Command-line version parameter:

- True — The best constant value for the specified loss function
- False — 0

### --print-scale-and-bias

#### Description

Return the scale and bias of the model.

These values affect the results of applying the model, since the model prediction results are calculated as follows:

Scale and bias are not output


**{{ cli__params-table__title__default }}**

 Set the scale and bias to 0.8:


### --logging-level

#### Description

The logging level to output to stdout.

Possible values:
- Silent — Do not output any logging information to stdout.
- Verbose — Output the following data to stdout:
    - optimized metric
    - elapsed time of training
    - remaining time of training
- Info — Output additional information and the number of trees.
- Debug — Output debugging information.


**{{ cli__params-table__title__default }}**

 Info

### -T, --thread-count

#### Description

The number of threads to use.

**{{ cli__params-table__title__default }}**

 4

### --input-path

#### Description

The name of the input file with the [dataset description](input-data_values-file.md#input-data_values-file).

**{{ cli__params-table__title__default }}**

 {{ calc--input-path }}

### --output-model

#### Description

The path to the output model.

**{{ cli__params-table__title__default }}**

 {{ calc--output-model }}

### --output-model-format

#### Description

The format of the output model.

Possible values:

- CatboostBinary.
- AppleCoreML (only datasets without categorical features are currently supported).
- json (multiclassification models are not currently supported). Refer to the [CatBoost JSON model tutorial](https://github.com/catboost/tutorials/blob/master/model_analysis/model_export_as_json_tutorial.ipynb) for format details.
- onnx — ONNX-ML format (only datasets without categorical features are currently supported). Refer to https://onnx.ai for details. See the [ONNX](apply-onnx-ml.md) section for details on applying the resulting model.
- pmml — [PMML version 4.3](http://dmg.org/pmml/pmml-v4-3.html) format. Categorical features must be interpreted as one-hot encoded during the training if present in the training dataset. This can be accomplished by setting the --one-hot-max-size/one_hot_max_size parameter to a value that is greater than the maximum number of unique categorical feature values among all categorical features in the dataset. See the [PMML](apply-pmml.html) section for details on applying the resulting model.


**{{ cli__params-table__title__default }}**

 {{ calc--output-model-format }}


## {{ dl__usage-examples }} {#usage-examples}


```python
catboost normalize-model --set-scale 0.8 --set-bias 0.8 --print-scale-and-bias
```

The output of this example:

```bash
Input model scale 1 bias 1.405940652
Output model scale 0.8 bias 0.8
```
