### -m, --model-file,--model-path

#### Description

The name of the input file with the description of the model obtained as the result of training.

**{{ cli__params-table__title__default }}**

{{ calc--model-path }}

### --model-format

#### Description

The format of the input model.

Possible values:
- {{ fit__model-format_CatboostBinary }}.{{ fit__model-format_applecoreml }}(only datasets without categorical features are currently supported).{{ fit__model-format_json }} (multiclassification models are not currently supported). Refer to the [CatBoost JSON model tutorial]({{ catboost-repo__json-tutorial }}) for format details.

**{{ cli__params-table__title__default }}**

{{ fit__model-format }}


### --input-path

#### Description

The name of the input file with the [dataset description](../../../concepts/input-data_values-file.md).

**{{ cli__params-table__title__default }}**

{{ calc--input-path }}


### --column-description, --cd

#### Description

The path to the input file {% if audience == "internal" %}or table{% endif %} that contains the [columns description](../../../concepts/input-data_column-descfile.md).

{% if audience == "internal" %}

{% include [internal__cd-internal-cd-desc](../../../yandex_specific/_includes/reusage-formats/internal-cd-desc.md) %}

{% endif %}


**{{ cli__params-table__title__default }}**

{{ column-desc__default }}


### --input-pairs

#### Description

The path to the input file that contains the pairs description for the dataset.

This information is used for the calculation of Pairwise metrics.

**{{ cli__params-table__title__default }}**

Omitted

Pairwise metrics require pairs of data. If this data is not provided explicitly by specifying this parameter, pairs are generated automatically in each group using object label values

### -o, --output-path

#### Description

Defines the output settings for the resulting values of the model.

Supported value formats and types:
- `stream://<stream>` — Output the results to one of the program's standard output streams.

    `stream` is the name of the output stream. Possible values: `stdout` or `stderr`.

    For example, set the following value to output the results of applying the model to `stdout`:
    ```
    -o stream://stdout
    ```

- `[<path>/]<filename>.tsv` — Write the results into the specified file.
    - `path` is the optional path to the directory, where the resulting file should be saved to. By default, the file is saved to the same directory, from which the application is launched.
    - `filename` is the name of the output file.

    For example, set the following value to output the results of applying the model to the `/home/model/output-results.tsv` file:
    ```
    -o /home/model/output-results.tsv
    ```
The output data [format](../../../concepts/output-data_model-value-output.md) depends on the machine learning task being solved.

**{{ cli__params-table__title__default }}**

{{ calc--output-path }}

### --output-columns

#### Description

A comma-separated list of columns names to output when forming the [results of applying the model](../../../concepts/output-data_model-value-output.md) (including the ones obtained for the validation dataset when training).

Prediction and feature values can be output for each object of the input dataset. Additionally, some [column types](../../../concepts/input-data_column-descfile.md) can be output if specified in the input data.

{% cut "Supported prediction types" %}

- {{ prediction-type--Probability }}
- {{ prediction-type--Class }}
- {{ prediction-type--RawFormulaVal }}
- {{ prediction-type--Exponent }}
- {{ prediction-type--LogProbability }}

{% endcut %}

{% cut "Supported column types" %}

- `{{ cd-file__col-type__label }}`
- `{{ cd-file__col-type__Baseline }}`
- `{{ cd-file__col-type__Weight }}`
- `{{ cd-file__col-type__SampleId }}` (`{{ cd-file__col-type__DocId }}`)
- `{{ cd-file__col-type__GroupId }}` (`{{ cd-file__col-type__QueryId }}`)
- `{{ cd-file__col-type__QueryId }}`
- `{{ cd-file__col-type__SubgroupId }}`
- `{{ cd-file__col-type__Timestamp }}`
- `{{ cd-file__col-type__GroupWeight }}`

{% endcut %}

The output columns can be set in any order. Format:

```
<prediction type 1>,[<prediction type 2> .. <prediction type N>][columns to output],[#<feature index 1>[:<name to output (user-defined)>] .. #<feature index N>[:<column name to output>]]
```

**Example**

```
--output-columns Probability,#3,#4:Feature4,Label,SampleId
```

In this example, features with indices 3 and 4 are output. The header contains the index (<q>#3</q>) for the feature indexed 3 and the string <q>Feature4</q> for the feature indexed 4.

{% cut "A fragment of the output" %}

```
Probability	#3	Feature4	Label	SampleId
0.4984999565	1	50.7799987793	0	0
0.8543220144	1	48.6333312988	2	1
0.7358535042	1	52.5699996948	1	2
0.8788711681	1	48.1699981689	2	3
```

{% endcut %}

{% note info %}

At least one of the specified columns must contain prediction values. For example, the following value raises an error:
```
--output-columns SampleId
```

{% endnote %}


**{{ cli__params-table__title__default }}**

All columns that are supposed to be output according to the chosen parameters are output

### -T, --thread-count

#### Description

{% include [reusage-thread-count-short-desc](thread-count-short-desc.md) %}

{% include [reusage-thread_count__cpu_cores__optimizes-the-speed-of-execution](thread_count__cpu_cores__optimizes-the-speed-of-execution.md) %}


**{{ cli__params-table__title__default }}**

{{ fit--thread_count }}


### --tree-count-limit

#### Description

The number of trees from the model to use when applying. If specified, the first <value> trees are used.

**{{ cli__params-table__title__default }}**

{{ calc--tree-count-limit }}


### --eval-period

#### Description

To reduce the number of trees to use when the model is applied or the metrics are calculated, setthe step of the trees to use to `eval-period`.


{% include [eval-start-end-cli__eval__period__desc](../reusage-common-phrases/cli__eval__period__desc.md) %}


**{{ cli__params-table__title__default }}**

0 (the staged prediction mode is turned off)


### --prediction-type

#### Description

A comma-separated list of prediction types.

Supported prediction types:
- {{ prediction-type--Probability }}
- {{ prediction-type--Class }}
- {{ prediction-type--RawFormulaVal }}
- {{ prediction-type--Exponent }}
- {{ prediction-type--LogProbability }}


**{{ cli__params-table__title__default }}**

{{ prediction-type--RawFormulaVal }}
