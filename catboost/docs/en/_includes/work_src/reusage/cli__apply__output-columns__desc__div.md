
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

#### Example

```
--output-columns Probability,#3,#4:Feature4,Label,SampleId
```

In this example, features with indices 3 and 4 are output. The header contains the index (<q>#3</q>) for the feature indexed 3 and the string <q>Feature4</q> for the feature indexed 4.

A fragment of the output

```
Probability	#3	Feature4	Label	SampleId
0.4984999565	1	50.7799987793	0	0
0.8543220144	1	48.6333312988	2	1
0.7358535042	1	52.5699996948	1	2
0.8788711681	1	48.1699981689	2	3
```


{% note info %}

At least one of the specified columns must contain prediction values. For example, the following value raises an error:
```
--output-columns SampleId
```

{% endnote %}
