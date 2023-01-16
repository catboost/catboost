# Model values

The results of applying the model on a dataset.

The output information and format depends on the machine learning problem being solved:

- [Regression or Ranking](#regression)
- [Classification](#classification)
- [Multiclassification](#multiclassification)
- [Multiregression](#multiregression)

Additionally, the required output columns can be defined in the `--output-columns` command-line [applying](cli-reference_calc-model.md) parameter. All columns that are supposed to be output according to the chosen parameters are output by default.

{% cut "Parameter description" %}

{% include [concept_pcd_bsy_xz-cli__apply__output-columns__desc__div](../_includes/work_src/reusage/cli__apply__output-columns__desc__div.md) %}

{% endcut %}


## Regression or Ranking {#regression}

#### {{ output--contains }}

A number resulting from applying the model.
- RMSEWithUncertainty (only for models trained with RMSEWithUncertainty loss) It is RawFormulaVal with exponent function applied to second dimension of approx due to obtain estimation of the variance (See the [Uncertainty section](https://catboost.ai/docs/references/uncertainty.html#uncertainty)).


#### {{ output__header-format }}

The first row in the output file contains a tab-separated description of data in the corresponding column.

Format:

```
[EvalSet:]{{ cd-file__col-type__SampleId }}<\t><Prediction type 1><\t>..<\t><Prediction type N>[<\t>Label]
```

- `EvalSet:` is output for the evaluation file only if several validation datasets are input.
- `Prediction type` is specified in the starting parameters and takes one or several of the following values:

    - {{ prediction-type--Probability }}
    - {{ prediction-type--Class }}
    - {{ prediction-type--RawFormulaVal }}
    - {{ prediction-type--Exponent }}
    - {{ prediction-type--LogProbability }}
    - RMSEWithUncertainty
    - VirtEnsembles
    - TotalUncertainty

- `Label`is only output for the validation dataset in training mode and the cross-validation dataset in cross-validation mode if it is specified in the input dataset.

#### {{ output--format }}

{% include [reusage-formats-regression__format__type-of-data](../_includes/work_src/reusage-formats/regression__format__type-of-data.md) %}


Format:

```
[<Validation dataset ID>:]<{{ cd-file__col-type__SampleId }}><\t><model value for prediction type 1><\t>..<\t><model value for prediction type N>[<\t><Label>]
```

- `Validation dataset ID` is the serial number of the input validation dataset. The value is output if several validation datasets are input for model evaluation purposes.
- `SampleId` is an alphanumeric ID of the object given in the [Dataset description in delimiter-separated values format](../concepts/input-data_values-file.md). If the identifiers are not set in the input data the objects are sequentially numbered, starting from zero.
- `model value for prediction type` is the float number resulting from applying the model for the corresponding prediction type.
- `label` is the label value for the object. This value is only output for the validation dataset in training mode and the cross-validation dataset in cross-validation mode if it is specified in the input dataset.

#### {{ output--example }}

The resulting file without alphanumeric IDs:

```
{{ cd-file__col-type__SampleId }}<\t>{{ prediction-type--Probability }}<\t>{{ prediction-type--Class }}
0<\t>0.8<\t>1
1<\t>0.3<\t>0
```

The resulting file for the cross-validation mode with alphanumeric IDs set:

```
{{ cd-file__col-type__SampleId }}<\t>Probability<\t>Label
LT<\t>75.1<\t>73.6
LV<\t>73.2<\t>72.15
PL<\t>78.22<\t>77.5
```

## Classification {#classification}

#### {{ output--contains }}

Depends on the selected output mode for approximated values of the formula:

- {{ prediction-type--RawFormulaVal }} —A number resulting from applying the model.
- {{ prediction-type--Probability }} — A number indicating the probability that the object belongs to the class (a sigmoid of the result of applying the model).
- {{ prediction-type--Class }} — The predicted class (output with the value <q>1</q> if the probability is higher than 0.5, otherwise <q>0</q>).

#### {{ output__header-format }}

The first row in the output file contains a tab-separated description of data in the corresponding column.

Format:
```
[EvalSet:]{{ cd-file__col-type__SampleId }}<\t><Prediction type 1><\t>..<\t><Prediction type N>[<\t>Label]
```

- `EvalSet:` is output for the evaluation file only if several validation datasets are input.
- `Prediction type` is specified in the starting parameters and takes one or several of the following values:

    - {{ prediction-type--Probability }}
    - {{ prediction-type--Class }}
    - {{ prediction-type--RawFormulaVal }}
    - {{ prediction-type--Exponent }}
    - {{ prediction-type--LogProbability }}
    - VirtEnsembles
    - TotalUncertainty

- `Label`is only output for the validation dataset in training mode and the cross-validation dataset in cross-validation mode if it is specified in the input dataset.

#### {{ output--format }}

Each row in the output file contains tab-separated information about a single object from the input dataset.

Format:
```
[<Validation dataset ID>:]<{{ cd-file__col-type__SampleId }}><\t><model value>[<\t><Label>]
```

- `Validation dataset ID` is the serial number of the input validation dataset. The value is output if several validation datasets are input for model evaluation purposes.
- `SampleId` is an alphanumeric ID of the object given in the [Dataset description in delimiter-separated values format](../concepts/input-data_values-file.md). If the identifiers are not set in the input data the objects are sequentially numbered, starting from zero.
- `model value` is the number resulting from applying the model for the corresponding prediction type.
- `label`is the label value for the object. This value is only output for the validation dataset in training mode and the cross-validation dataset in cross-validation mode if it is specified in the input dataset.

#### {{ output--example }}

The resulting file for the {{ prediction-type--RawFormulaVal }} cross-validation mode:

```
{{ cd-file__col-type__SampleId }}<\t>{{ prediction-type--RawFormulaVal }}<\t>Label
0<\t>0.1685379577<\t>1
1<\t>0.2379356203<\t>1
2<\t>-0.04871954376<\t>1
```

The resulting file for the {{ prediction-type--Probability }} cross-validation mode with alphanumeric IDs set for objects:
```
{{ cd-file__col-type__SampleId }}<\t>{{ prediction-type--Probability }}<\t>Label
{{ cd-file__col-type__SampleId }}1<\t>0.5592048528<\t>1
{{ cd-file__col-type__SampleId }}2<\t>0.5595881735<\t>1
{{ cd-file__col-type__SampleId }}3<\t>0.5592048528<\t>1
```

The resulting file for the {{ prediction-type--Class }} mode:
```
{{ cd-file__col-type__SampleId }}<\t>{{ prediction-type--Class }}
0<\t>0
1<\t>1
2<\t>1
3<\t>s0
```

## Multiclassification {#multiclassification}

#### {{ output--contains }}

Depends on the selected output mode for approximated values of the formula:

- {{ prediction-type--RawFormulaVal }} — A list of numbers resulting from applying the model. Values for the different classes are tab-separated.
- {{ prediction-type--Probability }} — A list of numbers indicating the probability that the object belongs to each of the classes. Values for the different classes are tab-separated.
- {{ prediction-type--Class }} —The number of the class that the object most likely belongs to.

#### {{ output__header-format }}

{% include [reusage-formats-header__intro](../_includes/work_src/reusage-formats/header__intro.md) %}


Format:

```
[EvalSet:]{{ cd-file__col-type__SampleId }}</t><PredictionType1>[:Class=<ClassID>]</t>..</t><PredictionTypeN>:Class=<ClassID>[<\t>Label]
```

- `EvalSet:` is output for the evaluation file only if several validation datasets are input.
- `Prediction type` is specified in the starting parameters and takes one or several of the following values:

    - {{ prediction-type--Probability }}
    - {{ prediction-type--Class }}
    - {{ prediction-type--RawFormulaVal }}
    - {{ prediction-type--Exponent }}
    - {{ prediction-type--LogProbability }}
    - VirtEnsembles
    - TotalUncertainty

- `ClassID` is the identifier of the class being described in the column. It is omitted for the {{ prediction-type--Class }} prediction type.
- `Label`is only output for the validation dataset in training mode and the cross-validation dataset in cross-validation mode if it is specified in the input dataset.

The number of <q>Prediction type–ClassID</q> pairs depends on the input parameters. It is always limited to one pair for the {{ prediction-type--Class }} prediction type.

#### {{ output--format }}

Each row in the output file contains tab-separated information about a single object from the input dataset.

Format:
```
[Validation dataset ID:]<{{ cd-file__col-type__SampleId }}><\t><Model value 1>..<Model value N>[<\t><Label>]
```

- `Validation dataset ID` is the serial number of the input validation dataset. The value is output if several validation datasets are input for model evaluation purposes.
- `SampleId` is an alphanumeric ID of the object given in the [Dataset description in delimiter-separated values format](../concepts/input-data_values-file.md). If the identifiers are not set in the input data the objects are sequentially numbered, starting from zero.
- `Model value` is a number or a list of numbers depending on the selected output mode for approximated values of the formula for the corresponding prediction type.
- `label`is the label value for the object. This value is only output for the validation dataset in training mode and the cross-validation dataset in cross-validation mode if it is specified in the input dataset.

#### {{ output--example }}

The resulting file for prediction in  {{ prediction-type--Class }} mode with alphanumeric IDs set for objects:

```
{{ cd-file__col-type__SampleId }}<\t>Class
{{ cd-file__col-type__SampleId }}1<\t>2
{{ cd-file__col-type__SampleId }}2<\t>1
{{ cd-file__col-type__SampleId }}3<\t>2
```

The resulting file for the {{ prediction-type--Probability }} cross-validation mode:

```
{{ cd-file__col-type__SampleId }}<\t>Probability:Class=0<\t>CProbability:Class=1<\t>Probability:Class=2<\t>Label
1<\t>0.3232259635</t>0.315456703</t>0.3613173334</t>2
2<\t>0.335771253</t>0.3247524917</t>0.3394762553</t>0
3<\t>0.3181931812</t>0.3242628483</t>0.3575439705</t>1
```

The resulting file for the {{ prediction-type--RawFormulaVal }} cross-validation mode:

```
{{ cd-file__col-type__SampleId }}<\t>RawFormulaVal:Class=0<\t>RawFormulaVal:Class=1<\t>RawFormulaVal:Class=2<\t>Label
1<\t>0.001232427024</t>-0.04141999431</t>0.04018756728</t>2
2<\t>-0.04822847313</t>-0.05520994445</t>0.1034384176</t>2
3<\t>-0.05717915565</t>-0.06548867981</t>0.1226678355</t>2
```

The resulting file for prediction in {{ prediction-type--RawFormulaVal }} and {{ prediction-type--Probability }} modes:
```
{{ cd-file__col-type__SampleId }}<\t>Probability:Class=0<\t>Probability:Class=1<\t>RawFormulaVal:Class=0<\t>RawFormulaVal:Class=1
1<\t>0.01593276944<\t>0.02337982256<\t>-1.494255509<\t>-1.110760101
2<\t>0.4060707366<\t>0.09565861257<\t>0.4137085351<\t>-1.032033103
3<\t>0.006235130003<\t>0.01759049831<\t>-2.03020042<\t>-0.9930409613
```

## Multiregression {#multiregression}

#### {{ output--contains }}

The numbers resulting from applying the model.

#### {{ output__header-format }}

{% include [reusage-formats-header__intro](../_includes/work_src/reusage-formats/header__intro.md) %}


Format:
```
[EvalSet:]{{ cd-file__col-type__SampleId }}<\t><Prediction type 1:Dim=0><\t>..<\t><Prediction type 1:Dim=M><\t>..<\t><Prediction type N:Dim=0><\t>..<\t><Prediction type N:Dim=M>[<\t>Label 1<\t>..<\t>Label M]
```

- `EvalSet:` is output for the evaluation file only if several validation datasets are input.
- {% include [reusage-formats-prediction-type__list-intro](../_includes/work_src/reusage-formats/prediction-type__list-intro.md) %}

    - {{ prediction-type--Probability }}
    - {{ prediction-type--RawFormulaVal }}

- `Dim` is the identifier of the label value from the labels vector.
- `Label`is only output for the validation dataset in training mode and the cross-validation dataset in cross-validation mode if it is specified in the input dataset.

#### {{ output--format }}

{% include [reusage-formats-regression__format__type-of-data](../_includes/work_src/reusage-formats/regression__format__type-of-data.md) %}


Format:

```
[<Validation dataset ID>:]<{{ cd-file__col-type__SampleId }}><\t><model value for prediction type 1 and dimension 0><\t>..<\t><model value for prediction type N and dimension 0><\t>..<\t><model value for prediction type N and dimension M>[<\t><Label 1><\t>..<\t><Label M>]
```

- `Validation dataset ID` is the serial number of the input validation dataset. The value is output if several validation datasets are input for model evaluation purposes.
- `SampleId` is an alphanumeric ID of the object given in the [Dataset description in delimiter-separated values format](../concepts/input-data_values-file.md). If the identifiers are not set in the input data the objects are sequentially numbered, starting from zero.
- `model value for prediction type and dimension` is the float number resulting from applying the model for the corresponding prediction type and label dimension ID.
- `label`is the label value for the object. This value is only output for the validation dataset in training mode and the cross-validation dataset in cross-validation mode if it is specified in the input dataset.

#### {{ output--example }}

The resulting file for one prediction type and two label values:

```
SampleId	RawFormulaVal:Dim=0	RawFormulaVal:Dim=1	Label:Dim=0	Label:Dim=1
0	-0.8314181536	0.0933995366	-1.3208890577723018	0.025460321322479378
1	0.2543254277	0.2431282743	0.6210866265458542	0.4546559804798615
2	0.1160567752	0.2253731494	-0.11214938100684924	0.1511700693162882
3	0.6113665201	-0.4325967375	1.7442698560770338	-0.3495077953076593
4	1.939011554	0.132468688	2.2505580326330246	0.2428380077970017
5	0.7703675176	-0.1621147587	0.9890286771162449	-0.2159423001955859
6	-1.404801493	0.1512662166	-2.5933998532067952	-0.41641507539874467
7	-1.029800989	0.3759548141	-3.210838860491183	0.9898948495861314
8	1.068953517	-0.2148923107	1.0980660342295991	-0.36311073650281733
9	-0.6949856863	0.1497956154	-0.8800937021376194	0.10275208758300074

```

## Regression or Ranking prediction
