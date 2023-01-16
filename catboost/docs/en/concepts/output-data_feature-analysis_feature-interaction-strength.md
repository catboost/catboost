# Feature interaction strength

The following types of [feature interaction strength](fstr.md) files are created depending on the task and the execution parameters:
- [{{ title__Interaction }}](#per-feature-interaction-strength)
- [{{ title__InternalInteraction }}](#internal-interaction-strength)

## {{ title__Interaction }} {#per-feature-interaction-strength}

#### {{ output--contains }}

{% include [reusage-formats-feature-interaction-strength](../_includes/work_src/reusage-formats/feature-interaction-strength.md) %}

#### {{ output--format }}

- The rows are sorted in descending order of the feature interaction strength value.
- Each row contains information related to one pair of features.

    Format:
    ```
    <feature interaction strength><\t><feature name 1><\t><feature name 2>
    ```

    - `feature interaction strength` is the value of the feature interaction strength.

    - `feature name` is the zero-based index of the feature.

    An alphanumeric identifier is used instead if specified in the corresponding `{{ cd-file__col-type__Num }}` or `{{ cd-file__col-type__Categ }}` column of the [input data](../concepts/input-data_column-descfile.md).

    For example, let's assume that the [columns description](../concepts/input-data_column-descfile.md) file has the following structure:
    ```
    0<\t>Label value<\t>
    1<\t>Num
    2<\t>Num<\t>ratio
    3<\t>Categ
    4<\t>Auxiliary
    5<\t>Num
    ```

    The input [dataset description](../concepts/input-data_values-file.md) file contains the following line:
    ```
    120<\t>80<\t>0.8<\t>rock<\t>some useless information<\t>12
    ```

    The table below shows the compliance between the given feature values and the feature indices.

    {% include [reusage-formats-internal-feature-importance-desc](../_includes/concepts/feature-interaction-strength_table.md) %}


#### {{ output--example }}

```
0.5<\t>0<\t>1
30<\t>1<\t>2
```

## {{ title__InternalInteraction }} {#internal-interaction-strength}

#### {{ output--contains }}

{% include [reusage-formats-internal-feature-interaction-strength](../_includes/work_src/reusage-formats/internal-feature-interaction-strength.md) %}


#### {{ output--format }}

- The rows are sorted in descending order of the feature interaction strength value.
- Each row contains information related to one pair of features and/or their combinations.

    Format:
    ```
    <feature interaction strength><\t><feature name 1><\t><feature name 2>
    ```

    - `feature interaction strength` is the value of the internal feature interaction strength.
    - `feature name` is the zero-based index of the feature.

    An alphanumeric identifier is used instead if specified in the corresponding `{{ cd-file__col-type__Num }}` or `{{ cd-file__col-type__Categ }}` column of the [input data](../concepts/input-data_column-descfile.md).

    For example, let's assume that the [columns description](../concepts/input-data_column-descfile.md) file has the following structure:
    ```
    0<\t>Label value<\t>
    1<\t>Num
    2<\t>Num<\t>ratio
    3<\t>Categ
    4<\t>Auxiliary
    5<\t>Num
    ```

    The input [dataset description](../concepts/input-data_values-file.md) file contains the following line:
    ```
    120<\t>80<\t>0.8<\t>rock<\t>some useless information<\t>12
    ```

    The table below shows the compliance between the given feature values and the feature indices.

    {% include [reusage-formats-internal-feature-importance-desc](../_includes/concepts/feature-interaction-strength_table.md) %}

#### {{ output--example }}

```
0.4004860988<\t>15<\t>13
0.1134764975<\t>{4} prior_num=0 prior_denom=1 targetborder=0 type=Borders<\t>15
```
