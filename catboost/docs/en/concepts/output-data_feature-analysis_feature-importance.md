# Feature importance

The following types of [feature importance](fstr.md) files are created depending on the task and the execution parameters:
- [Regular feature importance](#per-feature-importance)
- [{{ title__internal-feature-importance }}](#internal-feature-importance)

## Regular feature importance {#per-feature-importance}

#### {{ output--contains }}

{% include [reusage-formats-regular-feature-importance-desc](../_includes/work_src/reusage-formats/regular-feature-importance-desc.md) %}

{% include [reusage-formats-regular-feature-importance-type](../_includes/work_src/reusage-formats/regular-feature-importance-type.md) %}

#### {{ output--format }}

- The rows are sorted in descending order of the feature importance value.

- Each row contains information related to one feature.

  Format:
  ```
  <feature strength><\t><feature name>
  ```

  - `feature strength` is the value of the of the regular feature importance.
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
8.4 <\t> 2
5.5 <\t> 0
2.6 <\t> 3
1.5 <\t> ratio
```

## {{ title__internal-feature-importance }} {#internal-feature-importance}

#### {{ output--contains }}

{% include [reusage-formats-internal-feature-importance-desc](../_includes/work_src/reusage-formats/internal-feature-importance-desc.md) %}


#### {{ output--format }}

- The rows are sorted in descending order of the feature importance value.

- Each row contains information related to one feature or a combination of features.

    Format:
    ```
    <feature strength><\t><{feature name 1,.., feature name n} pr<value> tb<value> type<value>
    ```

    - `feature strength` is the value of the internal feature importance.

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

    - `pr` is the prior value.
    - `tb` is the label value border value.
    - `type` is the feature border type.


#### {{ output--example }}

```
8.4<\t>0
5.2<\t>{2, ratio} pr2 tb0 type0
2.6<\t>{2} pr2 tb0 type0
```
