#### {{ cd-file__col-type__label }}

{% include [reusage-input-data-label__shortdesc](label__shortdesc.md) %}

The type of data depends on the machine learning task being solved:
- Regression, multiregression and ranking — Numeric values.
- Binary classification
    One of:

    - Integers or strings that represent the labels of the classes (only two unique values).
    - Numeric values.
        The interpretation of numeric values depends on the selected loss function:

        - {{ error-function--Logit }} — The value is considered a positive class if it is strictly greater than the value of the `target_border` training parameter. Otherwise, it is considered a negative class.
        - {{ error-function--CrossEntropy }} — The value is interpreted as the probability that the dataset object belongs to the positive class. Possible values are in the range `[0; 1]`.

- Multiclassification — Integers or strings that represents the labels of the classes.

#### {{ cd-file__col-type__Num }}

A numerical feature.

{% include [reusage-input-data-a-tab-delimited-feature-id-can-be-set](a-tab-delimited-feature-id-can-be-set.md) %}

#### {{ cd-file__col-type__Categ }}

A categorical feature.

{% include [reusage-input-data-a-tab-delimited-feature-id-can-be-set](a-tab-delimited-feature-id-can-be-set.md) %}

#### {{ cd-file__col-type__Text }}

A text feature.

#### {{ cd-file__col-type__NumVector }}

An array of numbers that represent an embedding feature. Numbers in the string are separated by a single character separator. Its default value is `;`.

#### {{ cd-file__col-type__Auxiliary }}

Any data.

A tab-delimited Auxiliary column ID can be added for this type of column. The specified value can be used in the `--output-columns` command-line [applying](../../../concepts/cli-reference_calc-model.md) parameter.

The value of this column is ignored (the behavior is the same as when this column is omitted in the file with the [dataset description](../../../concepts/input-data_values-file.md)).


#### {{ cd-file__col-type__SampleId }}

_Alias:_`{{ cd-file__col-type__DocId }}`

{% include [reusage-input-data-docid__shortdesc](docid__shortdesc.md) %}


#### {{ cd-file__col-type__Weight }}

{% include [reusage-input-data-weight__shortdesc](weight__shortdesc.md) %}

{% include [reusage-input-data-weight__desc](weight__desc.md) %}

{% note info %}

Do not use this column type if the `{{ cd-file__col-type__GroupWeight }}` column is defined in the dataset description.

{% endnote %}

#### {{ cd-file__col-type__GroupWeight }}

{% include [reusage-input-data-weight__short_desc](weight__short_desc.md) %}

Used as an additional coefficient in the [objective functions and metrics](../../../concepts/loss-functions.md). By default, it is set to 1 for all objects in the group.

{% note info %}

- The weight must be the same for all objects in one group.
- Do not use this column type if the `{{ cd-file__col-type__Weight }}` column is defined in the dataset description.

{% endnote %}

#### {{ cd-file__col-type__Baseline }}

{% include [reusage-input-data-baseline__shortdesc](baseline__shortdesc.md) %}

Used for calculating the final values of trees.

The required number of these columns depends on the machine learning mode:
- For classification and regression – one column.
- For multiclassification – the same as the number of classes.


#### {{ cd-file__col-type__GroupId }}

_Alias:_`{{ cd-file__col-type__QueryId }}`

{% include [loss-functions-object-id__full](../reusage-common-phrases/object-id__full.md) %}

{% include [methods-param-desc-group-id__desc__group-by-group-id__obligatory__note](../reusage/group-id__desc__group-by-group-id__obligatory__note.md) %}


#### {{ cd-file__col-type__SubgroupId }}

The identifier of the object's subgroup. Used to divide objects within a group. An arbitrary string, possibly representing an integer.


#### {{ cd-file__col-type__Timestamp }}

Should be a non-negative integer.

{% include [reusage-input-data-timestamp__shortdesc](timestamp__shortdesc.md) %}


#### {{ cd-file__col-type__Position }}

Should be a non-negative integer.

The ranking position of the object. The value is used to calculate the {{ error-function__StochasticFilter }} metric.
