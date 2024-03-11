# Group weights

{% if audience == "internal" %}

{% include [internal__group-weights-type-of-group-weights-to-use__desc](../yandex_specific/_includes/type-of-group-weight-to-use__desc.md) %}

## Local TSV files {#local-file}

{% endif %}

#### {{ input_data__title__contains }}
The weights of groups within the dataset.
#### {{ input_data__title__specification }}

- List the weight of each group on a new line.
- Use a tab as the delimiter to separate the columns on a line.

#### {{ input_data__title__row-format }}

```
<group id><\t><weight>
```

- `group id` is the identifier of a group. Should match one of the values specified in the [Dataset description in delimiter-separated values format](../concepts/input-data_values-file.md).
- `weight` is the weight of the corresponding group.

#### {{ input_data__title__peculiarities }}

- {% include [reusage-input-data-group_weights__precedence-over-datasetdesc](../_includes/work_src/reusage-input-data/group_weights__precedence-over-datasetdesc.md) %}

- {% include [reusage-input-data-group_weights__input-dataset-requirement](../_includes/work_src/reusage-input-data/group_weights__input-dataset-requirement.md) %}


#### {{ input_data__title__example }}

```
10<\t>0.01
1007<\t>0.02
1009<\t>0.03
1017<\t>0.04
1018<\t>0.05
```

{% if audience == "internal" %}

{% include [reusage-file-with-group-weights](../yandex_specific/_includes/yt_tables_group_weights.md) %}

{% endif %}


