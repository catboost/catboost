# Dataset description in delimiter-separated values format

{% if audience == "internal" %}

{{ product }} supports several dataset formats.

## {{ input_dataset_format__native_catboost }} {#native-catboost-format}

{% endif %}

#### {{ input_data__title__contains }}

{% include [dataset-dataset-desc__native-catboost__contains__full](../_includes/work_src/reusage-formats/dataset-desc__native-catboost__contains__full.md) %}

#### {{ input_data__title__specification }}

- List each object on a new line.
- {% include [methods-param-desc-group-id__desc__group-by-group-id__obligatory](../_includes/work_src/reusage/group-id__desc__group-by-group-id__obligatory.md) %}

- If the group weight is specified, it must be the same for all objects in one group.
- Use any single char delimiters to separate data about a single object. The required delimiter can be specified in the training parameters. Tabs are used as the default separator.
- Use the feature types that are specified in the [columns description](input-data_column-descfile.md).
- List features in the same order for all the objects.
- Feature numbering starts from zero.

#### {{ input_data__title__example }}

The dataset consists of 6 columns.

The first column (indexed 0) contains label values.

{% include [reusage-hypothesis-value](../_includes/work_src/reusage/hypothesis-value.md) %}

Columns indexed 1, 2, 3 and 5 contain features.

The column indexed 4 contains arbitrary data.

{% include [reusage-file-with-column-descs](../_includes/work_src/reusage/file-with-column-descs.md) %}

The feature indexed 3 is categorical, so the value in the second column of the description file is set to . The name of this feature is set to <q>wind direction</q> in the third column of the description file.

Other features are numerical and are omitted from the columns description file.

The dataset description looks like this:
```
1<\t>–10<\t>5<\t>north<\t>Memphis TN<\t>753
0<\t>30<\t>1<\t>south<\t>Los Angeles CA<\t>760
0<\t>40<\t>0.1<\t>south<\t>Las Vegas NV<\t>705
```
{% if audience == "internal" %}

{% include [internal-input-data_values-file](../yandex_specific/_includes/internal-input-data_values-file.md) %}

{% endif %}
