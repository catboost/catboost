# Columns description

{% if audience == "internal" %}

{% include [internal__cd-type-of-cd-to-use__desc](../yandex_specific/_includes/type-of-cd-to-use__desc.md) %}

## Local TSV files {#local-file}

{% endif %}

## {{ input_data__title__contains }}

{% include [catboost-cd__full-desc-contains__full](../_includes/work_src/reusage-input-data/contains__full.md) %}

{% note info %}

The columns description file is optional. {{ column-desc__default }}

{% endnote %}

### Supported column types

* [{{ cd-file__col-type__label }}](#label)
* [{{ cd-file__col-type__Num }}](#num)
* [{{ cd-file__col-type__Categ }}](#categ)
* [{{ cd-file__col-type__Text }}](#text)
* [{{ cd-file__col-type__NumVector }}](#numvector)
* [{{ cd-file__col-type__Auxiliary }}](#auxiliary)
* [{{ cd-file__col-type__SampleId }}](#sampleId)
* [{{ cd-file__col-type__Weight }}](#weight)
* [{{ cd-file__col-type__GroupWeight }}](#groupWeight)
* [{{ cd-file__col-type__Baseline }}](#baseline)
* [{{ cd-file__col-type__GroupId }}](#groupId)
* [{{ cd-file__col-type__SubgroupId }}](#subgroupId)
* [{{ cd-file__col-type__Timestamp }}](#timestamp)
* [{{ cd-file__col-type__Position }}](#position)
{% if audience == "internal" %}

{% include [internal__cd-type-of-cd-to-use__desc](../yandex_specific/_includes/column_types.md) %}
{% endif %}

{% include [catboost-cd__full-desc-table-descriptions__full](../_includes/work_src/reusage-input-data/table-descriptions__full.md) %}
{% if audience == "internal" %}

{% include [catboost-cd__yandex_specific__full-desc-table-descriptions__full](../yandex_specific/_includes/reusage-input-data/table-descriptions__full.md) %}

{% endif %}

## {{ input_data__title__specification }}

- List each column on a new line.
- Additional properties are set on the corresponding line.
- Use a tab as the delimiter to separate data for a single column.
- Columns that contain numerical features don't require descriptions. Any columns that aren't specified in the file are assumed to be `Num`.

## {{ input_data__title__row-format }}

```
<column index (numbering starts from zero)><\t><data type><\t><feature id/name (optional, applicable for feature type columns only ({{ cd-file__col-type__Num }}, {{ cd-file__col-type__Categ }}, {{ cd-file__col-type__Text }} and {{ cd-file__col-type__NumVector }})>
```

## {{ input_data__title__peculiarities }}

- The feature indices and the column indices usually differ.

    The table below shows the difference between these indices on the columns description example given above.

    Column index | Column data | Feature index
    ----- | ----- | -----
    0 | `{{ cd-file__col-type__label }}` | —
    1 | `{{ cd-file__col-type__Num }}` | 0
    2 | `{{ cd-file__col-type__Num }}` | 1
    3 | `{{ cd-file__col-type__Categ }}<\t>wind direction` | 2
    4 | `{{ cd-file__col-type__Auxiliary }}` | —
    5 | `{{ cd-file__col-type__Num }}` | 3

- Multiregression labels are specified in several separate columns.


{% cut "Example" %}

    ```
    0<\t>Label
    1<\t>Label
    ```

{% endcut %}


## {{ input_data__title__example }}

{% include [reusage-weather-example-feature-list](../_includes/work_src/reusage/weather-example-feature-list.md) %}


{% include [reusage-hypothesis-value](../_includes/work_src/reusage/hypothesis-value.md) %}


{% include [reusage-arbitrary-data-column](../_includes/work_src/reusage/arbitrary-data-column.md) %}


{% include [reusage-feature-wind-direction](../_includes/work_src/reusage/feature-wind-direction.md) %}


{% include [reusage-file-with-column-descs](../_includes/work_src/reusage/file-with-column-descs.md) %}


The following variant is equivalent to the previous but is redundant:

```
0<\t>{{ cd-file__col-type__label }}<\t>
1<\t>{{ cd-file__col-type__Num }}
2<\t>{{ cd-file__col-type__Num }}
3<\t>{{ cd-file__col-type__Categ }}<\t>wind direction
4<\t>{{ cd-file__col-type__Auxiliary }}
5<\t>{{ cd-file__col-type__Num }}
```
{% if audience == "internal" %}

{% include [reusage-file-with-column-descs](../yandex_specific/_includes/yt_tables.md) %}

{% endif %}
