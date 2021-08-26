
{% note info %}

The object weights are not used to optimize this metric. The group weights are used instead.

This objective is used to optimize PairLogit. Automatically generated object pairs are used for this purpose. These pairs are generated independently for each object group. Use the [Group weights](../../../concepts/input-data_group-weights.md) file or the {{ cd-file__col-type__GroupWeight }} column of the [Columns description](../../../concepts/input-data_column-descfile.md) file to change the group importance. In this case, the weight of each generated pair is multiplied by the value of the corresponding group weight.

{% endnote %}

