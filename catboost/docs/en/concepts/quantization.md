# Quantization

Before learning, the possible values of objects are divided into disjoint ranges (_buckets_) delimited by the threshold values (_splits_). The size of the quantization (the number of splits) is determined by the starting parameters (separately for numerical features and numbers obtained as a result of [converting categorical features into numerical features](../concepts/algorithm-main-stages_cat-to-numberic.md)).

Quantization is also used to split the label values when working with categorical features. А random subset of the dataset is used for this purpose on large datasets.

The table below shows the quantization modes provided in {{ product }}.

{% include [reusage-binarization-modes](../_includes/work_src/reusage/binarization-modes.md) %}


