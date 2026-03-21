# Quantization

Before learning, the possible values for each numerical feature (for both specified directly in the input data and obtained as results of internal processing, see the documentation about converting [categorical](../concepts/algorithm-main-stages_cat-to-numberic.md), [text](../concepts/algorithm-main-stages_text-to-numeric.md) and [embedding](../concepts/algorithm-main-stages_embedding-to-numeric.md) features for details) are divided into disjoint ranges (_buckets_) delimited by the threshold values (_splits_). The size of the quantization (the number of splits) is determined by the starting parameters (separately for numerical features specified directly in the input data and obtained as results of internal processing, see the documentation about converting [categorical](../concepts/algorithm-main-stages_cat-to-numberic.md), [text](../concepts/algorithm-main-stages_text-to-numeric.md) and [embedding](../concepts/algorithm-main-stages_embedding-to-numeric.md) features for details).

Quantization is also used to split the label values when working with categorical features. А random subset of the dataset is used for this purpose on large datasets.

The table below shows the quantization modes provided in {{ product }}.

{% include [reusage-binarization-modes](../_includes/work_src/reusage/binarization-modes.md) %}


