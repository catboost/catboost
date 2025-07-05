# Input data

Input data for training, testing, and applying models can be passed as files containing:

- [Columns description](input-data_column-descfile.md)
- [Dataset description in delimiter-separated values format](input-data_values-file.md)
- [Dataset description in extended libsvm format](input-data_libsvm.md)
- [Pairs description](input-data_pairs-description.md)
- [Custom quantization borders and missing value modes](input-data_custom-borders.md)
- [Group weights](input-data_group-weights.md)
- [Baseline](input-data_baseline.md)

Non-CLI packages and libraries also support passing input data in data structures native to their languages.

{% note info %}

The data format is the same for training, testing, and applying models. Model application uses only features data.

{% endnote %}

