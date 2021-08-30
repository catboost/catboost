# Input data

Input data for training, testing, and applying the model can be passed as files containing:

- [Columns description](input-data_column-descfile.md)
- [Dataset description in delimiter-separated values format](input-data_values-file.md)
- [Dataset description in extended libsvm format](input-data_libsvm.md)
- [Pairs description](input-data_pairs-description.md)
- [Custom quantization borders and missing value modes](input-data_custom-borders.md)
- [Group weights](input-data_group-weights.md)
- [Baseline ](input-data_baseline.md)

Other data transmission methods are available for the [Python](python-installation.md) and [R](r-installation.md) packages.

{% note info %}

The data format is the same for training, testing, and applying the model. Information about the label value (if available) is ignored during model application.

{% endnote %}

