# {{ title__missing-values-processing }}

The missing values processing mode depends on the feature type and the selected package.


## Numerical features {#numerical-features}

{{ product }} interprets the value of a numerical feature as a missing value if it is equal to one of the following values, which are package-dependant:

{% list tabs %}

- {{ python-package }}

    - `None`
    - [Floating point NaN value](https://en.wikipedia.org/wiki/NaN)
    - One of the following strings when loading the values from files or as Python strings:

        {% include [reusage-missing-values-python__mv-processing-mode__list-full](../_includes/work_src/reusage-missing-values/python__mv-processing-mode__list-full.md) %}

- {{ r-package }}

    - [Floating point NaN value](https://en.wikipedia.org/wiki/NaN)
    - One of the following strings when loading the values from files:

        {% include [reusage-missing-values-python__mv-processing-mode__list-full](../_includes/work_src/reusage-missing-values/python__mv-processing-mode__list-full.md) %}

- Command-line version

  One of the following strings when loading the values from files when reading from an input file:

  {% include [reusage-missing-values-python__mv-processing-mode__list-full](../_includes/work_src/reusage-missing-values/python__mv-processing-mode__list-full.md) %}


{% endlist %}

The following modes for processing missing values are supported:


{% include [reusage-missing-values-mv-processing-methods](../_includes/work_src/reusage-missing-values/mv-processing-methods.md) %}


The default processing mode is {{ fit--nan_mode }}. The methods for changing the default mode are package-dependant:

{% list tabs %}

- {{ python-package }}

    - Globally for all features in the `nan_mode` [training parameter](../references/training-parameters/index.md).
    - Individually for each feature in the [Custom quantization borders and missing value modes](../concepts/input-data_custom-borders.md) input file. Such values override the global default setting.

-  {{ r-package }}

    - Globally for all features in the `nan_mode` [training parameter](../references/training-parameters/index.md).
    - Individually for each feature in the [Custom quantization borders and missing value modes](../concepts/input-data_custom-borders.md) input file. Such values override the global default setting.

- Command-line version

    - Globally for all features in the `--nan-mode` [training parameter](../references/training-parameters/index.md).
    - Individually for each feature in the [Custom quantization borders and missing value modes](../concepts/input-data_custom-borders.md) input file. Such values override the global default setting.

{% endlist %}

## Categorical features {#categorical-features}

{% include [reusage-missing-values-missing-values__categorical-features-values](../_includes/work_src/reusage-missing-values/missing-values__categorical-features-values.md) %}
