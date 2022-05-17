# Categorical features

{% include [parameter-tuning-one-hot-encoding__note](../_includes/work_src/reusage-common-phrases/one-hot-encoding__note.md) %}

{{ product }} supports numerical, categorical, text, and embeddings features.

Categorical features are used to build new numeric features based on categorical features and their combinations. See the [Transforming categorical features to numerical features](../concepts/algorithm-main-stages_cat-to-numberic.md) section for details.

By default, {{ product }} uses one-hot encoding for categorical features with a small amount of different values in most modes. It is not available if training is performed on CPU in

{% cut "Pairwise scoring" %}

The following loss functions use Pairwise scoring:

- {{ error-function__YetiRankPairwise }}
- {{ error-function__PairLogitPairwise }}
- {{ error-function__QueryCrossEntropy }}

Pairwise scoring is slightly different from regular training on pairs, since pairs are generated only internally during the training for the corresponding metrics. One-hot encoding is not available for these loss functions.

{% endcut %}

 mode. The default threshold for the number of unique values of the feature to be processed as one-hot encoded depends on various conditions, which are described in the table below.

Ctrs are not calculated for features that are used with one-hot encoding.

Some types of Ctrs require target data in the training dataset. Such Ctrs are not calculated if this data is not available. In this, case only one-hot encoded categorical features are used if training is performed on GPU (and the default value of unique values threshold for a categorical feature to be considered one-hot is increased according to this condition) and all categorical features are ignored if training is performed on CPU.

Use the following parameters to change the maximum number of unique values of categorical features for applying one-hot encoding:

{% if audience == "internal" %}

{% include [parameter-tuning-one-hot-encoding-features__internal](../yandex_specific/_includes/one-hot-encoding-features__internal.md) %}

{% endif %}

{% if audience == "external" %}

{% include [parameter-tuning-one-hot-encoding-features](../_includes/work_src/reusage-common-phrases/one-hot-encoding-features.md) %}

{% endif %}
