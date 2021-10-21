
{% note info %}

Only the values of calculated metrics are output. The following metrics are not calculated by default for the training dataset and therefore these metrics are not output:

- {{ error-function__PFound }}
- {{ error-function__YetiRank }}
- {{ error-function__ndcg }}
- {{ error-function__YetiRankPairwise }}
- {{ error-function--AUC }}
- {{ error-function--NormalizedGini }}
- {{ error-function__FilteredDCG }}
- {{ error-function__dcg }}

Use the `hints=skip_train~false` parameter to enable the calculation. See the [Enable, disable and configure metrics calculation](../../../concepts/loss-functions.md#enable-disable-configure-metrics) section for more details.

{% endnote %}

