### {{ error-function--PRAUC }} {#PRAUC}

PRAUC is the area under the curve $Precision(q)$ vs $Recall(q)$ for $q \in [0,1]$ where $Precision(q)$ and $Recall(q)$ are defined as follows.
$$Precision(q) = \frac{TP(q)}{TP(q) + FP(q)}, Recall(q) = \frac{TP(q)}{TP(q) + FN(q)}$$

Above $TP(q)$, $FP(q)$, $FN(q)$ are weights of the true positive, false positive, and false negative samples, respectively.

To calculate PRAUC for a binary classification model, specify type `{{ loss-functions__params__auc__type__Classic }}`.
In this case, $TP(q)=\sum w_i [p_i > q] c_i$, etc.

To calculate PRAUC for a multi-classification model, specify type `{{ loss-functions__params__auc__type__onevsall }}`.
In this case, positive samples are samples having class 0, all other samples are negative, and $TP(q)=\sum w_i [p_{i0} > q] [c_i = 0]$, etc.

{% cut "{{ loss-functions__params__type }}" %}

The type of PRAUC. Defines the metric calculation principles.

Type `{{ loss-functions__params__auc__type__Classic }}` is compatible with binary classification models.
Type `{{ loss-functions__params__auc__type__onevsall }}` is compatible with multi-classification models.

_Default_: `{{ loss-functions__params__auc__type__Classic }}`.
_Possible values_: `{{ loss-functions__params__auc__type__Classic }}`, `{{ loss-functions__params__auc__type__onevsall }}`.
_Examples_: `PRAUC:type=Classic`, `PRAUC:type=OneVsAll`.

{% endcut %}

{% cut "{{loss-functions__params__use_weights}}" %}

{% include [use-weights__desc__without__note](../work_src/reusage-loss-functions/use-weights__desc__without__note.md) %}

_Default_: `False`.
_Examples_: `PRAUC:type=Classic;use_weights=False`.

{% endcut %}
