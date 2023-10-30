# Feature interaction

## {{ title__Interaction }} {#feature-interaction-strength}

{% include [reusage-formats-feature-interaction-strength](../_includes/work_src/reusage-formats/feature-interaction-strength.md) %}


All splits of features $f1$ and $f2$ in all trees of the resulting ensemble are observed when calculating the interaction between these features.

If splits of both features are present in the tree, then we are looking on how much leaf value changes when these splits have the same value and they have opposite values.

See the [{{ title__Interaction }}](output-data_feature-analysis_feature-interaction-strength.md#per-feature-interaction-strength) file format.


{% cut "{{ title__fstr__calculation-principle }}" %}

$interaction(f_{1}, f_{2}) = \sum_{trees} \left |\sum_{leafs: split(f_1)=split(f_2)} LeafValue { } - \sum_{leafs: split(f_1)\ne split(f_2)}LeafValue \right |$
The sum inside the modulus always contains an even number of terms. The first half of terms contains leaf values when splits by $f1$ have the same value as splits by $f2$, the second half contains leaf values when two splits have different values, and the second half is in the sum with a different sign.

The larger the difference between sums of leaf values, the bigger the interaction. This process reflects the following idea: let's fix one feature and see if the changes to the other one will result in large formula changes.

{% endcut %}


## {{ title__InternalInteraction }} {#internal-feature-interaction-strength}

{% include [reusage-formats-internal-feature-interaction-strength](../_includes/work_src/reusage-formats/internal-feature-interaction-strength.md) %}


See the [{{ title__InternalInteraction }}](output-data_feature-analysis_feature-interaction-strength.md#internal-interaction-strength) file format.


{% cut "{{ title__fstr__calculation-principle }}" %}

$interaction(f_{1}, f_{2}) = \sum_{trees} \left |\sum_{leafs: split(f_1)=split(f_2)} LeafValue { } - \sum_{leafs: split(f_1) \neq split(f_2)}LeafValue \right |$

{% endcut %}


#### Related information
[Detailed information regarding usage specifics for different Catboost implementations.](../features/feature-importances-calculation.md#feature-importances-calculation)
