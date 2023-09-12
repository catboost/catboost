# {{ title__ShapValues }}

A vector $v$ with contributions of each feature to the prediction for every input object and the expected value of the model prediction for the object (average prediction given no knowledge about the object).

- $v_{i}$ is the contribution of the i-th feature.
- $v_{feature\_count}$ is the expected value of the model prediction.

For a given object the sum $\sum\limits_{i=0}^{feature\_count} v[i]$ is equal to the prediction on this object.

This is an implementation of the [Consistent Individualized Feature Attribution for Tree Ensembles](https://arxiv.org/abs/1802.03888) approach.

See the [{{ title__ShapValues }}](output-data_feature-analysis_shap-values.md) file format.

{% include [reusage-formats-use-the-shap-package](../_includes/work_src/reusage-formats/use-the-shap-package.md) %}


{% cut "{{ title__fstr__calculation-principle }}" %}

The feature importance $ShapValues_{i}$ is calculated as follows for each feature $i$:

$ShapValues_{i} = \displaystyle\sum_{S \subseteq N \backslash \{i\}} \displaystyle\frac{|S|! \left(M - |S| - 1 \right)!}{M!} [f_{x}(S \cup \{i\}) - f_{x}(S)] { , where}$

- $M$ is the number of input features.
- $N$ is the set of all input features.
- $S$ is the set of non-zero feature indices (the features that are being observed and not unknown).
- $f_{x} (S) = E[f(x) | x_{s}]$ is the model's prediction for the input $x$, where $E[f(x) | x_{s}]$ is the expected value of the function conditioned on a subset S of the input features.

{% endcut %}


{% cut "{{ title__fstr__complexity }}" %}

The complexity of computation depends on several conditions:
- If the mean leaf count in the tree is less than the number of documents and trees are oblivious:
    $O(samples\_count \cdot trees\_count \cdot 2 ^ { depth} \cdot dimension \cdot depth ^ 2)$
- In all other cases:

    $O(trees\_count \cdot {leaves\_in\_tree} \cdot dimension \cdot average\_depth^2 ) +$

    $+O(trees\_count \cdot samples\_count \cdot (features\_in\_tree\_count + dimension)$

Used variables:

- `samples_count` is the number of documents in the dataset.
- `dimension` is the dimensionality for Multiclassification and Multiregression.
- `trees_count` is the number of trees.
- `depth` is the max depth of trees.
- `average_depth` is the average depth of the trees.
- `leaves_in_tree` is the number of leaves in the tree.
- `features_in_tree_count` is the number of features in the tree.

{% endcut %}


#### Related information
[Detailed information regarding usage specifics for different Catboost implementations.](../features/feature-importances-calculation.md#feature-importances-calculation)
