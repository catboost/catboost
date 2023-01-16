# Transforming categorical features to numerical features

{% include [types-of-supported-features-supported-feature-types](../_includes/work_src/reusage-common-phrases/supported-feature-types.md) %}


Before each split is selected in the tree (see [Choosing the tree structure](algorithm-main-stages_choose-tree-structure.md)), categorical features are transformed to numerical. This is done using various statistics on combinations of categorical features and combinations of categorical and numerical features.

The method of transforming categorical features to numerical generally includes the following stages:
1. Permutating the set of input objects in a random order.

1. Converting the label value from a floating point to an integer.

    The method depends on the machine learning problem being solved (which is determined by the selected [loss function](loss-functions.md)).
    Problem | How transformation is performed
    ----- | -----
    Regression | [Quantization](quantization.md) is performed on the label value. The mode and number of buckets ($k+1$) are set in the starting parameters. All values located inside a single bucket are assigned a label value class – an integer in the range $[0;k]$ defined by the formula: `<bucket ID – 1>`.
    Classification | Possible values for label value are <q>0</q> (doesn't belong to the specified target class) and "1" (belongs to the specified target class).
    Multiclassification | The label values are integer identifiers of target classes (starting from "0").


1. Transforming categorical features to numerical features.

    The method is determined by the starting parameters.

    {% include [reusage-ctrs](../_includes/work_src/reusage/ctrs.md) %}

As a result, each categorical feature values or feature combination value is assigned a numerical feature.

 # Example of aggregating multiple features

 Assume that the objects in the training set belong to two categorical features: the musical genre (<q>rock</q>, <q>indie</q>) and the musical style (<q>dance</q>, <q>classical</q>). These features can occur in different combinations. {{ product }} can create a new feature that is a combination of those listed (<q>dance rock</q>, <q>classic rock</q>, <q>dance indie</q>, or <q>indie classical</q>). Any number of features can be combined.

 # Transforming categorical features to numerical features in classification

 1. {{ product }} accepts a set of object properties and model values as input.

     The table below shows what the results of this stage look like.

     Object # | $f_{1}$ | $f_{2}$ | ... | $f_{n}$ | Function value
     ----- | ----- | ----- | ----- | ----- | -----
     1 | 2 | 40 | ... | rock | 1
     2 | 3 | 55 | ... | indie | 0
     3 | 5 | 34 | ... | pop | 1
     4 | 2 | 45 | ... | rock | 0
     5 | 4 | 53 | ... | rock | 0
     6 | 2 | 48 | ... | indie | 1
     7 | 5 | 42 | ... | rock | 1
     ...

 1. The rows in the input file are randomly shuffled several times. Multiple random permutations are generated.

     {% include [algorithm-main-stages_cat-to-numberic-schematic-view](../_includes/concepts/algorithm-main-stages_cat-to-numberic/schematic-view.md) %}

     Object # | $f_{1}$ | $f_{2}$ | ... | $f_{n}$ | Function value
     ----- | ----- | ----- | ----- | ----- | -----
     1 | 4 | 53 | ... | rock | 0
     2 | 3 | 55 | ... | indie | 0
     3 | 2 | 40 | ... | rock | 1
     4 | 5 | 42 | ... | rock | 1
     5 | 5 | 34 | ... | pop | 1
     6 | 2 | 48 | ... | indie | 1
     7 | 2 | 45 | ... | rock | 0
     ...

 1. All categorical feature values are transformed to numerical using the following formula:
     $avg\_target = \frac{countInClass + prior}{totalCount + 1}$
     - $countInClass$ is how many times the label value was equal to <q>1</q> for objects with the current categorical feature value.
     - $prior$ is the preliminary value for the numerator. It is determined by the starting parameters.
     - $totalCount$ is the total number of objects (up to the current one) that have a categorical feature value matching the current one.

     {% note info %}

     These values are calculated individually for each object using data from previous objects.

     {% endnote %}

     In the example with musical genres, $j \in [1;3]$ accepts the values <q>rock</q>, <q>pop</q>, and <q>indie</q>, and prior is set to 0.05.

     {% include [algorithm-main-stages_cat-to-numberic-schematic-view](../_includes/concepts/algorithm-main-stages_cat-to-numberic/schematic-view.md) %}

     Object # | $f_{1}$ | $f_{2}$ | ... | $f_{n}$ | Function value
     ----- | ----- | ----- | ----- | ----- | -----
     1 | 4 | 53 | ... | 0,05 | 0
     2 | 3 | 55 | ... | 0,05 | 0
     3 | 2 | 40 | ... | 0,025 | 1
     4 | 5 | 42 | ... | 0,35 | 1
     5 | 5 | 34 | ... | 0,05 | 1
     6 | 2 | 48 | ... | 0,025 | 1
     7 | 2 | 45 | ... | 0,5125 | 0
     ...

One-hot encoding is also supported. Use one of the following training parameters to enable it.
Command-line version parameter | Python parameter | R parameter | Description
----- | ----- | ----- | -----
`--one-hot-max-size` | `one_hot_max_size` | `one_hot_max_size` | Use one-hot encoding for all categorical features with a number of different values less than or equal to the given parameter value. Ctrs are not calculated for such features.<br/><br/>See [details](../features/categorical-features.md).
