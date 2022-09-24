# Variables used in formulas

The following common variables are used in formulas of the described metrics:

- $t_{i}$ is the label value for the i-th object (from the input data for training).
- $a_{i}$ is the result of applying the model to the i-th object.
- $p_{i}$ is the predicted success probability $\left(p_{i} = \frac{1}{1 + e^{-a_{i}}}\right)$
- $N$ is the total number of objects.
- $M$ is the number of classes.
- $c_{i}$ is the class of the object for binary classification.

    $\begin{cases} c_{i} = 0{ , } & t_{i} \leqslant border \\ c_{i} = 1{ , } & t_{i} > border \end{cases}$

- $w_{i}$ is the weight of the i-th object. It is set in the [dataset description](input-data_values-file.md) in columns with the `Weight`[type](input-data_column-descfile.md) (if otherwise is not stated) or in the `sample_weight` parameter of the {{ python-package }}. The default is 1 for all objects.
- $P$, $TP$, $TN$, $FP$, $FN$ are abbreviations for Positive, True Positive, True Negative, False Positive and False Negative.

    By default, $P$, $TP$, $TN$, $FP$, $FN$ use weights. For example, $TP = \sum\limits_{i=1}^{N} w_{i} [p_{i} > 0.5] c_{i}$

- $Pairs$ is the array of pairs specified in the [Pairs description](input-data_pairs-description.md) or in the `pairs` parameter of the {{ python-package }}.
- $N_{Pairs}$ is the number of pairs for the Pairwise metrics.
- $a_{p}$ is the value calculated using the resulting model for the winner object for the Pairwise metrics.
- $a_{n}$ is the value calculated using the resulting model for the loser object for the Pairwise metrics.
- $w_{pn}$ is the weight of the ($p$; $n$) pair for the Pairwise metrics.
- $Group$ is the array of object identifiers from the input dataset with a common `{{ cd-file__col-type__GroupId }}`. It is used to calculate the Groupwise metrics.
- $Groups$ is the set of all arrays of identifiers from the input dataset with a common `{{ cd-file__col-type__GroupId }}`. It is used to calculate the Groupwise metrics.
