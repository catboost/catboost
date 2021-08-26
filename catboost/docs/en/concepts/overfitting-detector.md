# Overfitting detector

{% include [overfitting-detector-od__purpose](../_includes/work_src/reusage-common-phrases/od__purpose.md) %}


The following overfitting detection methods are supported:
- [{{ fit--od-type-inctodec }}](#inctodec)
- [{{ fit--od-type-iter }}](#iter)


## {{ fit--od-type-inctodec }} {#inctodec}

Before building each new tree, {{ product }} checks the resulting loss change on the validation dataset. The overfit detector is triggered if the $Threshold$ value set in the starting parameters is greater than $CurrentPValue$:

$CurrentPValue < Threshold$

How $CurrentPValue$ is calculated from a set of values for the maximizing metric $score[i]$:
1. $ExpectedInc$ is calculated:

    $ExpectedInc = max_{i_{1} \leq i_{2} \leq i } 0.99^{i - i_{1}} \cdot (score[i_{2}] - score[i_{1}])$

1. $x$ is calculated:

    $x = \frac{ExpectedInc[i]}{max_{j \leq i} { } score[j] - score[i]}$

1. $CurrentPValue$ is calculated:

    $CurrentPValue = exp \left(- \frac{0.5}{x}\right)$


## {{ fit--od-type-iter }} {#iter}

Before building each new tree, {{ product }} checks the number of iterations since the iteration with the optimal [loss function](loss-functions.md) value.

The model is considered overfitted if the number of iterations exceeds the value specified in the training parameters.
