# {{ error-function__QueryAverage }}

- [{{ title__loss-functions__calculation-principles }}](#calculation)
- [{{ title__loss-functions__text__user-defined-params }}](#user-defined-parameters)

## {{ title__loss-functions__calculation-principles }} {#calculation}

{% include [loss-functions-function-calculation](../_includes/work_src/reusage-common-phrases/function-calculation.md) %}


1. Model values are calculated for the objects from the input dataset.

1. Top $M$ model values are selected for each group. The quantity $M$ is user-defined.

    For example, let's assume that the number of top model values is limited to 2 and the following values are calculated for the input dataset:
    ```no-highlight
    Document ID    Model value
    1              10.4
    2              20.1
    3              1.1
    ```

    In this case, the objects with indices 2 and 1 are selected.

1. The average of the label values is calculated for the objects selected at step [2](#step-top-label-values).

    For example, if the dataset consists of one group and the documents match the ones mentioned in the description of step [2](#step-top-label-values), the {{ error-function__QueryAverage }} metric is calculated as follows:

    $QueryAverage = \displaystyle\frac{LabelValue_{object2} + LabelValue_{object1}}{2}$



## {{ title__loss-functions__text__user-defined-params }} {#user-defined-parameters}

### top

#### Description


{% include [reusage-loss-functions-top__desc](../_includes/work_src/reusage-loss-functions/top__desc.md) %}


{{ loss-functions__params__top__default }}


