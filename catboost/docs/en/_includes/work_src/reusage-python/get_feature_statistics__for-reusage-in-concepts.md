
An example of plotted statistics:
![](../../../images/calc_feature_statistics__adult.png)
The X-axis of the resulting chart contains values of the feature divided into buckets. For numerical features, the splits between buckets represent conditions (`feature < value`) from the trees of the model. For categorical features, each bucket stands for a category.

The Y-axis of the resulting chart contains the following graphs:

- {% include [get_feature_statistics-python__statistics__mean-target__p](python__statistics__mean-target__p.md) %}
    
- {% include [get_feature_statistics-python__statistics__mean-prediction__p](python__statistics__mean-prediction__p.md) %}
    
- {% include [get_feature_statistics-python__statistics__object-per-bin](python__statistics__object-per-bin.md) %}
    
- {% include [get_feature_statistics-python__statistics__predictions-on-varying-feature](python__statistics__predictions-on-varying-feature.md) %}

The return value of the function contains the data from these graps.

The following information is used for calculation:

- Trained model
- Dataset
- Label values
