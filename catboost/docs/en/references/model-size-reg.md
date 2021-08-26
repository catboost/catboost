# Model size regularization coefficient

This parameter influences the model size if training data has categorical features.

The information regarding categorical features makes a great contribution to the final size of the model. The mapping from the categorical feature value hash to some statistic values is stored for each categorical feature that is used in the model. The size of this mapping for a particular feature depends on the number of unique values that this feature takes.

Therefore, the potential weight of a categorical feature can be taken into account in the final model when choosing a split in a tree to reduce the final size of the model. When choosing the best split, all split scores are calculated and then the split with the best score is chosen. But before choosing the split with the best score, all scores change according to the following formula:
$s^{new} = s^{old} \cdot \left(1 + \frac{u}{U}\right)^M { , where}$
- $s^{new}$ is the new score for the split by some categorical feature or combination feature.
- $s^{old}$ is the old score for the split by the feature.
- $u$ is the cost of the combination. Calculation principles depend on the processing unit type:
    #### {{ calcer_type__cpu }}
    The cost of a combination is equal to the number of different feature values in this combinations that are present in the training dataset.
    #### {{ calcer_type__gpu }}
    The cost of a combination is equal to number of all possible different values of this combination. For example, if the combination contains two categorical features (c1 and c2), the cost is calculated as $number\_of\_categories\_in\_c1 \cdot number\_of\_categories\_in\_c2$, even though many of the values from this combination might not be present in the dataset.

- $U$ depends on the processing unit type:
    #### {{ calcer_type__cpu }}

    The maximum of all $u$ values among all features.

    #### {{ calcer_type__gpu }}

    The maximum of all $u$ values among all features processed on this device.

- $M$ is the value of the `model_size_reg` (`--model-size-reg`) parameter.

Thus, the score of the split by the feature is reduced depending on how much it affects the final model size.


## Calculation differences for {{ calcer_type__cpu }} and {{ calcer_type__gpu }} {#cpu-vs-gpu}

{% include [model-reg-size-model-size-reg-gpu-vs-cpu](../_includes/work_src/reusage-common-phrases/model-size-reg-gpu-vs-cpu.md) %}


