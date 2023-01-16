
Feature combinations are regularized more aggressively on {{ calcer_type__gpu }}.

#### {{ calcer_type__cpu }}
The cost of a combination is equal to the number of different feature values in this combinations that are present in the training dataset. 
#### {{ calcer_type__gpu }}
The cost of a combination is equal to number of all possible different values of this combination. For example, if the combination contains two categorical features (c1 and c2), the cost is calculated as $number\_of\_categories\_in\_c1 \cdot number\_of\_categories\_in\_c2$, even though many of the values from this combination might not be present in the dataset.
