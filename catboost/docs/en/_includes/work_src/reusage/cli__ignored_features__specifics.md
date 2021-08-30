
Specifics:

- Non-negative indices that do not match any features are successfully ignored. For example, if five features are defined for the objects in the dataset and this parameter is set to <q>42</q>, the corresponding non-existing feature is successfully ignored.
    
- The identifier corresponds to the feature's index. Feature indices used in train and feature importance are numbered from 0 to `featureCount – 1`. If a file is used as [input data](../../../concepts/input-data.md) then any non-feature column types are ignored when calculating these indices. For example, each row in the input file contains data in the following order: `cat feature<\t>label value<\t>num feature`. So for the row `rock<\t>0<\t>42`, the identifier for the <q>rock</q> feature is 0, and for the <q>42</q> feature it's 1.
    
