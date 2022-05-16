
Automatically calculate class weights based either on the total weight or the total number of objects in each class. The values are used as multipliers for the object weights.

Supported values:

- {{ autoclass__weights__default }}
- {{ autoclass__weights__balanced }}:
    
    $CW_k=\displaystyle\frac{max_{c=1}^K(\sum_{t_{i}=c}{w_i})}{\sum_{t_{i}=k}{w_{i}}}$
    
- {{ autoclass__weights__SqrtBalanced }}:
    
    $CW_k=\sqrt{\displaystyle\frac{max_{c=1}^K(\sum_{t_i=c}{w_i})}{\sum_{t_i=k}{w_i}}}$

    
