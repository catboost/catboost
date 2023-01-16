# {{ title__ShapValues }}

#### {{ output--contains }}

A vector $v$ with contributions of each feature to the prediction for every input object and the expected value of the model prediction for the object (average prediction given no knowledge about the object).

{% include [reusage-formats-use-the-shap-package](../_includes/work_src/reusage-formats/use-the-shap-package.md) %}


#### {{ output--format }}

- The rows are sorted in the same order as the order of objects in the input dataset.
    
- Each row contains information related to one object from the input dataset.
    
    Format:
    ```
    <contribution of feature 1><\t><contribution of feature 2><\t> .. <\t><contribution of feature N><\t><expected value of the model prediction>
    ```
    

#### {{ output--example }}

```
-0.0001401524197<\t>0.0001269417313<\t>0.004920700379<\t>0,00490749
```

