# get_feature_names

{% include [featuresdata-get_feature_names-desc](../_includes/work_src/reusage-python/get_feature_names-desc.md) %}


## {{ dl--invoke-format }} {#call-format}

```python
get_feature_names()
```

## {{ dl--output-format }} {#output-format}

List of strings.

## {{ dl__usage-examples }} {#usage-examples}

#### Feature names are not set

```python
import numpy as np
from catboost import FeaturesData

fd = FeaturesData(
    num_feature_data=np.array([[1, 4, 5, 6], [4, 5, 6, 7], [30, 40, 50, 60]], dtype=np.float32),
    cat_feature_data=np.array([["a", "b"], ["a", "b"], ["c", "d"]], dtype=object)
)
# print feature names
# the returned value is ['', '', '', '', '', ''] as neither num_feature_names nor cat_feature_names are specified
print(fd.get_feature_names())
```

#### Feature names are set

```python
import numpy as np
from catboost import FeaturesData

fd = FeaturesData(
    num_feature_data=np.array([[1, 4, 5, 6], [4, 5, 6, 7], [30, 40, 50, 60]], dtype=np.float32),
    num_feature_names=['num_feat0', 'num_feat1', 'num_feat2', 'num_feat3'],
    cat_feature_data=np.array([["a", "b"], ["a", "b"], ["c", "d"]], dtype=object),
    cat_feature_names=['cat_feat0', 'cat_feat1']
)
# prints ['num_feat0', 'num_feat1', 'num_feat2', 'num_feat3', 'cat_feat0', 'cat_feat1']
print(fd.get_feature_names())
```

