# catboost.shrink

```r
catboost.shrink(model,
                ntree_end,
                ntree_start = {{ shrink__ntree_start__default }})
```

## {{ dl--purpose }} {#purpose}

{% include [sections-with-methods-desc-shrink__purpose](../_includes/work_src/reusage/shrink__purpose.md) %}


## {{ dl--args }} {#arguments}
### model

#### Description

The model obtained as the result of training.


**Default value**

{{ r--required }}

### ntree_end


#### Description
А zero-based index of the first tree not to be used.

**Default value**

{{ shrink__ntree_end__default }}

### ntree_start


#### Description
А zero-based index of the first tree to be used.

**Default value**

{{ shrink__ntree_start__default }}

