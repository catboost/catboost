# calc_leaf_indexes

Returns indexes of leafs to which objects from pool are mapped by model trees.

## {{ dl--invoke-format }} {#python__calc_leaf_indexes__call-format}

```python
calc_leaf_indexes(data, ntree_start=0, ntree_end=0, thread_count=-1, verbose=False)
```

## {{ dl--parameters }} {#params}

### data

#### Description

A file or matrix with the input dataset.

**Possible values** 

{{ python-type--pool }}

**Default value** 

{{ python--required }}


### ntree_start

#### Description

To reduce the number of trees to use when the model is applied or the metrics are calculated, setthe range of the tree indices to`[ntree_start; ntree_end)`.

{% include [eval-start-end-ntree_start__short-param-desc](../_includes/work_src/reusage-common-phrases/ntree_start__short-param-desc.md) %}

**Possible values** 

{{ python-type--int }}

**Default value** 

{{ fit--ntree_start }}


### ntree_end

#### Description

To reduce the number of trees to use when the model is applied or the metrics are calculated, setthe range of the tree indices to`[ntree_start; ntree_end)` and the step of the trees to use to`eval_period`.

{% include [eval-start-end-ntree_end__short-param-desc](../_includes/work_src/reusage-common-phrases/ntree_end__short-param-desc.md) %}

**Possible values** 

{{ python-type--int }}

**Default value** 

{{ fit--ntree_end }}

### thread_count

#### Description

{% include [reusage-thread-count-short-desc](../_includes/work_src/reusage/thread-count-short-desc.md) %}


{% include [reusage-thread_count__cpu_cores__optimizes-the-speed-of-execution](../_includes/work_src/reusage/thread_count__cpu_cores__optimizes-the-speed-of-execution.md) %}

**Possible values** 

{{ python-type--int }}

**Default value** 

{{ fit__thread_count__wrappers }}


### verbose

#### Description

Enable debug logging level.

**Possible values** 

bool 

**Default value** 

False



## {{ dl--output-format }} {#output}

leaf_indexes : 2-dimensional numpy.ndarray of numpy.uint32 with shape (object count, ntree_end – ntree_start). i-th row is an array of leaf indexes for i-th object.
