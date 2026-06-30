# set_params

{% include [sections-with-methods-desc-set_params--purpose](../_includes/work_src/reusage/set_params--purpose.md) %}


## {{ dl--invoke-format }} {#call-format}

```
set_params(** params)
```

## {{ dl--parameters }} {#parameters}

### **params

#### Description

A list of parameters to start training with.

If omitted, default values are used.

If set, the passed list of parameters overrides the default values.

Format:
```
parameter_1=<value>, parameter_2=<value>, ..., parameter_N=<value>
```

An example of the method call:

```
model.set_params(iterations=500, thread_count=2, use_best_model=True)
```

{% include [sections-with-methods-desc-see-training-params](../_includes/work_src/reusage/see-training-params.md) %}

**Possible types**

key=value format

**Default value**

{{ python--required }}

