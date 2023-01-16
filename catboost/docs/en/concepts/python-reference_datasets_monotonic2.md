# monotonic2

{% include [datasets-datasets__monotonic2](../_includes/work_src/reusage-python/datasets__monotonic2.md) %}


This dataset can be used for regression.

The contents of columns depends on the name or on the pattern of the name of the corresponding column:

- `Target`(the first column) — Target values.
    
- `MonotonicNeg*` — Monotonic negative numerical features.
    
    If values of such features decrease, then the prediction value must not decrease. Thus, if there are two objects $x_{1}$ and $x_{2}$ with all features being equal except for a monotonic negative feature $MNeg$, such that $x_{1}[MNeg] > x_{2}[MNeg]$, then the following inequality must be met for predictions:
    
    $f(x_{1}) \leq f(x_{2})$
    
- `MonotonicPos*` — Monotonic positive numerical features.
    
    If values of such features decrease, then the prediction value must not increase. Thus, if there are two objects $x_{1}$ and $x_{2}$ with all features being equal except for a monotonic positive feature $MPos$, such that $x_{1}[MPos] > x_{2}[MPos]$, then the following inequality must be met for predictions:
    
    $f(x_{1}) \geq f(x_{2})$
    

## {{ dl--invoke-format }} {#method-call}

```python
monotonic2()
```

## {{ dl--output-format }} {#type-of-return-value}

{% include [datasets-datasets__output](../_includes/work_src/reusage-python/datasets__output.md) %}


## {{ dl__usage-examples }} {#usage-examples}

```python
from catboost.datasets import monotonic2
monotonic2_train, monotonic2_test = monotonic2()

print(monotonic2_train.head(3))

```

{% include [reusage-common-phrases-example-output](../_includes/work_src/reusage-common-phrases/example-output.md) %}


```no-highlight
   Target  MonotonicNeg0  MonotonicPos0  MonotonicPos1  MonotonicNeg1
0     0.0            NaN            NaN       0.010356       0.032638
1     0.0            NaN            NaN       0.010356       0.032638
```

