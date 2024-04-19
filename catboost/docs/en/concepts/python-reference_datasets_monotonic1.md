# monotonic1

{% include [datasets-datasets__monotonic1](../_includes/work_src/reusage-python/datasets__monotonic1.md) %}


This dataset can be used for regression.

It contains several numerical and categorical features.

The contents of columns depends on the name or on the pattern of the name of the corresponding column:
- `Target` (the first column) — Target values.

- `Cat*` — Categorical features.

- `Num*` — Numerical features.

- `MonotonicNeg*` — Numerical features, for which monotonic constraints must hold.

    If values of such features decrease, then the prediction value must not decrease. Thus, if there are two objects $x_{1}$ and $x_{2}$ with all features being equal except for a monotonic negative feature M, such that $x_{1}[M] > x_{2}[M]$, then the following inequality must be met for predictions:

    $f(x_{1}) \leq f(x_{2})$

## {{ dl--invoke-format }} {#method-call}

```python
monotonic1()
```

## {{ dl--output-format }} {#type-of-return-value}

{% include [datasets-datasets__output](../_includes/work_src/reusage-python/datasets__output.md) %}


## {{ dl__usage-examples }} {#usage-examples}

```python
from catboost.datasets import monotonic1
monotonic1_train, monotonic1_test = monotonic1()

print(monotonic1_train.head(3))

```

{% include [reusage-common-phrases-example-output](../_includes/work_src/reusage-common-phrases/example-output.md) %}


```no-highlight
     Target      Num0                              Cat0                              Cat1                              Cat2                              Cat3  ... MonotonicNeg10     Num24     Num25     Num26     Num27     Num28
0  0.011655  0.078947  20e17f9f3b3e82f1aaff5b98b03cd8ac  7a5b46681f07babda8b98d095458a000  772e5fb884f6c15057b1d3ed9a8cf8b2  eccf825029482fceedeb96f3dfccb7d4  ...       0.019833  0.220801  0.088723  0.316292  0.999997  0.999994
1  0.011655  0.039474  c1ce747b53bb9886fd3814788bd6016a  8a36f4f35f490a01466c5c9a682db621  68d0b1837ca49afba289deb74715c363  e1e1db28712431600d2df4d63c872701  ...       0.000727  0.847325  0.999976  0.811932  0.017582  0.999994
2  0.000000  0.144737  f880f22c2c9a2f8711ef66785f26bcc3  6b6f19508a3baa90b3503b7b2c0d784f  4342d5154ad04acc6ce1031bc3c488c0  ba34b6b8c8b64d9c80d859caa0b314ed  ...       0.018917  0.996521  0.249741  0.175907  0.004534  0.999994
```

