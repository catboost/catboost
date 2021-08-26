
Training and applying models for the classification problems. Provides compatibility with the scikit-learn tools.

The default optimized objective depends on various conditions:
- {{ error-function--Logit }} — The target has only two different values or the `target_border` parameter is not None.
- {{ error-function--MultiClass }} — The target has more than two different values and the `border_count` parameter is None.
