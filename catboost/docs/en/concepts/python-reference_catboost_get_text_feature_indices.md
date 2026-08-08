# get_text_feature_indices

Return the indices of text features used by the model. The method is available after fitting or loading a model.

## {{ dl--invoke-format }} {#method-call-format}

```python
get_text_feature_indices()
```

## {{ dl--output-format }} {#output-format}

{{ python-type--list }} of integer feature indices.

To recover the type of every feature, call this method together with
`get_cat_feature_indices()` and `get_embedding_feature_indices()`. Feature indices that are
not returned by any of these methods are numerical.
