
Use only if the `data` parameter is a two-dimensional feature matrix (has one of the following types: {{ python-type--list }}, {{ python-type__np_ndarray }}, {{ python-type--pandasDataFrame }}, {{ python-type--pandasSeries }}).

If any elements in this array are specified as names instead of indices, names for all columns must be provided. To do this, either use the `feature_names` parameter of this constructor to explicitly specify them or pass a {{ python-type--pandasDataFrame }} with column names specified in the `data` parameter.
