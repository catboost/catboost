
The following column types are supported:
- {{ r-types--double }}
- {{ r-types--factor }}. It is assumed that categorical features are given in this type of columns. A standard {{ product }} processing procedure is applied to this type of columns:
    1. The values are converted to strings.
    1. The `ConvertCatFeatureToFloat` function is applied to the resulting string.
