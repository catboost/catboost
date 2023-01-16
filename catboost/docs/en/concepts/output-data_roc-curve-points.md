# ROC curve points

#### {{ output--contains }}
Points of the ROC curve.
#### {{ output__header-format }}

{% include [reusage-formats-header__intro](../_includes/work_src/reusage-formats/header__intro.md) %}


Format:
```
FPR</t>TPR</t>Threshold
```

- `FPR` is the false positive rate.
- `TPR` is the true positive rate.
- `Threshold` is the probability boundary required to achieve the specified false and true positive rates.

#### {{ output--format }}

Each row starting from the second contains tab-separated information regarding FPR and TPR and the corresponding boundary.

#### {{ output--example }}

```
FPR</t>TPR</t>Threshold
0</t>0</t>1
0</t>0.01265822785</t>0.9963511558
0</t>0.0253164557</t>0.996198873
0</t>0.6582278481</t>0.9772358693
0</t>0.7215189873</t>0.9607024818
0</t>0.746835443</t>0.9353759324
0.04545454545</t>0.746835443</t>0.92152338
0.04545454545</t>0.7594936709</t>0.8998709593
0.1818181818</t>0.8101265823</t>0.7042075722
0.2272727273</t>0.8987341772</t>0.6193897296
0.2727272727</t>0.8987341772</t>0.5973261646
0.2727272727</t>0.9113924051</t>0.5344706824
0.3636363636</t>0.9113924051</t>0.4197043336
0.5454545455</t>0.9367088608</t>0.3030610228
0.5909090909</t>0.9746835443</t>0.2358379561
0.6363636364</t>0.9746835443</t>0.2023097788
0.8181818182</t>0.9746835443</t>0.1253457115
0.8636363636</t>0.9746835443</t>0.1033187262
0.9545454545</t>0.9873417722</t>0.0490713091
1</t>0.9873417722</t>0.02776289088
1</t>1</t>0
```

