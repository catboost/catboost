# SHAP values comparison


We used [Epsilon dataset](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html) (400К samples, 2000 features) to benchmark our performance on dense numerical dataset. We compared speed of shap values calculation on 10k samples from test. We run experiments on different depths and other parameters were equal for each library. In the table we report time for default 128 bins. Training and evaluation has been done on Titan Pascal X GPU.


|            |   2  |   4  |   6  |   8  |   10  |        
|:-----------|:-----|:-----|:-----|:-----|:------|
| CatBoost   | 2.145| 2.079| **2.515**| **7.190**| **95.030**| 
| LightGBM   | 0.502 | 4.241 | 41.210 | 236.708 | 1188.494 |  
| XGBoost    | 0.495 | 1.996 | 12.864 | 74.940 | 298.284
