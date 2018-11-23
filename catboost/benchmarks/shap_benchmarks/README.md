# [SHAP](https://arxiv.org/abs/1705.07874) values comparison

1. Experiment infrastructure: 

	* GPU: Titan X Pascal
	* Intel(R) Xeon(R) CPU E5-2683 v3 @ 2.00GHz

	We trained and evaluated perfomance on GPU.

2. Parameters

	We run experiments on different depths for each 	library. ``max bin`` parameter was set up to 128 and 	other parameters were default for every library.
	
3. Dataset

	We used [Epsilon dataset](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html) (400К samples, 2000 features) to benchmark our performance on dense numerical dataset. We compared speed of shap values calculation on 10k samples from test. 
	
		
4. Results

Time in table is given in seconds and we didn't take into account time for data preprocessing.


|   depth     |   2  |   4  |   6  |   8  |   10  |        
|:-----------|:-----|:-----|:-----|:-----|:------|
| CatBoost   | 2.145| 2.079| **2.515**| **7.190**| **95.030**| 
| LightGBM   | 0.502 | 4.241 | 41.210 | 236.708 | 1188.494 |  
| XGBoost    | 0.495 | 1.996 | 12.864 | 74.940 | 298.284
