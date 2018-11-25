Benchmark that compares quality of GBDT packages on rossman-store-sales dataset.

## Results

### Hyperparameters tuned with hyperopt

Number of hyperopt iterations was set to 50, final model is tuned with best hyperparameters on all train data.

<table>
    <tr>
        <td>Experiment</td>
        <td>Best hyperparameters</td>
        <td>RMSE on test</td>
    </tr>
    <tr>
        <td>catboost with specifying cat features</td>
        <td>best_n_estimators = 1415<br>
params = {'random_seed': 0, 'learning_rate': 0.10663314690544494, 'iterations': 1500, 'od_wait': 100, 'one_hot_max_size': 143.0, 'bagging_temperature': 0.39933964736871874, 'random_strength': 1, 'depth': 8.0, 'loss_function': 'RMSE', 'l2_leaf_reg': 5.529962582104021, 'border_count': 254, 'boosting_type': 'Plain', 'bootstrap_type': 'Bayesian'}</td>
        <td><font color="green">489.75</font></td>
    </tr>
    <tr>
        <td>lightgbm with specifying cat features</td>
        <td>best_n_estimators = 3396<br>
params = {'num_leaves': 63, 'max_cat_threshold': 2, 'cat_l2': 12.93150760783131, 'verbose': -1, 'bagging_seed': 3, 'max_cat_to_onehot': 2, 'learning_rate': 0.12103165638430856, 'max_delta_step': 0.0, 'data_random_seed': 1, 'cat_smooth': 4.287437698866151, 'min_data_in_leaf': 26, 'bagging_fraction': 0.6207358917316325, 'min_data_per_group': 261, 'min_sum_hessian_in_leaf': 7.515138790064522e-05, 'feature_fraction_seed': 2, 'min_gain_to_split': 0.0, 'lambda_l1': 0, 'bagging_freq': 1, 'lambda_l2': 0.1709660204090765, 'max_depth': -1, 'objective': 'mean_squared_error', 'drop_seed': 4, 'metric': 'l2', 'feature_fraction': 0.8168930995735235}</td>
        <td>504.76</td>
    </tr>
    <tr>
        <td>xgboost</td>
        <td>best_n_estimators = 4011<br>
params = {'reg_alpha': 0.14747200224681817, 'tree_method': 'gpu_hist', 'colsample_bytree': 0.883176060062088, 'silent': 1, 'eval_metric': 'rmse', 'grow_policy': 'depthwise', 'learning_rate': 0.10032091014826115, 'subsample': 0.5740170782945163, 'reg_lambda': 0, 'max_bin': 1020, 'objective': 'reg:linear', 'min_split_loss': 0, 'max_depth': 7}</td>
        <td><font color="green">490.83</font></td>
    </tr>
</table>

### Early stopping with default hyperparameters

Max iterations limit was set to 9999 and `early_stopping_rounds` to 100.

Note that for CatBoost results differ between CPU and GPU implementations because ```border_count``` parameter has default value 254 in CPU mode and 128 in GPU mode.

#### Results on CPU

CPU - Intel Xeon E312xx (Sandy Bridge) VM, 16 cores.

<table>
    <tr>
        <td>Experiment</td>
        <td>Early stopping time (sec)</td>
        <td>RMSE on test</td>
        <td>Comments</td>
    </tr>
    <tr>
        <td>catboost w/o specifying cat features</td>
        <td>212.67</td>
        <td><font color="red">578.10</font></td>
        <td>reached max iterations limit</td>
    </tr>
    <tr>
        <td>catboost with specifying cat features</td>
        <td><font color="red">894.51</font></td>
        <td>520.07</td>
        <td/>
    </tr>
    <tr>
        <td>lightgbm w/o specifying cat features</td>
        <td>51.17</td>
        <td>499.67</td>
        <td/>
    </tr>
    <tr>
        <td>lightgbm with specifying cat features</td>
        <td><font color="green">9.90</font></td>
        <td><font color="green">490.57</font></td>
        <td/>
    </tr>
    <tr>
        <td>xgboost</td>
        <td>272.3</td>
        <td>567.8</td>
        <td>reached max iterations limit</td>
    </tr>
</table>

#### Results on GPU

GPU - 2x nVidia GeForce 1080 Ti.

<table>
    <tr>
        <td>Experiment</td>
        <td>Early stopping time (sec)</td>
        <td>RMSE on test</td>
        <td>Comments</td>
    </tr>
    <tr>
        <td>catboost w/o specifying cat features</td>
        <td><font color="green">39.5</font></td>
        <td>575.75</td>
        <td>reached max iterations limit</td>
    </tr>
    <tr>
        <td>catboost with specifying cat features</td>
        <td>90.83</td>
        <td>528.63</td>
        <td/>
    </tr>
    <tr>
        <td>lightgbm w/o specifying cat features</td>
        <td>97.93</td>
        <td><font color="green">501.22</font></td>
        <td/>
    </tr>
    <tr>
        <td>lightgbm with specifying cat features</td>
        <td>n/a</td>
        <td>n/a</td>
        <td><font color="red">Failed: [LightGBM] [Fatal] bin size 1093 cannot run on GPU</font>, see <a href="https://github.com/Microsoft/LightGBM/issues/1116">https://github.com/Microsoft/LightGBM/issues/1116</a> </td>
    </tr>
    <tr>
        <td>xgboost in 'gpu-exact' mode</td>
        <td>125.48</td>
        <td>566.55</td>
        <td>reached max iterations limit</td>
    </tr>
    <tr>
        <td>xgboost in 'gpu-hist' mode</td>
        <td>68.04</td>
        <td><font color="red">626.09</font></td>
        <td>reached max iterations limit</td>
    </tr>
</table>


### Hyperparameters tuned with RandomizedSearchCV

Hyperparameter distributions:

```
            'n_estimators' : LogUniform(100, 1000, True),
            'max_depth' : scipy.stats.randint(low=1, high=16),
            'learning_rate' : scipy.stats.uniform(0.01, 1.0)
```

(see experiments_lib.py file for LogUniform definition)

<h3>Results on CPU</h3>

CPU - Intel Xeon E312xx (Sandy Bridge) VM, 16 cores.

<table>
    <tr>
        <td>Experiment</td>
        <td>Time (sec)</td>
        <td>RMSE on test</td>
    </tr>
    <tr>
        <td>catboost w/o specifying cat features</td>
        <td>239.91</td>
        <td><font color="red">568.38</font></td>
    </tr>
    <tr>
        <td>catboost with specifying cat features</td>
        <td><font color="red">1145.98</font></td>
        <td>534.13</td>
    </tr>
    <tr>
        <td>lightgbm w/o specifying cat features</td>
        <td>105.02</td>
        <td>523.62</td>
    </tr>
    <tr>
        <td>lightgbm with specifying cat features</td>
        <td><font color="green">97.94</font></td>
        <td><font color="green">510.53</font></td>
    </tr>
    <tr>
        <td>xgboost</td>
        <td>437.8</td>
        <td>512.74</td>
    </tr>
</table>

## Requirements

OS - Linux (was tested on Ubuntu LTS 16.04)

Installed packages (via 'pip install'):
- kaggle
- hyperopt
- numpy
- pandas
- scipy
- scikit-learn

## GBDT packages versions

Tested on:
- catboost 0.11.0
- lightgbm 2.2.1
- xgboost 0.80
   
## How to run

- Download dataset from kaggle
- Preprocess it (extract features and save in CatBoost data format)
- Run benchmarks

(see 'run_all.sh' that does all these steps)


