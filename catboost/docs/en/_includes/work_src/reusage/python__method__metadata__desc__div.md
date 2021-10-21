
Return a proxy object with metadata from the model's internal key-value string storage. This proxy object mimics a Python dictionary with string-only keys and values. All modifying operations write changes directly to the inner C++ model object behind the corresponding class.

By default, a trained {{ product }} model contains the following metadata keys:
- `catboost_version_info` — The {{ product }} library version information, including revision details.


{% cut "Example" %}


    #### {{ input_data__title__example }}
    
    #### {{ header__key-invoke-format }}
    
    ```python
    print(metadata["catboost_version_info"])
    ```
    
    #### {{ header__key-format-example }}
    
    ```
    Git info:
    Commit: 73f2e5e34e8add6f2b6bd22b4c0cc1c0456ec7da
    Author: 'format:mail <mail@example.com>'
    
    Summary: Fix a bug caused by difference in indices in CoreML model and source flat feature indices. Canonize GPU CoreML onehot tests. MLTOOLS-2436.
    
    Other info:
    Build by: go
    Top src dir: /var/lib/go-agent/pipelines/BuildMaster/catboost.git
    Top build dir: /storage/.ya/build/build_root/zj35/0000e3
    Hostname: catboost-builder-linux
    Host information:
    Linux catboost-builder-linux 4.4.0-139-generic #165-Ubuntu SMP Wed Oct 24 10:58:50 UTC 2018 x86_64 x86_64 x86_64 GNU/Linux
    ```
{% endcut %}


- `model_guid` — A randomly generated identifier of the trained model. This key is used for model identification purposes.


{% cut "Example" %}


    #### {{ input_data__title__example }}
    
    #### {{ header__key-invoke-format }}
    
    ```python
    print(metadata["model_guid"])
    ```
    
    #### {{ header__key-format-example }}
    
    ```
    47ebcf93-81d7170a-2e69cf54-804224ed
    ```
    
{% endcut %}


- `params` — A JSON dictionary with full training parameters serialized to a string.


{% cut "Example" %}


    #### {{ input_data__title__example }}
    
    #### {{ header__key-invoke-format }}
    
    ```python
    print(metadata["params"])
    ```
    
    #### {{ header__key-format-example }}
    
    ```
    {"detailed_profile":false,"boosting_options":{"approx_on_full_history":false,"fold_len_multiplier":2,"fold_permutation_block":0,"boosting_type":"Ordered","iterations":1000,"od_config":{"wait_iterations":20,"type":"None","stop_pvalue":0},"permutation_count":4,"learning_rate":0.02999999933},"metrics":{"objective_metric":{"type":"RMSE","params":{}},"eval_metric":{"type":"RMSE","params":{}},"custom_metrics":[]},"metadata":{},"cat_feature_params":{"store_all_simple_ctr":false,"ctr_leaf_count_limit":18446744073709551615,"simple_ctrs":[{"ctr_binarization":{"border_count":15,"border_type":"Uniform"},"target_binarization":{"border_count":1,"border_type":"MinEntropy"},"prior_estimation":"No","priors":[[0,1],[0.5,1],[1,1]],"ctr_type":"Borders"},{"ctr_binarization":{"border_count":15,"border_type":"Uniform"},"prior_estimation":"No","priors":[[0,1]],"ctr_type":"Counter"}],"counter_calc_method":"SkipTest","one_hot_max_size":2,"max_ctr_complexity":4,"combinations_ctrs":[{"ctr_binarization":{"border_count":15,"border_type":"Uniform"},"target_binarization":{"border_count":1,"border_type":"MinEntropy"},"prior_estimation":"No","priors":[[0,1],[0.5,1],[1,1]],"ctr_type":"Borders"},{"ctr_binarization":{"border_count":15,"border_type":"Uniform"},"prior_estimation":"No","priors":[[0,1]],"ctr_type":"Counter"}],"target_binarization":{"border_count":1,"border_type":"MinEntropy"},"per_feature_ctrs":{}},"logging_level":"Verbose","data_processing_options":{"has_time":false,"allow_const_label":false,"class_names":[],"class_weights":[],"target_border":null,"float_features_binarization":{"border_count":254,"nan_mode":"Min","border_type":"GreedyLogSum"},"classes_count":0,"ignored_features":[]},"loss_function":{"type":"RMSE","params":{}},"tree_learner_options":{"rsm":1,"random_strength":1,"leaf_estimation_iterations":1,"dev_efb_max_buckets":1024,"dev_score_calc_obj_block_size":5000000,"leaf_estimation_backtracking":"AnyImprovement","bayesian_matrix_reg":0.1000000015,"leaf_estimation_method":"Newton","sampling_frequency":"PerTree","model_size_reg":0.5,"bootstrap":{"bagging_temperature":1,"type":"Bayesian"},"efb_max_conflict_fraction":0,"l2_leaf_reg":3,"depth":6},"task_type":"CPU","flat_params":{"verbose":1},"random_seed":0,"system_options":{"thread_count":4,"file_with_hosts":"hosts.txt","node_type":"SingleHost","node_port":0,"used_ram_limit":""}}
    ```

{% endcut %}


- `train_finish_time` — Date and time of model training completion.

{% cut "Example" %}

    #### {{ input_data__title__example }}
    
    #### {{ header__key-invoke-format }}
    
    ```python
    print(metadata["train_finish_time"])
    ```
    
    #### {{ header__key-format-example }}
    
    ```
    2019-06-04T06:30:13Z
    ```
{% endcut %}
