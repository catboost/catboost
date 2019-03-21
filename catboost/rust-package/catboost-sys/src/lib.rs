#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CStr;
    use std::slice;
    use std::str;

    #[test]
    fn it_works() {
        let model_handle = unsafe { ModelCalcerCreate() };
        let ret_val = unsafe {
            LoadFullModelFromFile(
                model_handle,
                std::ffi::CString::new("../tmp/model.bin").unwrap().as_ptr(),
            )
        };
        if !ret_val {
            let c_str = unsafe { CStr::from_ptr(GetErrorString()) };
            let str_slice = c_str.to_str().unwrap();
            panic!(str_slice);
        }

        let tree_count = unsafe { GetTreeCount(model_handle) };
        println!("Loaded model with {} trees.", tree_count);

        let float_features_count = unsafe { GetFloatFeaturesCount(model_handle) };
        let cat_features_count = unsafe { GetCatFeaturesCount(model_handle) };
        println!(
            "Model expects {} float features and {} categorical features",
            float_features_count, cat_features_count
        );

        let params_key = std::ffi::CString::new("params").unwrap();
        let model_metadata_has_key = unsafe {
            CheckModelMetadataHasKey(
                model_handle,
                params_key.as_ptr(),
                params_key.as_bytes().len(),
            )
        };
        if model_metadata_has_key {
            let params = unsafe {
                CStr::from_ptr(GetModelInfoValue(
                    model_handle,
                    params_key.as_ptr(),
                    params_key.as_bytes().len(),
                ))
            };
            println!(
                "Applying model trained with params: {}",
                params.to_str().unwrap()
            );
        }

        let float_features = vec![
            vec![-10.0, 5.0, 753.0],
            vec![30.0, 1.0, 760.0],
            vec![40.0, 0.1, 705.0],
        ];
        // let float_features_ptr = &float_features.map(|x| x.as_ptr()).collect::<Vec<_>>();
        let float_features_ptr = vec![
            float_features[0].as_ptr(),
            float_features[1].as_ptr(),
            float_features[2].as_ptr(),
        ];
        let cat_features = vec![
            vec!["north", "Memphis TN"],
            vec!["south", "Los Angeles CA"],
            vec!["south", "Las Vegas NV"],
        ];

        // const size_t docCount = 3;
        // const size_t floatFeaturesCount = 6;
        // const float floatFeatures[docCount ][floatFeaturesCount ] = {
        //     {28.0, 120135.0, 11.0, 0.0, 0.0, 40.0},
        //     {49.0, 57665.0, 13.0, 0.0, 0.0, 40.0},
        //     {34.0, 355700.0, 9.0, 0.0, 0.0, 20.0}
        // };
        // const float* floatFeaturesPtrs[docCount] = {
        //     floatFeatures[0],
        //     floatFeatures[1],
        //     floatFeatures[2]
        // };
        // const size_t catFeaturesCount = 8;
        // const char* catFeatures[docCount][8] = {
        //     {"Private", "Assoc-voc", "Never-married", "Sales", "Not-in-family", "White", "Female", "United-States"},
        //     {"?", "Bachelors", "Divorced", "?", "Own-child", "White", "Female", "United-States"},
        //     {"State-gov", "HS-grad", "Separated", "Adm-clerical", "Unmarried", "White", "Female", "United-States"}
        // };
        // const char** catFeaturesPtrs[docCount] = {
        //     catFeatures[0],
        //     catFeatures[1],
        //     catFeatures[2]
        // };
        // double result[3] = { 0 };
        // if (!CalcModelPrediction(
        //     modelHandle,
        //     docCount,
        //     floatFeaturesPtrs, floatFeaturesCount,
        //     catFeaturesPtrs, catFeaturesCount,
        //     result, docCount)
        // ) {
        //     std::cout << "Prediction failed: " << GetErrorString() << std::endl;
        //     return;
        // }
        // std::cout << "Results: ";
        // for (size_t i = 0; i < 3; ++i) {
        //     std::cout << result[i] << " ";
        // }
        // std::cout << std::endl;
        // /* Sometimes you need to evaluate model on single object.
        //    We provide special method for this case which is prettier and is little faster than calling batch evaluation for single object
        // */
        // double singleResult = 0.0;
        // if (!CalcModelPredictionSingle(
        //     modelHandle,
        //     floatFeatures[0], floatFeaturesCount,
        //     catFeatures[0], catFeaturesCount,
        //     &singleResult, 1)
        // ) {
        //     std::cout << "Single prediction failed: " << GetErrorString() << std::endl;
        //     return;
        // }
        // std::cout << "Single prediction: " << singleResult << std::endl;
        // }
        unsafe { ModelCalcerDelete(model_handle) };
    }
}
