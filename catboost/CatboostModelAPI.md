If you want to use a trained model, we provide 4 different ways to do this: C++ while building with ya.make, Python API, dynamic C library with C++ wrapper and C++ header-only evaluator.

## ya.make based projects
If you use `ya.make` build system, the most convenient interface for model evaluation is ```TFullModel``` class, defined in [catboost/libs/model/model.h](https://github.com/catboost/catboost/tree/master/catboost/libs/model/model.h).
Small example:
```cpp
model = ReadModel(modelPath);

double result = 0.;
const TVector<float> floatFeatures[] = {{1, 2, 3}};
const TVector<TStringBuf> catFeatures[] = {{"a", "b", "c"}};
model.Calc(floatFeatures, catFeatures, MakeArrayRef(&result, 1));
```

## Python API
The simplest but not the fastest way to evaluate model predictions, just use ```model.predict```.
Small example:
```python
model = CatBoostClassifier()
model.load_model('catboost_model.dump')
model.predict(object_features)
```
If your data is bigger than 1000 objects we recommend you to use ```catboost.TPool``` class to get full performance.

Read python API documentation [here](https://catboost.ai/docs/concepts/python-quickstart.html) for more information.


## Dynamic library ```libcatboostmodel.(so|dll|dylib)```
This is the fastest way to evaluate model. Library interface is C API but we provide simple C++ header-only wrapper.
but you can easily use it's C API interface from any language you like.
The code of library and small CMake usage example can be found in [model_interface](https://github.com/catboost/catboost/tree/master/catboost/libs/model_interface)
For debian based linux distributions we provide catboost-model-library debian package that comes with dynamic library`/usr/lib/libcatboostmodel.so.1`, symlink to current version `/usr/lib/libcatboostmodel.so -> /usr/lib/libcatboostmodel.so.1`
also it provides two header files:
```
/usr/include/catboost_model/c_api.h
/usr/include/catboost_model/wrapped_calcer.h
```
For other platforms you can build this library with ymake:
```bash
ya make -r catboost/libs/model_interface/
```

With this library you can:
 - Load model from file or initialize it from memory
 - Evaluate model
 - Access basic model information: tree count, float features count, categorical features count and model metadata.
**NOTICE:** Currently library provides read-only model methods only. If you need model metadata modification or model truncation abilities - feel free to contact catboost team for API extension.


We will describe process of model integration for model trained on Adult dataset with model trained via CLI catboost with that command:
`./catboost_linux fit -f adult/train --cd adult/cd -i 500 -m adult_500.cbm`

### C API
You can find some useful information in doxygen-style documentation in [c_api.h](https://github.com/catboost/catboost/tree/master/catboost/libs/model_interface/c_api.h)

**Note:** if trained model uses only numeric features you, can switch evaluation backend to CUDA supporting GPU with `EnableGPUEvaluation` method both in C API and C++ wrapper.

Sample C code:
```cpp
#include <catboost_model/c_api.h> // this include is valid only for debian
#include <iostream>
#include <string>

void catboost_demo() {
    ModelCalcerHandle* modelHandle = ModelCalcerCreate();
    if (!modelHandle) {
        std::cout << "Model handle creation failed: " << GetErrorString() << std::endl;
        return;
    }
    if (!LoadFullModelFromFile(modelHandle, "adult_500.cbm")) {
        std::cout << "Load model failed: " << GetErrorString() << std::endl;
        return;
    }
    std::cout << "Loaded model with " << GetTreeCount(modelHandle) << " trees." << std::endl;
    std::cout << "Model expects " << GetFloatFeaturesCount(modelHandle) << " float features and " << GetCatFeaturesCount(modelHandle) << " categorical features" << std::endl;
    const std::string paramsKey = "params";
    if (CheckModelMetadataHasKey(modelHandle, paramsKey.c_str(), paramsKey.size())) {
        size_t paramsStringLength = GetModelInfoValueSize(modelHandle, paramsKey.c_str(), paramsKey.size());
        std::string params(GetModelInfoValue(modelHandle, paramsKey.c_str(), paramsKey.size()), paramsStringLength);
        std::cout << "Applying model trained with params: " << params << std::endl;
    }
    const size_t docCount = 3;
    const size_t floatFeaturesCount = 6;
    const float floatFeatures[docCount ][floatFeaturesCount ] = {
        {28.0, 120135.0, 11.0, 0.0, 0.0, 40.0},
        {49.0, 57665.0, 13.0, 0.0, 0.0, 40.0},
        {34.0, 355700.0, 9.0, 0.0, 0.0, 20.0}
    };
    const float* floatFeaturesPtrs[docCount] = {
        floatFeatures[0],
        floatFeatures[1],
        floatFeatures[2]
    };
    const size_t catFeaturesCount = 8;
    const char* catFeatures[docCount][8] = {
        {"Private", "Assoc-voc", "Never-married", "Sales", "Not-in-family", "White", "Female", "United-States"},
        {"?", "Bachelors", "Divorced", "?", "Own-child", "White", "Female", "United-States"},
        {"State-gov", "HS-grad", "Separated", "Adm-clerical", "Unmarried", "White", "Female", "United-States"}
    };
    const char** catFeaturesPtrs[docCount] = {
        catFeatures[0],
        catFeatures[1],
        catFeatures[2]
    };
    double result[3] = { 0 };
    if (!CalcModelPrediction(
        modelHandle,
        docCount,
        floatFeaturesPtrs, floatFeaturesCount,
        catFeaturesPtrs, catFeaturesCount,
        result, docCount)
    ) {
        std::cout << "Prediction failed: " << GetErrorString() << std::endl;
        return;
    }
    std::cout << "Results: ";
    for (size_t i = 0; i < 3; ++i) {
        std::cout << result[i] << " ";
    }
    std::cout << std::endl;
    /* Sometimes you need to evaluate model on single object.
       We provide special method for this case which is prettier and is little faster than calling batch evaluation for single object
    */
    double singleResult = 0.0;
    if (!CalcModelPredictionSingle(
        modelHandle,
        floatFeatures[0], floatFeaturesCount,
        catFeatures[0], catFeaturesCount,
        &singleResult, 1)
    ) {
        std::cout << "Single prediction failed: " << GetErrorString() << std::endl;
        return;
    }
    std::cout << "Single prediction: " << singleResult << std::endl;
}
```
### C++ wrapper API
We also provide simple but useful C++ wrapper for C API interface: [wrapped_calcer.h](https://github.com/catboost/catboost/tree/master/catboost/libs/model_interface/wrapped_calcer.h)
There is a sample CMake project in [catboost/libs/model_interface/cmake_example/CMakeLists.txt](https://github.com/catboost/catboost/tree/master/catboost/libs/model_interface/cmake_example/CMakeLists.txt)
Using this wrapper is as simple as
```cpp
#include <catboost_model/wrapped_calcer.h>
#include <iostream>
#include <vector>
#include <string>

void cpp_catboost_wrapper_demo() {
    ModelCalcerWrapper calcer("adult_500.cbm"); // load model from file
    std::cout << "Loaded model with " << calcer.GetTreeCount() << " trees, " << calcer.GetFloatFeaturesCount() << " float features and " << calcer.GetCatFeaturesCount() << " categorical features" << std::endl;
    // Access model metadata fields
    if (calcer.CheckMetadataHasKey("params")) {
        std::cout << "Model trained with params: " << calcer.GetMetadataKeyValue("params") << std::endl;
    }
    // Batch model evaluation
    std::vector<std::vector<std::string>> catFeatures = {
        {"Private", "Assoc-voc", "Never-married", "Sales", "Not-in-family", "White", "Female", "United-States"},
        {"?", "Bachelors", "Divorced", "?", "Own-child", "White", "Female", "United-States"},
        {"State-gov", "HS-grad", "Separated", "Adm-clerical", "Unmarried", "White", "Female", "United-States"}
    };
    std::vector<std::vector<float>> floatFeatures = {
        {28.0, 120135.0, 11.0, 0.0, 0.0, 40.0},
        {49.0, 57665.0, 13.0, 0.0, 0.0, 40.0},
        {34.0, 355700.0, 9.0, 0.0, 0.0, 20.0}
    };
    std::vector<double> results = calcer.Calc(floatFeatures, catFeatures);
    std::cout << "Results: ";
    for (const auto result : results) {
        std::cout << result << " ";
    }
    std::cout << std::endl;
    // Single object model evaluation
    std::cout << "Single result: " << calcer.Calc(floatFeatures[0], catFeatures[0]) << std::endl;
}
```
ModelCalcerWrapper also has a constructor from a memory buffer.

## C++ header-only evaluator
**Not recommended to use**. This variant is the simplest way to integrate model evaluation code into your C++ application, but have lots limitations:
* Only valid for models with float features only.
* Depends on Flatbuffers library - you have to install and build `flatc` toolkit manually or have it integrated in your build system.
We implemented this evaluator to allow simple catboost code integration in some Android projects and to simplify integration of Catboost models in CERN experiments.
Look at [standalone_evaluator/CMakeLists.txt](https://github.com/catboost/catboost/tree/master/catboost/libs/standalone_evaluator/CMakeLists.txt) and
[standalone_evaluator/example.cpp](https://github.com/catboost/catboost/tree/master/catboost/libs/standalone_evaluator/example.cpp).
Code snippet:
```cpp
NCatboostStandalone::TOwningEvaluator evaluator("model.cbm");
auto modelFloatFeatureCount = (size_t)evaluator.GetFloatFeatureCount();
std::cout << "Model uses: " << modelFloatFeatureCount << " float features" << std::endl;
std::vector<float> features(modelFloatFeatureCount);
std::cout << evaluator.Apply(features, NCatboostStandalone::EPredictionType::RawValue) << std::endl;
```
