If you want to use a trained model, we provide 3 different ways to do this: Python API, dynamic C library with C++ wrapper and C++ header-only evaluator.
If you use `ya.make` build system, the most convenient interface for model evaluation is ```TFullModel``` class, defined in [model.h](https://github.com/catboost/catboost/tree/master/catboost/libs/model/model.h).

## Python API
The simplest but not the fastest way to evaluate model predictions, just use ```model.predict```.
Small example: 
```python
model = CatBoostClassifier()
model.load_model('catboost_model.dump')
model.predict(object_features)
```
If your data is bigger than 1000 objects we recommend you to use ```catboost.TPool``` class to get full performance.

Read python API documentation [here](https://tech.yandex.com/catboost/doc/dg/concepts/python-usages-examples-docpage/) for more information.


## Dynamic library ```libcatboostmodel.(so|dll|dylib)```
This is the fastest way to evaluate model. Library interface is C API but we provide simple C++ header-only wrapper.
but you can easily use it's C API interface from any language you like.
The code of library and small CMake usage example can be found in [model_interface](https://github.com/catboost/catboost/tree/master/catboost/libs/model_interface)
You can build this library with command
```bash
ya make -r catboost/libs/model_interface/ 
```
You can load your catboost model from file or initialize it from buffer in memory.

### C API
We provide really simple model api in catboost/libs/model_interface. Just link ```libcatboostmodel.(so|dll|dylib)``` and use methods from model_calcer_wrapper.h file.
Just look at doxygen-style documentation in [model_calcer_wrapper.h](https://github.com/catboost/catboost/tree/master/catboost/libs/model_interface/model_calcer_wrapper.h)
Sample C code (without includes): 
```cpp
float floatFeatures[100];
char* catFeatures[2] = {"1", "2"};
double result[1];
ModelCalcerHandle modelHandle;
modelHandle = ModelCalcerCreate();
if (!LoadFullModelFromFile(modelHandle, "model.cbm")) {
    printf("LoadFullModelFromFile error message: %s\n", GetErrorString());
}
if (!CalcModelPrediction(
        modelHandle,
        1,
        &floatFeatures, 100,
        &catFeatures, 2,
        &result, 1
    )) {
    printf("CalcModelPrediction error message: %s\n", GetErrorString());
}
ModelCalcerDelete(modelHandle);
```
### C++ wrapper API
We also provide simple but useful C++ wrapper for C API interface: [wrapped_calcer.h](https://github.com/catboost/catboost/tree/master/catboost/libs/model_interface/wrapped_calcer.h)
There is a sample CMake project in [catboost/libs/model_interface/cmake_example/CMakeLists.txt](https://github.com/catboost/catboost/tree/master/catboost/libs/model_interface/cmake_example/CMakeLists.txt)
Using this wrapper is as simple as 
```cpp
ModelCalcerWrapper calcer("model.cbm");
std::vector<float> floatFeatures(100);
std::vector<std::string> catFeatures = {"one", "two", "three"};
std::cout << calcer.Calc(floatFeatures, catFeatures) << std::endl;
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
