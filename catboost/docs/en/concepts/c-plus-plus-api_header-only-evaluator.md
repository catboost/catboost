# Standalone evaluator

Allows to integrate the {{ product }} code into Android projects and simplifies the integration of {{ product }} in the CERN experiments.

This method of using a trained model is not recommended due to several limitations:
- Only models with float features are supported.
- Dependency from the FlatBuffers library. The `flatc` toolkit must either be built manually or integrated into your build system.

Refer to the [CMake project](https://github.com/catboost/catboost/blob/master/catboost/libs/standalone_evaluator/CMakeLists.txt) and an [example](https://github.com/catboost/catboost/blob/master/catboost/libs/standalone_evaluator/example.cpp) in the {{ product }} repository for more details.

A code snippet:
```cpp
NCatboostStandalone::TOwningEvaluator evaluator("model.cbm");
auto modelFloatFeatureCount = (size_t)evaluator.GetFloatFeatureCount();
std::cout << "Model uses: " << modelFloatFeatureCount << " float features" << std::endl;
std::vector<float> features(modelFloatFeatureCount);
std::cout << evaluator.Apply(features, NCatboostStandalone::EPredictionType::RawValue) << std::endl;
```
