Export of CatBoost model as standalone C++ code
===============================================

Catboost model could be saved as standalone C++ code. This can ease an integration of a generated model into an application built from C++ sources, simplify porting the model to an architecture not direcly supported by CatBoost (eq. ARM), or allow manual exploration and editing of the model parameters by advanced users.

The exported model code contains complete data for the current trained model and *apply_catboost_model()* function which applies the model to a given dataset. The only current dependency for the code is [CityHash library](https://github.com/google/cityhash/tree/00b9287e8c1255b5922ef90e304d5287361b2c2a) (NOTE: The exact revision under the link is required).


### Exporting from Catboost application via command line interface:

```bash
catboost fit --model-format CPP <other_fit_parameters>
```

By default model is saved into *model.cpp* file. One could alter the output name using *-m* key. If there is more that one model-format specified, then the *.cpp* extention will be added to the name provided after *-m* key.


### Exporting from Catboost python library interface:

```python
model = CatBoost(<train_params>)
model.fit(train_pool)
model.save_model(OUTPUT_CPP_MODEL_PATH, format="CPP")
```


## Models trained with only Float features

If the model was trained using only numerical features (no cat features), then the application function in generated code will have the following interface:

```cpp
double ApplyCatboostModel(const std::vector<float>& features);
```


### Parameters

| parameter | description                                      |
|-----------|--------------------------------------------------|
| features  | features of a single document to make prediction |


### Return value

Prediction of the model for the document with given features.

The result is equivalent to the code below except it won't require linking of libcatboostmodel.<so|dll|dylib>.

```cpp
#include <catboost/libs/model_interface/wrapped_calcer.h>
double ApplyCatboostModel(const std::vector<float>& features) {
    ModelCalcerWrapper calcer("model.cbm");
    return calcer.Calc(features, {});
}
```

### Compiler requirements

C++11 support of non-static data member initializers and extended initializer lists


## Models trained with Categorical features

If the model was trained with categorical features present, then the application function in output code will be generated with the following interface:

```cpp
double ApplyCatboostModel(const std::vector<float>& floatFeatures, const std::vector<std::string>& catFeatures);
```


### Parameters

| parameter     | description                               |
|---------------|-------------------------------------------|
| floatFeatures | numerical features of a single document   |
| catFeatures   | categorical features of a single document |

NOTE: You need to pass float and categorical features separately in the same order they appeared in the train dataset. For example if you had features f1,f2,f3,f4, where f2 and f4 were considered categorical, you need to pass here floatFeatures = {f1, f3}, catFeatures = {f2, f4}.


### Return value

Prediction of the model for the document with given features.

The result is equivalent to the code below except it won't require linking of libcatboostmodel.<so|dll|dylib>.

```cpp
#include <catboost/libs/model_interface/wrapped_calcer.h>
double ApplyCatboostModel(const std::vector<float>& floatFeatures, const std::vector<std::string>& catFeatures) {
    ModelCalcerWrapper calcer("model.cbm");
    return calcer.Calc(floatFeatures, catFeatures);
}
```


### Compiler requiremens

C++14 compiler with aggregate member initialization support. Tested compilers: g++ 5(5.4.1 20160904), clang++ 3.8.


## Current limitations

- MultiClassification models are not supported.
- applyCatboostModel() function has reference implementation and may lack of performance comparing to native applicator of CatBoost, especially on large models and multiple of documents.


## Troubleshooting

Q: Generated model results differ from native model when categorical features present
A: Please check that CityHash version 1 is used. Exact required revision of [C++ Google CityHash library](https://github.com/Amper/cityhash/tree/4f02fe0ba78d4a6d1735950a9c25809b11786a56%29). There is also proper CityHash implementation in [Catboost repository](https://github.com/catboost/catboost/blob/master/util/digest/city.h). This is due other versions of CityHash may produce different hash code for the same string.


