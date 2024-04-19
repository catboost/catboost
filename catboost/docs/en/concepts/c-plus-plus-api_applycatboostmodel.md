# C++

## {{ dl--purpose }}

Apply the model in C++ format. The method is available within the output C++ file with the model description.

{% note info %}

- {% include [reusage-common-phrases-cplusplus_apply_catboost_model__performance](../_includes/work_src/reusage-common-phrases/cplusplus_apply_catboost_model__performance.md) %}

{% endnote %}


## {{ dl--invoke-format }}

{% include [reusage-common-phrases-for-datasets-that-contain-only-numeric-features](../_includes/work_src/reusage-common-phrases/for-datasets-that-contain-only-numeric-features.md) %}


```cpp
double {{ method_name__apply_cplusplus_model }}(const std::vector<float>& features);
```

{% include [reusage-common-phrases-for-datasets-that-contain-both-numerical-and-categorical-features](../_includes/work_src/reusage-common-phrases/for-datasets-that-contain-both-numerical-and-categorical-features.md) %}


```cpp
double {{ method_name__apply_cplusplus_model }}(const std::vector<float>& floatFeatures, const std::vector<std::string>& catFeatures);
```

## {{ dl--parameters }}

### features (floatFeatures)


{% include [exported-models-float-features-desc](../_includes/work_src/reusage-common-phrases/float-features-desc.md) %}

Possible types: float


### catFeatures


{% include [exported-models-categorical-features-list](../_includes/work_src/reusage-common-phrases/categorical-features-list.md) %}

Possible types: string




{% note info %}

{% include [exported-models-numerical-and-categorical-features-start](../_includes/work_src/reusage-common-phrases/numerical-and-categorical-features-start.md) %}


```python
{{ method_name__apply_cplusplus_model }}({f1, f3}, {f2, f4})
```

{% endnote %}


## {{ dl--output-format }}

Prediction of the model for the object with given features.

The result is identical to the code below but does not require the library linking (`libcatboostmodel.<so|dll|dylib>` for Linux/macOS or `libcatboostmodel.dll` for Windows):
- {% include [reusage-common-phrases-for-datasets-that-contain-only-numeric-features](../_includes/work_src/reusage-common-phrases/for-datasets-that-contain-only-numeric-features.md) %}

    ```cpp
    #include <catboost/libs/model_interface/wrapped_calcer.h>
    double ApplyCatboostModel(const std::vector<float>& features) {
    ModelCalcerWrapper calcer("model.cbm");
    return calcer.Calc(features, {});
    }
    ```

- {% include [reusage-common-phrases-for-datasets-that-contain-both-numerical-and-categorical-features](../_includes/work_src/reusage-common-phrases/for-datasets-that-contain-both-numerical-and-categorical-features.md) %}

    ```cpp
    #include <catboost/libs/model_interface/wrapped_calcer.h>
    double ApplyCatboostModel(const std::vector<float>& floatFeatures, const std::vector<std::string>& catFeatures) {
    ModelCalcerWrapper calcer("model.cbm");
    return calcer.Calc(floatFeatures, catFeatures);
    }
    ```

## Compilers

- {% include [reusage-common-phrases-for-datasets-that-contain-only-numeric-features](../_includes/work_src/reusage-common-phrases/for-datasets-that-contain-only-numeric-features.md) %}

    C++11 with support of non-static data member initializers and extended initializer lists.

- {% include [reusage-common-phrases-for-datasets-that-contain-both-numerical-and-categorical-features](../_includes/work_src/reusage-common-phrases/for-datasets-that-contain-both-numerical-and-categorical-features.md) %}

    C++14 compiler with aggregate member initialization support. Tested with the following compilers:
    - Clang++ 3.8
    - g++ 5.4.1 20160904
