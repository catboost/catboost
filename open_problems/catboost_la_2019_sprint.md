# Tasks for sprint

1. `Better parameter checks:`
 if leaf_estimation_iterations:5 with RMSE, there should be warning and 1 iteration
2. `Skip invalid parameter configurations in grid_search and randomized_search methods:`
 The python code of running parameter search should check if configuration is valid. If it is not valid it should be skipped and a warning msg should be printed. In case of randomized_search, where n_iter is number of checked configurations, invalid configurations should not be count as checked ones.
3. `Add model.class_count_ property to CatBoostClassifier class:`
 It should return `len(model.class_names_)`
4.  Add `feature_names_`, `cat_feature_names_`, `num_feature_names_`, `cat_feature_indices_` properties to CatBoost* classes.
5. `Implement CatBoostRanker class:`
 Currently we only have CatBoostRegressor and CatBoostClassifier. It would be nice to implement a class for ranking also. The default loss function in this case will be YetiRank.
6. `Implement a ColumnDescription class in Python:`
 This class could be used instead of cd file https://catboost.ai/docs/concepts/input-data_column-descfile.html when creating Pool from filez. The class should have init function, methods load and save, and Pool init method should be able to use object of this class instead of cd file during initialization.
7. `Tutorial on poisson regression using monotonic1 dataset:` 
Jupyter notebook should give text explanation of what is the task, examples when it might appear and how it is solved 
8. `treat_object_as_categorical:`
Currently you have to pass cat_features to CatBoost* init function or to fit function. Many people ask for automatic detection of categorical features. This flag would solve the problem. It is suggested to add the flag to Pool init function, CatBoost* init functions and to fit function the same way cat_features parameter is added.
Tests for all these cases must be provided.
9. `allow_float_categories:`
 Categorical features are treated in the following way. We first convert them to strings, then calculate hash from the string, then use the hash value in the algorithm. For this reason it is only allowed to use data types that can be converted to string in a unique way. Otherwise if you are training from python and applying from C++, you might get different results because of different string representation. But if you are only working from python, then it can be safe to use float numbers if user has explicitly confirmed that this is what the user wants. 
10 `Python cv  should check loss_function tobe set`
In python cv request `loss_function` to be set
   Currently if no `loss_function` is passed, then RMSE is used by default.
   This might be misleading in the following case.
   A user creates CatBoostClassifier and calls `get_params()` from a not trained model. The resulting parameters don't contain the `loss_function` parameter, because the default `loss_function` for CatBoostClassifier depends on number of classes in train dataset. If there are 2 classes, it is Logloss, if there are more than 2 classes, it is MultiClass.
   These parameters are passed to cv function. And it trains an RMSE model, because it is the default loss.
   This is not expected behaviour. So it is better to check that the loss is present among parameters passed to cv method.
   
# Quick start guide

Repo on GitHub: https://github.com/catboost/catboost
Clone repo via git

## Build

### Python-package
cd SOURCE_ROOT/catboost/python-package/catboost
../../../ya make -r  -DUSE_CUDA=false  -DUSE_ARCADIA_PYTHON=no -DPYTHON_CONFIG=python-config-path -DOS_SDK=local
For testing add SOURCE_ROOT/catboost/python-package for python-path

Change USE_CUDA=true to build with CUDA

### command-line binary
cd catboost/app
../../../ya make -r  -DUSE_CUDA=false 

## ya make

CatBoost uses own build system, ya, to build project.
It could be used to generate projects for c++ IDE (Clion or Qt),  ./ya ide --help for more information

## Repo structure:

util/ — system libraries, c++
library/  — general libraries
contrib/  — external libraries
catboost/ — project source root
catboost/python-package — python package sources


Important sources:

catboost/libs/model/model.h
catboost/python_package/catboost/_catboost.pyx
catboost/python_package/catboost/core.py
catboost/R-package/src/catboostr.cpp
catboost/private/libs/options

catboost/libs/train_lib/train_model.cpp
catboost/private/libs/algo/train.cpp
catboost/private/libs/algo/approx_calcer.cpp
catboost/private/libs/algo/score_calcer.cpp
catboost/private/libs/algo/greedy_tensor_search.cpp


## C++ for CatBoost repo
CatBoost is based on Yandex c++ code based, we've been using c++ since 1998, so we are using our own boost-like libraries from util/*

### Smart Pointers:

util/ptr.h

THolder<T> —  std::unique_ptr<T>
TIntrusivePtr<T> —  intrusive pointer, objects should be descent of TRefCounted

### Input/Output:

util/stream

IInputStream/IOutputStream — base class
Cin ≈ std::cin
Cout ≈ std::cout
Cerr ≈ std::err
End ≈ std::err

util/stream/file

TInputFile / TOutputFile
util/stream/fs.h

NFs::Exists
NFs::Copy

### Containers:

util/generic/vector.h

TVector<T> instead of std::vector<T>

util/generic/hash.h

THashMap<T>
THashSet<T>

util/generic/set.h

TSet<T>

Util/generic/map.h

TMap<T>

util/generic/array_ref.h

TArrayRef<T>, TConstArrayRef<T> ≈ std::span<T> from c++20

util/generic/strbuf.h

TStringBuf ≈ std::string_view

Util/generic/string.h

TString  — Copy on Write string of chars
TUtf16String — CoW wchar16 string

util/string/cast.h

TString ToString<T> is analogue to std:to_string
T FromString<T>
bool TryFromString<T>(, T* value)


### Exceptions:

catboost/libs/helpers:

ythrow TCatBoostException() << message - how to throw exception
CB_ENSURE(condition, message). — macro for ensure
Y_ASSERT(condition) — macro for assets

### Other
util/generic/maybe.h
TMaybe<T> ≈ std::optional

util/generic/variant.h
TVariant<T> ≈ std::variant:

## Code Style:

### C++
https://github.com/catboost/catboost/blob/master/CPP_STYLE_GUIDE.md for whole repo
https://github.com/catboost/catboost/blob/master/catboost_command_style_guide_extension.md is CatBoost-specific extension

### Python
PEP 8 

## Pull Requests

Before submitting a pull request, please do the following steps:

Read instructions for contributors.
Run ya make in catboost folder to make sure the code builds.
Add tests that test your change.
Run tests using ya make -t -A command.
If you haven't already, complete the CLA. (look for more info here https://github.com/catboost/catboost/blob/master/CONTRIBUTING.md)
