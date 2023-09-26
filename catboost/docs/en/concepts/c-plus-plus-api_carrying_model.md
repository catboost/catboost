## Converting {{ product }} model into yet another {{ product }} model without specified float features

A {{ product }} model based on oblivious trees can be modified in way to remove float features by replacing them with constants. There are two types of replacing implemented:
- Carry: replace features subset by fixed values. Resulting model produce n-dimensional scores (where n is number of values specified for each feature).
- Uplift: replace features subset by value pair. Resulting model produce one dimensional score equal to score difference between carrying model by second and first value from pair.

The functions for carry and uplift are defined in the `catboost/libs/carry/carry.h`. These functions produce {{ product }} models wich can be applyed as usual.

> NOTE: Only float features not used in any CTR are supported


### Carry

Given a model `model` and a grid `factorValues` of feature values, CarryModelByFeatureIndex builds a model which takes samples without features `factorFeatureIndexes`, and returns a vector of predictions of `model` with features `factorFeatureIndexes` set to values `factorValues`.

You may also specify the features for carry by flat index, name, and position using other carry functions.

> NOTE: Model size increases proportional to row count in grid `factorValues`

```cpp
TFullModel CarryModelByFeatureIndex(const TFullModel& model, const TVector<int>& factorFeatureIndexes, const TVector<TVector<double>>& factorsValues);
TFullModel CarryModelByFlatIndex(const TFullModel& model, const TVector<int>& factorFlatIndexes, const TVector<TVector<double>>& factorsValues);
TFullModel CarryModelByName(const TFullModel& model, const TVector<TString>& factorNames, const TVector<TVector<double>>& factorsValues);
TFullModel CarryModel(const TFullModel& model, const TVector<TFeaturePosition>& factors, const TVector<TVector<double>>& factorValues);
```

### Uplift

Given a model `model` and two features samples `baseValues` and `nextValues` of values, `UpliftModelByFeatureIndex` builds a model which takes samples without features `factors`, and returns the difference of predictions of `model` with features factors set to `baseValues` and `nextValues`.

You may also specify the features for uplift by flat index, name, and position using other uplift functions.

```cpp
TFullModel UpliftModelByFeatureIndex(const TFullModel& model, const TVector<int>& factors, const TVector<double>& baseValues, const TVector<double>& nextValues);
TFullModel UpliftModelByFlatIndex(const TFullModel& model, const TVector<int>& factors, const TVector<double>& baseValues, const TVector<double>& nextValues);
TFullModel UpliftModelByName(const TFullModel& model, const TVector<TString>& factors, const TVector<double>& baseValues, const TVector<double>& nextValues);
TFullModel UpliftModel(const TFullModel& model, const TVector<TFeaturePosition>& factors, const TVector<double>& baseValues, const TVector<double>& nextValues);
```
