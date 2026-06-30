#pragma once

#include <catboost/libs/model/model.h>

#include <util/generic/string.h>
#include <util/generic/vector.h>

// creates a new model equivalent to the original one with substitutions of factors subset
TFullModel CarryModelByFeatureIndex(const TFullModel& model, const TVector<int>& factorFeatureIndexes, const TVector<TVector<double>>& factorsValues);
TFullModel CarryModelByFlatIndex(const TFullModel& model, const TVector<int>& factorFlatIndexes, const TVector<TVector<double>>& factorsValues);
TFullModel CarryModelByName(const TFullModel& model, const TVector<TString>& factorNames, const TVector<TVector<double>>& factorsValues);
TFullModel CarryModel(const TFullModel& model, const TVector<TFeaturePosition>& factors, const TVector<TVector<double>>& factorValues);

// creates a new model equivalent to difference of the original one with substitution of two factors subset
TFullModel UpliftModelByFeatureIndex(const TFullModel& model, const TVector<int>& factors, const TVector<double>& baseValues, const TVector<double>& nextValues);
TFullModel UpliftModelByFlatIndex(const TFullModel& model, const TVector<int>& factors, const TVector<double>& baseValues, const TVector<double>& nextValues);
TFullModel UpliftModelByName(const TFullModel& model, const TVector<TString>& factors, const TVector<double>& baseValues, const TVector<double>& nextValues);
TFullModel UpliftModel(const TFullModel& model, const TVector<TFeaturePosition>& factors, const TVector<double>& baseValues, const TVector<double>& nextValues);
