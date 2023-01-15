#include "shap_interaction_values.h"

#include "shap_values.h"
#include "shap_prepared_trees.h"
#include "util.h"

#include <catboost/private/libs/algo/features_data_helpers.h>
#include <catboost/private/libs/algo/index_calcer.h>
#include <catboost/libs/model/cpu/quantization.h>


using namespace NCB;

static void ValidateFeatureIndex(int featureCount, int featureIdx) {
    CB_ENSURE(featureIdx < featureCount, "Feature index " << featureIdx << " exceeds feature count " << featureCount);
}

void ValidateFeaturePair(int flatFeatureCount, std::pair<int, int> featurePair) {
    ValidateFeatureIndex(flatFeatureCount, featurePair.first);
    ValidateFeatureIndex(flatFeatureCount, featurePair.second);
}

void ValidateFeatureInteractionParams(
    const EFstrType fstrType,
    const TFullModel& model,
    const NCB::TDataProviderPtr dataset,
    ECalcTypeShapValues calcType
) {
    CB_ENSURE(model.GetTreeCount(), "Model is not trained");

    CB_ENSURE_INTERNAL(
        fstrType == EFstrType::ShapInteractionValues,
        ToString<EFstrType>(fstrType) + " is not suitable for calc shap interaction values"
    );

    CB_ENSURE(dataset, "Dataset is not provided");

    CB_ENSURE(
        calcType != ECalcTypeShapValues::Independent,
        "SHAP Interaction Values can't calculate in mode " + ToString<ECalcTypeShapValues>(calcType)
    );
}

using TInteractionValuesSubset = THashMap<std::pair<size_t, size_t>, TVector<TVector<double>>>;
using TInteractionValuesFull = TVector<TVector<TVector<TVector<double>>>>;

template <typename TStorageType>
TVector<TVector<double>>& GetDocumentsByClasses(TStorageType* rank4storage, size_t idx1, size_t idx2);

template <typename TStorageType>
const TVector<TVector<double>>& GetDocumentsByClasses(const TStorageType& rank4storage, size_t idx1, size_t idx2);

template <typename TStorageType>
void AllocateStorage(
    const TVector<size_t>& firstIndices,
    const TVector<size_t>& secondIndices,
    const size_t dim1,
    const size_t dim2,
    NPar::ILocalExecutor* localExecutor,
    TStorageType* shapInteractionValuesInternal
);

template <>
TVector<TVector<double>>& GetDocumentsByClasses(TInteractionValuesSubset* rank4storage, size_t idx1, size_t idx2) {
    return rank4storage->at(std::make_pair(idx1, idx2));
}

template <>
TVector<TVector<double>>& GetDocumentsByClasses(TInteractionValuesFull* rank4storage, size_t idx1, size_t idx2) {
    return rank4storage->at(idx1)[idx2];
}

template <>
const TVector<TVector<double>>& GetDocumentsByClasses(const TInteractionValuesSubset& rank4storage, size_t idx1, size_t idx2) {
    return rank4storage.at(std::make_pair(idx1, idx2));
}

template <>
const TVector<TVector<double>>& GetDocumentsByClasses(const TInteractionValuesFull& rank4storage, size_t idx1, size_t idx2) {
    return rank4storage[idx1][idx2];
}

template <>
void AllocateStorage(
    const TVector<size_t>& firstIndices,
    const TVector<size_t>& secondIndices,
    const size_t dim1,
    const size_t dim2,
    NPar::ILocalExecutor* localExecutor,
    TInteractionValuesSubset* map
) {
    for (auto idx1 : firstIndices) {
        for (auto idx2 : secondIndices) {
            auto& value = (*map)[std::make_pair(idx1, idx2)];
            value.resize(dim1);
            ParallelFill(
                TVector<double>(dim2, 0.0),
                /*blockSize*/ Nothing(),
                localExecutor,
                MakeArrayRef(value)
            );
        }
    }
}

static inline void Allocate4DimensionalVector(
    const size_t dim1,
    const size_t dim2,
    const size_t dim3,
    const size_t dim4,
    NPar::ILocalExecutor* localExecutor,
    TInteractionValuesFull* newVector
) {
    newVector->resize(dim1);
    for (size_t i = 0; i < dim1; ++i) {
        (*newVector)[i].resize(dim2);
        for (size_t j = 0; j < dim2; ++j) {
            (*newVector)[i][j].resize(dim3);
            ParallelFill(
                TVector<double>(dim4, 0.0),
                /*blockSize*/ Nothing(),
                localExecutor,
                MakeArrayRef((*newVector)[i][j])
            );
        }
    }
}

template <>
void AllocateStorage(
    const TVector<size_t>& firstIndices,
    const TVector<size_t>& secondIndices,
    const size_t dim1,
    const size_t dim2,
    NPar::ILocalExecutor* localExecutor,
    TInteractionValuesFull* vector
) {
    Allocate4DimensionalVector(firstIndices.size(), secondIndices.size(), dim1, dim2, localExecutor, vector);
}

static inline void AddValuesAllDocuments(
    const TVector<TVector<double>>& valuesBetweenClasses,
    double coefficient,
    TVector<TVector<double>>* valuesBetweenFeatures
) {
    const size_t documentCount = valuesBetweenClasses[0].size();
    for (int dimension : xrange(valuesBetweenClasses.size())) {
        TConstArrayRef<double> valuesBetweenClassesRef = MakeConstArrayRef(valuesBetweenClasses.at(dimension));
        TArrayRef<double> valueBetweenFeaturesRef = MakeArrayRef((*valuesBetweenFeatures)[dimension]);
        for (size_t documentIdx : xrange(documentCount)) {
            double effect = valuesBetweenClassesRef[documentIdx] * coefficient;
            valueBetweenFeaturesRef[documentIdx] += effect;
        }
    }
}

template <typename TStorageType>
static void UnpackInternalShapInteractionValues(
    const TStorageType& shapInteractionValuesInternal,
    const TVector<TVector<int>>& combinationClassFeatures,
    const TVector<size_t>& classIndicesFirst,
    const TVector<size_t>& classIndicesSecond,
    const TVector<double>& rescaleCoefficients,
    TInteractionValuesFull* shapInteractionValues
) {
    for (size_t classIdx1 : classIndicesFirst) {
        for (size_t classIdx2 : classIndicesSecond) {
            const auto& documentsByClasses = GetDocumentsByClasses(shapInteractionValuesInternal, classIdx1, classIdx2);
            if (classIdx1 == classIdx2) {
                // unpack main effect
                double coefficientForMainEffect = 1.0 / rescaleCoefficients[classIdx1];
                for (int featureIdx : combinationClassFeatures[classIdx1]) {
                    AddValuesAllDocuments(
                        documentsByClasses,
                        coefficientForMainEffect,
                        &shapInteractionValues->at(featureIdx)[featureIdx]
                    );
                }
                continue;
            }
            // unpack interaction effect
            double coefficientForInteractionEffect = 1.0 / (rescaleCoefficients[classIdx1] * rescaleCoefficients[classIdx2]);
            for (int featureIdx1 : combinationClassFeatures[classIdx1]) {
                for (int featureIdx2 : combinationClassFeatures[classIdx2]) {
                    if (featureIdx1 == featureIdx2) {
                        continue;
                    }
                    AddValuesAllDocuments(
                        documentsByClasses,
                        coefficientForInteractionEffect,
                        &shapInteractionValues->at(featureIdx1)[featureIdx2]
                    );
                }
            }
        }
    }
}

static inline TVector<size_t> IntersectClasses(
    const TVector<size_t>& classIndicesFirst,
    const TVector<size_t>& classIndicesSecond
) {
    TVector<size_t> sameClassIndices;
    std::set_intersection(
        classIndicesFirst.begin(),
        classIndicesFirst.end(),
        classIndicesSecond.begin(),
        classIndicesSecond.end(),
        std::back_inserter(sameClassIndices)
    );
    return sameClassIndices;
}

static inline void FillIndices(const TVector<size_t>& indices, TVector<bool>* isIndices) {
    for (size_t idx : indices) {
        isIndices->at(idx) = true;
    }
}

static void ContructClassIndices(
    const TVector<TVector<int>>& combinationClassFeatures,
    const TMaybe<std::pair<int, int>>& pairOfFeatures,
    TVector<size_t>* classIndicesFirst,
    TVector<size_t>* classIndicesSecond
) {
    const size_t classCount = combinationClassFeatures.size();
    if (pairOfFeatures.Defined()) {
        for (size_t classIdx : xrange(classCount)) {
            for (int flatFeatureIdx : combinationClassFeatures[classIdx]) {
                if (flatFeatureIdx == pairOfFeatures->first) {
                    classIndicesFirst->emplace_back(classIdx);
                }
                if (flatFeatureIdx == pairOfFeatures->second) {
                    classIndicesSecond->emplace_back(classIdx);
                }
            }
        }
        if (classIndicesFirst->size() > classIndicesSecond->size()) {
            std::swap(classIndicesFirst, classIndicesSecond);
        }
    } else {
        classIndicesFirst->resize(classCount);
        classIndicesSecond->resize(classCount);
        Iota(classIndicesFirst->begin(), classIndicesFirst->end(), 0);
        Iota(classIndicesSecond->begin(), classIndicesSecond->end(), 0);
    }
}

static inline double GetInteractionEffect(double contribOn, double contribOff) {
    return (contribOn - contribOff) / 2.0;
}

static inline void AddInteractionEffectAllDocumentsForPairClasses(
    const TVector<TVector<double>>& contribsOnByClass,
    const TVector<TVector<double>>& contribsOffByClass,
    bool isSameClasses,
    bool isSecondFeature,
    bool isCalcMainEffect,
    TVector<TVector<double>>* valuesBetweenDifferentClasses,
    TVector<TVector<double>>* valuesBetweenSameClasses
) {
    if (isSameClasses) {
        return;
    }
    for (size_t dimension : xrange(contribsOnByClass.size())) {
        TConstArrayRef<double> contribsOnByClassRef = MakeConstArrayRef(contribsOnByClass.at(dimension));
        TConstArrayRef<double> contribsOffByClassRef = MakeConstArrayRef(contribsOffByClass.at(dimension));
        TArrayRef<double> valuesBetweenDifferentClassesRef = MakeArrayRef((*valuesBetweenDifferentClasses)[dimension]);
        TArrayRef<double> valuesBetweenSameClassesRef = MakeArrayRef((*valuesBetweenSameClasses)[dimension]);
        for (size_t documentIdx : xrange(contribsOnByClass[dimension].size())) {
            double interactionEffect = GetInteractionEffect(contribsOnByClassRef[documentIdx], contribsOffByClassRef[documentIdx]);
            if (isSecondFeature) {
                valuesBetweenDifferentClassesRef[documentIdx] = interactionEffect;
            }
            if (isCalcMainEffect) {
                valuesBetweenSameClassesRef[documentIdx] -= interactionEffect;
            }
        }
    }
}

template <typename TStorageType>
static void CalcInternalShapInteractionValuesMulti(
    const TFullModel& model,
    const size_t documentCount,
    const TVector<TIntrusivePtr<NModelEvaluation::IQuantizedData>>& binarizedFeatures,
    const TVector<TVector<NModelEvaluation::TCalcerIndexType>>& indexes,
    const TVector<size_t>& classIndicesFirst,
    const TVector<size_t>& classIndicesSecond,
    int logPeriod,
    NPar::ILocalExecutor* localExecutor,
    TShapPreparedTrees* preparedTrees,
    TStorageType* shapInteractionValuesInternal,
    ECalcTypeShapValues calcType
) {
    if (classIndicesFirst.empty() || classIndicesSecond.empty()) {
        return;
    }
    const auto& combinationClassFeatures = preparedTrees->CombinationClassFeatures;
    const size_t classCount = combinationClassFeatures.size();
    const auto& sameClassIndices = IntersectClasses(classIndicesFirst, classIndicesSecond);
    // contruct vector of indices with same class and second feature indices
    TVector<bool> isSameClassIdx(classCount, false);
    TVector<bool> isSecondFeatureIdx(classCount, false);
    FillIndices(sameClassIndices, &isSameClassIdx);
    FillIndices(classIndicesSecond, &isSecondFeatureIdx);
    // calc shap values
    const auto& shapValuesInternal = CalcShapValueWithQuantizedData(
        model,
        binarizedFeatures,
        indexes,
        /*fixedFeatureParams*/ Nothing(),
        documentCount,
        logPeriod,
        preparedTrees,
        localExecutor,
        calcType
    );
    const int approxDimension = model.GetDimensionsCount();
    TVector<TVector<double>> emptyVector(approxDimension);
    // Φ(i,i) = ϕ(i) − sum(Φ(i,j)) i.e reducing path of sum
    // because in the first to add ф(i)
    for (size_t classIdx : xrange(sameClassIndices.size())) {
        const auto sameClassIdx = sameClassIndices[classIdx];
        auto& valuesBetweenClasses = GetDocumentsByClasses(shapInteractionValuesInternal, sameClassIdx, sameClassIdx);
        AddValuesAllDocuments(
            shapValuesInternal[sameClassIdx],
            /*coefficient*/ 1.0,
            &valuesBetweenClasses
        );
    }
    // calculate shap interaction values
    // Katsushige Fujimoto, Ivan Kojadinovic, and Jean-Luc Marichal. 2006. Axiomatic
    // characterizations of probabilistic and cardinal-probabilistic interaction indices.
    // Games and Economic Behavior 55, 1 (2006), 72–99
    for (size_t classIdx1 : classIndicesFirst) {
        const auto& contribsOn = CalcShapValueWithQuantizedData(
            model,
            binarizedFeatures,
            indexes,
            MakeMaybe<TFixedFeatureParams>(classIdx1, TFixedFeatureParams::EMode::FixedOn),
            documentCount,
            logPeriod,
            preparedTrees,
            localExecutor,
            calcType
        );
        const auto& contribsOff = CalcShapValueWithQuantizedData(
            model,
            binarizedFeatures,
            indexes,
            MakeMaybe<TFixedFeatureParams>(classIdx1, TFixedFeatureParams::EMode::FixedOff),
            documentCount,
            logPeriod,
            preparedTrees,
            localExecutor,
            calcType
        );
        // if needed calculate Ф(i, i) then to calculate all Ф(i, j) where i != j
        // Φ(i,i) = ϕ(i) − sum(Φ(i,j)) i.e reducing path of sum
        if (isSameClassIdx[classIdx1]) {
            for (size_t classIdx2 : xrange(classCount)) {
                AddInteractionEffectAllDocumentsForPairClasses(
                    contribsOn[classIdx2],
                    contribsOff[classIdx2],
                    /*isSameClasses*/ (classIdx2 == classIdx1),
                    isSecondFeatureIdx[classIdx2],
                    /*isCalcMainEffect*/ true,
                    isSecondFeatureIdx[classIdx2] ? &GetDocumentsByClasses(shapInteractionValuesInternal, classIdx1, classIdx2) : &emptyVector,
                    &GetDocumentsByClasses(shapInteractionValuesInternal, classIdx1, classIdx1)
                );
            }
        } else {
            for (size_t classIdx2 : classIndicesSecond) {
                AddInteractionEffectAllDocumentsForPairClasses(
                    contribsOn[classIdx2],
                    contribsOff[classIdx2],
                    /*isSameClasses*/ (classIdx2 == classIdx1),
                    /*isSecondFeature*/ true,
                    /*isCalcMainEffect*/ false,
                    &GetDocumentsByClasses(shapInteractionValuesInternal, classIdx1, classIdx2),
                    /*valuesBetweenSameClasses*/ &emptyVector
                );
            }
        }
    }
}

static void FilterClassFeaturesByPair(
    std::pair<int, int> pairOfFeatures,
    TVector<TVector<int>>* combinationClassFeatures
) {
    // leave only pair of features
    for (auto& classFeatures : *combinationClassFeatures) {
        EraseIf(
            classFeatures,
            [pairOfFeatures] (int feature) {
                return pairOfFeatures.first != feature && pairOfFeatures.second != feature;
            }
        );
    }
    // scalling
    for (auto& classFeatures : *combinationClassFeatures) {
        for (auto& feature : classFeatures) {
            feature = (pairOfFeatures.first != feature);
        }
    }
}

static inline TVector<double> ContructRescaleCoefficients(
    const TVector<TVector<int>>& combinationClassFeatures
) {
    TVector<double> rescaleCoefficients;
    for (const auto& combination : combinationClassFeatures) {
        rescaleCoefficients.emplace_back(combination.size());
    }
    return rescaleCoefficients;
}

static inline void SetBiasValues(const TFullModel& model, TVector<TVector<double>>* values) {
    const size_t documentsCount = (*values)[0].size();
    auto bias = model.GetScaleAndBias().GetBiasRef();
    for (size_t dimension = 0; dimension < values->size(); ++dimension) {
        TArrayRef<double> valuesRef = MakeArrayRef(values->at(dimension));
        for (size_t documentIdx = 0; documentIdx < documentsCount; ++documentIdx) {
            valuesRef[documentIdx] += bias[dimension];
        }
    }
}

static void CalcLeafIndices(
    const TFullModel& model,
    const TDataProvider& dataset,
    TVector<TIntrusivePtr<NModelEvaluation::IQuantizedData>>* binarizedFeatures,
    TVector<TVector<NModelEvaluation::TCalcerIndexType>>* indexes
) {
    const size_t documentCount = dataset.ObjectsGrouping->GetObjectCount();
    THolder<IFeaturesBlockIterator> featuresBlockIterator
        = CreateFeaturesBlockIterator(model, *dataset.ObjectsData, 0, documentCount);
    const TModelTrees& forest = *model.ModelTrees;
    const size_t documentBlockSize = CB_THREAD_LIMIT;
    for (ui32 startIdx = 0; startIdx < documentCount; startIdx += documentBlockSize) {
        NPar::ILocalExecutor::TExecRangeParams blockParams(startIdx, startIdx + Min(documentBlockSize, documentCount - startIdx));
        featuresBlockIterator->NextBlock(blockParams.LastId - blockParams.FirstId);
        binarizedFeatures->emplace_back();
        indexes->emplace_back();
        auto& indexesForBlock = indexes->back();
        indexesForBlock.resize(documentBlockSize * forest.GetTreeCount());
        auto& binarizedFeaturesForBlock = binarizedFeatures->back();
        binarizedFeaturesForBlock = MakeQuantizedFeaturesForEvaluator(model, *featuresBlockIterator, blockParams.FirstId, blockParams.LastId);

        model.GetCurrentEvaluator()->CalcLeafIndexes(
            binarizedFeaturesForBlock.Get(),
            0, forest.GetTreeCount(),
            MakeArrayRef(indexesForBlock.data(), binarizedFeaturesForBlock->GetObjectsCount() * forest.GetTreeCount())
        );
    }
}

template <typename TStorageType>
static void CalcShapInteraction(
    const TFullModel& model,
    const TDataProvider& dataset,
    const TMaybe<std::pair<int, int>>& pairOfFeatures,
    int logPeriod,
    NPar::ILocalExecutor* localExecutor,
    TShapPreparedTrees* preparedTrees,
    TInteractionValuesFull* shapInteractionValues,
    ECalcTypeShapValues calcType
) {
    TVector<TIntrusivePtr<NModelEvaluation::IQuantizedData>> binarizedFeatures;
    TVector<TVector<NModelEvaluation::TCalcerIndexType>> indexes;
    CalcLeafIndices(
        model,
        dataset,
        &binarizedFeatures,
        &indexes
    );
    TVector<size_t> classIndicesFirst;
    TVector<size_t> classIndicesSecond;
    ContructClassIndices(
        preparedTrees->CombinationClassFeatures,
        pairOfFeatures,
        &classIndicesFirst,
        &classIndicesSecond
    );
    const size_t documentCount = dataset.ObjectsGrouping->GetObjectCount();
    const size_t approxDimension = model.GetDimensionsCount();
    TStorageType shapInteractionValuesInternal;
    AllocateStorage(
        classIndicesFirst,
        classIndicesSecond,
        approxDimension,
        documentCount,
        localExecutor,
        &shapInteractionValuesInternal
    );
    CalcInternalShapInteractionValuesMulti(
        model,
        documentCount,
        binarizedFeatures,
        indexes,
        classIndicesFirst,
        classIndicesSecond,
        logPeriod,
        localExecutor,
        preparedTrees,
        &shapInteractionValuesInternal,
        calcType
    );
    int flatFeatureCount = SafeIntegerCast<int>(dataset.MetaInfo.GetFeatureCount());
    int featuresCount = pairOfFeatures ? (pairOfFeatures->first == pairOfFeatures->second ? 1 : 2) : flatFeatureCount;
    Allocate4DimensionalVector(
        featuresCount + 1,
        featuresCount + 1,
        approxDimension,
        documentCount,
        localExecutor,
        shapInteractionValues
    );
    // unpack shap interaction values
    TVector<double> rescaleCoefficients = ContructRescaleCoefficients(preparedTrees->CombinationClassFeatures);
    if (pairOfFeatures) {
        FilterClassFeaturesByPair(*pairOfFeatures, &preparedTrees->CombinationClassFeatures);
    }
    UnpackInternalShapInteractionValues(
        shapInteractionValuesInternal,
        preparedTrees->CombinationClassFeatures,
        classIndicesFirst,
        classIndicesSecond,
        rescaleCoefficients,
        shapInteractionValues
    );
    for (int featureIdx = 0; featureIdx < featuresCount; ++featureIdx) {
        auto& valuesInLastColumn = (*shapInteractionValues)[featuresCount][featureIdx];
        auto& valuesInLastRow = (*shapInteractionValues)[featureIdx][featuresCount];
        SetBiasValues(model, &valuesInLastColumn);
        SetBiasValues(model, &valuesInLastRow);
    }
}

static inline void SetSymmetricValues(TInteractionValuesFull* shapInteractionValues) {
    CB_ENSURE_INTERNAL(
        shapInteractionValues->size() == 3 && (*shapInteractionValues)[0].size() == 3,
        "Shap interaction values must be contain two features and bias"
    );
    const size_t approxDimension = (*shapInteractionValues)[0][0].size();
    const size_t documentCount = (*shapInteractionValues)[0][0][0].size();
    for (size_t dimension : xrange(approxDimension)) {
        for (size_t documentIdx : xrange(documentCount)) {
            (*shapInteractionValues)[1][0][dimension][documentIdx] =
                (*shapInteractionValues)[0][1][dimension][documentIdx];
        }
    }
}

TInteractionValuesFull CalcShapInteractionValuesWithPreparedTrees(
    const TFullModel& model,
    const TDataProvider& dataset,
    const TMaybe<std::pair<int, int>>& pairOfFeatures,
    int logPeriod,
    ECalcTypeShapValues calcType,
    NPar::ILocalExecutor* localExecutor,
    TShapPreparedTrees* preparedTrees
) {
    TInteractionValuesFull shapInteractionValues;
    if (pairOfFeatures.Defined()) {
        CalcShapInteraction<TInteractionValuesSubset>(
            model,
            dataset,
            pairOfFeatures,
            logPeriod,
            localExecutor,
            preparedTrees,
            &shapInteractionValues,
            calcType
        );
        if (pairOfFeatures->first != pairOfFeatures->second) {
            SetSymmetricValues(&shapInteractionValues);
        }
    } else {
        CalcShapInteraction<TInteractionValuesFull>(
            model,
            dataset,
            pairOfFeatures,
            logPeriod,
            localExecutor,
            preparedTrees,
            &shapInteractionValues,
            calcType
        );
    }
    return shapInteractionValues;
}

TInteractionValuesFull CalcShapInteractionValuesMulti(
    const TFullModel& model,
    const TDataProvider& dataset,
    const TMaybe<std::pair<int, int>>& pairOfFeatures,
    int logPeriod,
    EPreCalcShapValues mode,
    NPar::ILocalExecutor* localExecutor,
    ECalcTypeShapValues calcType
) {
    TShapPreparedTrees preparedTrees = PrepareTrees(
        model,
        &dataset,
        /*referenceDataset*/ nullptr,
        mode,
        localExecutor,
        /*calcInternalValues*/ true,
        calcType
    );
    return CalcShapInteractionValuesWithPreparedTrees(
        model,
        dataset,
        pairOfFeatures,
        logPeriod,
        calcType,
        localExecutor,
        &preparedTrees
    );
}
