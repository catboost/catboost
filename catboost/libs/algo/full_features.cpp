#include "full_features.h"
#include "split.h"

#include <catboost/libs/helpers/mem_usage.h>
#include <catboost/libs/data/load_data.h>

#include <library/threading/local_executor/local_executor.h>
#include <util/generic/set.h>

size_t TAllFeatures::GetDocCount() const {
    for (const auto& floatHistogram : FloatHistograms) {
        if (!floatHistogram.empty())
            return floatHistogram.size();
    }
    for (const auto& catFeatures : CatFeaturesRemapped) {
        if (!catFeatures.empty())
            return catFeatures.size();
    }
    return 0;
}

template <typename T>
static inline void ClearVector(TVector<T>* dst) {
    static_assert(std::is_pod<T>::value, "T must be a pod");
    dst->clear();
    dst->shrink_to_fit();
}

template <typename TDocSelector>
static inline bool IsConstCatValue(int featureIdx, const TDocumentStorage& docStorage, const TDocSelector& docSelector) {
    size_t docCount = docSelector.GetDocCount();
    if (docCount == 0) {
        return true;
    }
    const TVector<float>& src = docStorage.Factors[featureIdx];
    int src0 = ConvertFloatCatFeatureToIntHash(src[docSelector(0)]);
    for (size_t i = 1; i < docCount; ++i) {
        if (ConvertFloatCatFeatureToIntHash(src[docSelector(i)]) != src0) {
            return false;
        }
    }
    return true;
}

/// Binarize feature `featureIdx` from `docStorage` into cat-feature `catFeatureIdx` in `features`.
template <typename TDocSelector>
static inline void BinarizeCatFeature(int featureIdx,
                                      const TDocumentStorage& docStorage,
                                      const TDocSelector& docSelector,
                                      int catFeatureIdx,
                                      TAllFeatures* features) {
    size_t docCount = docSelector.GetDocCount();
    const TVector<float>& src = docStorage.Factors[featureIdx];
    TVector<int>& dstRemapped = features->CatFeaturesRemapped[catFeatureIdx];
    TVector<int>& dstValues = features->OneHotValues[catFeatureIdx];
    bool dstIsOneHot = features->IsOneHot[catFeatureIdx];

    dstRemapped.resize(docCount);

    using TCatFeaturesRemap = THashMap<int, int>;
    TCatFeaturesRemap uniqueFeaturesRemap;
    if (dstValues.empty()) {
        // Processing learn data
        for (size_t i = 0; i < docCount; ++i) {
            const auto val = ConvertFloatCatFeatureToIntHash(src[docSelector(i)]);
            TCatFeaturesRemap::insert_ctx ctx = nullptr;
            TCatFeaturesRemap::iterator it = uniqueFeaturesRemap.find(val, ctx);
            if (it == uniqueFeaturesRemap.end()) {
                it = uniqueFeaturesRemap.emplace_direct(ctx, val, (int)uniqueFeaturesRemap.size());
            }
            dstRemapped[i] = it->second;
        }
        dstValues.resize(uniqueFeaturesRemap.size());
        for (const auto& kv : uniqueFeaturesRemap) {
            dstValues[kv.second] = kv.first;
        }
        // Cases `dstValues.size() == 1` and `> oneHotMaxSize` are up to the caller.
    } else {
        for (size_t i = 0; i < dstValues.size(); ++i) {
            uniqueFeaturesRemap.emplace(dstValues[i], static_cast<int>(i));
        }
        if (dstIsOneHot) {
            for (size_t i = 0; i < docCount; ++i) {
                const auto val = ConvertFloatCatFeatureToIntHash(src[docSelector(i)]);
                TCatFeaturesRemap::iterator it = uniqueFeaturesRemap.find(val);
                if (it == uniqueFeaturesRemap.end()) {
                    dstRemapped[i] = static_cast<int>(uniqueFeaturesRemap.size());
                } else {
                    dstRemapped[i] = it->second;
                }
            }
        } else {
            for (size_t i = 0; i < docCount; ++i) {
                const auto val = ConvertFloatCatFeatureToIntHash(src[docSelector(i)]);
                TCatFeaturesRemap::insert_ctx ctx = nullptr;
                TCatFeaturesRemap::iterator it = uniqueFeaturesRemap.find(val, ctx);
                if (it == uniqueFeaturesRemap.end()) {
                    int remap = static_cast<int>(uniqueFeaturesRemap.size());
                    dstValues.push_back(val);
                    it = uniqueFeaturesRemap.emplace_direct(ctx, val, remap);
                    dstRemapped[i] = it->second;
                } else {
                    dstRemapped[i] = it->second;
                }
            }
        }
    }
}

/*
 * Binarize feature `floatFeature.FeatureIndex` from `docStorage`
 * into float feature `floatFeature.FeatureIndex` in `features`.
 */
template <typename TDocSelector>
static inline void BinarizeFloatFeature(const TFloatFeature& floatFeature,
                                        const TDocumentStorage& docStorage,
                                        const TDocSelector& docSelector,
                                        NPar::TLocalExecutor& localExecutor,
                                        TAllFeatures* features,
                                        bool* seenNans) {
    size_t docCount = docSelector.GetDocCount();
    const TVector<float>& src = docStorage.Factors[floatFeature.FlatFeatureIndex];
    TVector<ui8>& hist = features->FloatHistograms[floatFeature.FeatureIndex];

    hist.resize(docCount);

    ui8* histData = hist.data();
    const float* featureBorderData = floatFeature.Borders.data();
    const int featureBorderSize = floatFeature.Borders.ysize();
    NCatBoostFbs::ENanValueTreatment nanValueTreatment = floatFeature.NanValueTreatment;

    localExecutor.ExecRange([&] (int i) {
        const auto& featureVal = src[docSelector(i)];
        if (IsNan(featureVal)) {
            *seenNans = true;
            histData[i] =
                nanValueTreatment == NCatBoostFbs::ENanValueTreatment_AsTrue ? featureBorderSize : 0;
        } else {
            int j = 0;
            while (j < featureBorderSize && featureVal > featureBorderData[j]) {
                ++histData[i];
                ++j;
            }
        //    histData[i] = LowerBound(featureBorderData, featureBorderData + featureBorderSize, featureVal) - featureBorderData;
        }
    }
    , NPar::TLocalExecutor::TExecRangeParams(0, docCount).SetBlockSize(1000)
    , NPar::TLocalExecutor::WAIT_COMPLETE);
}

/// Allocate binarized data holders in `features`.
static void PrepareSlots(size_t catFeatureCount,
                         size_t floatFeatureCount,
                         TMaybe<const TVector<TOneHotFeature>*> oneHotFeatures,
                         TAllFeatures* features) {
    features->CatFeaturesRemapped.resize(catFeatureCount);
    features->OneHotValues.resize(catFeatureCount);
    if (oneHotFeatures.Defined()) {
        features->IsOneHot.resize(catFeatureCount, false);
        for (const auto& oneHotFeature : **oneHotFeatures) {
            features->IsOneHot[oneHotFeature.CatFeatureIndex] = true;
            features->OneHotValues[oneHotFeature.CatFeatureIndex] = oneHotFeature.Values;
        }
    } else {
        // will be set proper values later in the process
        features->IsOneHot.resize(catFeatureCount, true);
    }
    features->FloatHistograms.resize(floatFeatureCount);
}

/// Prepare slots of `testFeatures` after that of `learnFeatures`.
static void PrepareSlotsAfter(const TAllFeatures& learnFeatures, TAllFeatures* testFeatures) {
    testFeatures->CatFeaturesRemapped.resize(learnFeatures.IsOneHot.size());
    testFeatures->IsOneHot = learnFeatures.IsOneHot;
    testFeatures->OneHotValues = learnFeatures.OneHotValues;
    testFeatures->FloatHistograms.resize(learnFeatures.FloatHistograms.size());
}

// Apply one-hot value count limit to cat features.
static void CleanupOneHotFeatures(size_t oneHotMaxSize, TAllFeatures* features) {
    for (int catFeatureIdx = 0; catFeatureIdx < features->OneHotValues.ysize(); ++catFeatureIdx) {
        if (features->OneHotValues[catFeatureIdx].size() > oneHotMaxSize) {
            features->IsOneHot[catFeatureIdx] = false;
        }
    }
}

namespace {
    /// Select all documents in range [0, docCount).
    class TSelectAll {
    public:
        explicit TSelectAll(size_t docCount)
        : DocCount(docCount)
        {}

        size_t GetDocCount() const {
            return DocCount;
        }

        size_t operator()(size_t i) const {
            return i;
        }
    private:
        size_t DocCount;
    };

    /// Select documents by specified indices.
    class TSelectIndices {
    public:
        explicit TSelectIndices(const TVector<size_t>& indices)
        : Indices(indices)
        {}

        size_t GetDocCount() const {
            return Indices.size();
        }

        size_t operator()(size_t i) const {
            return Indices[i];
        }
    private:
        const TVector<size_t>& Indices;
    };

    /// File-local type for preparing and performing binarization
    class TBinarizer {
    public:
        TBinarizer(int featureCount,
                   const THashSet<int>& categFeatures,
                   const TVector<TFloatFeature>& floatFeatures,
                   NPar::TLocalExecutor& localExecutor)
        : FeatureCount(featureCount)
        , CategFeatures(categFeatures)
        , FloatFeatures(floatFeatures)
        , LocalExecutor(localExecutor)
        {
            TypedFeatureIdx.resize(featureCount);
            for (int featureIdx = 0; featureIdx < featureCount; ++featureIdx) {
                if (CategFeatures.has(featureIdx)) {
                    TypedFeatureIdx[featureIdx] = CatFeatureCount++;
                } else {
                    TypedFeatureIdx[featureIdx] = FloatFeatureCount++;
                }
            }
        }

        size_t GetCatFeatureCount() const {
            return CatFeatureCount;
        }

        size_t GetFloatFeatureCount() const {
            return FloatFeatureCount;
        }

        /// Register the features listed in `ignoredFeatures` for skipping.
        void SetupToIgnoreFeatures(const TVector<int>& ignoredFeatures, bool ignoreRedundantCatFeatures) {
            IgnoredFeatures.clear();
            IgnoredFeatures.insert(ignoredFeatures.begin(), ignoredFeatures.end());
            IgnoreRedundantCatFeatures = ignoreRedundantCatFeatures;
        }

        /// Register the features absent in `learnFeatures` for skipping.
        void SetupToIgnoreFeaturesAfter(const TAllFeatures& learnFeatures) {
            IgnoredFeatures.clear();
            for (int featureIdx = 0; featureIdx < FeatureCount; ++featureIdx) {
                if (CategFeatures.has(featureIdx)) {
                    auto catFeatureIdx = TypedFeatureIdx[featureIdx];
                    if (learnFeatures.CatFeaturesRemapped[catFeatureIdx].empty()) {
                        IgnoredFeatures.insert(featureIdx);
                    }
                } else {
                    auto floatFeatureIdx = TypedFeatureIdx[featureIdx];
                    if (learnFeatures.FloatHistograms[floatFeatureIdx].empty()) {
                        IgnoredFeatures.insert(featureIdx);
                    }
                }
            }
        }

        /// Perform binarization of `docStorage` into `features`.
        void Binarize(bool allowNans,
                      TDocumentStorage* docStorage,
                      const TVector<size_t>& selectedDocIndices,
                      bool clearPool,
                      TAllFeatures* features) const {

            auto binarizeBlockOfFeatures = [&](int blockId) {
                int lastFeatureIdx = Min((blockId + 1) * BlockSize, FeatureCount);
                for (int featureIdx = blockId * BlockSize; featureIdx  < lastFeatureIdx; ++featureIdx) {
                    if (IgnoredFeatures.has(featureIdx)) {
                        if (clearPool) {
                            ClearVector(&docStorage->Factors[featureIdx]);
                        }
                        continue;
                    }
                    if (CategFeatures.has(featureIdx)) {
                        int catFeatureIdx = TypedFeatureIdx[featureIdx];
                        if (selectedDocIndices.empty()) {
                            TSelectAll selectedDocs(docStorage->GetDocCount());
                            if (IgnoreRedundantCatFeatures && IsConstCatValue(featureIdx, *docStorage, selectedDocs)) {
                                MATRIXNET_INFO_LOG << "feature " << featureIdx << " is redundant categorical feature, skipping it" << Endl;
                                if (clearPool) {
                                    ClearVector(&docStorage->Factors[featureIdx]);
                                }
                                continue;
                            }
                            BinarizeCatFeature(featureIdx, *docStorage, selectedDocs, catFeatureIdx, features);
                        } else {
                            TSelectIndices selectedDocs(selectedDocIndices);
                            if (IgnoreRedundantCatFeatures && IsConstCatValue(featureIdx, *docStorage, selectedDocs)) {
                                MATRIXNET_INFO_LOG << "feature " << featureIdx << " is redundant categorical feature, skipping it" << Endl;
                                if (clearPool) {
                                    ClearVector(&docStorage->Factors[featureIdx]);
                                }
                                continue;
                            }
                            BinarizeCatFeature(featureIdx, *docStorage, selectedDocs, catFeatureIdx, features);
                        }
                    } else {
                        int floatFeatureIdx = TypedFeatureIdx[featureIdx];
                        if (FloatFeatures[floatFeatureIdx].Borders.empty()) {
                            if (clearPool) {
                                ClearVector(&docStorage->Factors[featureIdx]);
                            }
                            continue;
                        }
                        bool seenNans = false;
                        if (selectedDocIndices.empty()) {
                            BinarizeFloatFeature(
                                FloatFeatures[floatFeatureIdx],
                                *docStorage,
                                TSelectAll(docStorage->GetDocCount()),
                                LocalExecutor,
                                features,
                                &seenNans
                            );
                        } else {
                            BinarizeFloatFeature(
                                FloatFeatures[floatFeatureIdx],
                                *docStorage,
                                TSelectIndices(selectedDocIndices),
                                LocalExecutor,
                                features,
                                &seenNans
                            );
                        }
                        if (seenNans) {
                            bool mayHaveNans = FloatFeatures[floatFeatureIdx].HasNans || allowNans;
                            CB_ENSURE(mayHaveNans, "There are NaNs in test dataset (feature number " << featureIdx << ") but there were no NaNs in learn dataset");
                        }
                        if (clearPool) {
                            ClearVector(&docStorage->Factors[featureIdx]);
                        }
                    }
                }
            };

            int blockCount = static_cast<int>((FeatureCount + BlockSize - 1) / BlockSize);
            LocalExecutor.ExecRangeWithThrow(
                binarizeBlockOfFeatures,
                0, blockCount, NPar::TLocalExecutor::WAIT_COMPLETE);
        }

    private:
        int FeatureCount = 0;
        size_t CatFeatureCount = 0;
        size_t FloatFeatureCount = 0;
        const THashSet<int>& CategFeatures;
        const TVector<TFloatFeature>& FloatFeatures;
        NPar::TLocalExecutor& LocalExecutor;
        THashSet<int> IgnoredFeatures;
        bool IgnoreRedundantCatFeatures = false;
        TVector<size_t> TypedFeatureIdx;
        const int BlockSize = 10;
    };
}

void PrepareAllFeaturesLearn(const THashSet<int>& categFeatures,
                             const TVector<TFloatFeature>& floatFeatures,
                             TMaybe<const TVector<TOneHotFeature>*> oneHotFeatures,
                             const TVector<int>& ignoredFeatures,
                             bool ignoreRedundantCatFeatures,
                             size_t oneHotMaxSize,
                             bool clearPool,
                             NPar::TLocalExecutor& localExecutor,
                             const TVector<size_t>& selectedDocIndices,
                             TDocumentStorage* learnDocStorage,
                             TAllFeatures* learnFeatures) {
    if (learnDocStorage->GetDocCount() == 0) {
        return;
    }

    TBinarizer binarizer(learnDocStorage->GetEffectiveFactorCount(), categFeatures, floatFeatures, localExecutor);
    binarizer.SetupToIgnoreFeatures(ignoredFeatures, ignoreRedundantCatFeatures);
    PrepareSlots(binarizer.GetCatFeatureCount(), binarizer.GetFloatFeatureCount(), oneHotFeatures, learnFeatures);
    binarizer.Binarize(/*allowNans=*/true, learnDocStorage, selectedDocIndices, clearPool, learnFeatures);
    CleanupOneHotFeatures(oneHotMaxSize, learnFeatures);
    CB_ENSURE(learnFeatures->GetDocCount() > 0, "Train dataset is empty after binarization");
    DumpMemUsage("Extract bools done");
}

void PrepareAllFeaturesTest(const THashSet<int>& categFeatures,
                            const TVector<TFloatFeature>& floatFeatures,
                            const TAllFeatures& learnFeatures,
                            bool allowNansOnlyInTest,
                            bool clearPool,
                            NPar::TLocalExecutor& localExecutor,
                            const TVector<size_t>& selectedDocIndices,
                            TDocumentStorage* testDocStorage,
                            TAllFeatures* testFeatures) {
    if (testDocStorage->GetDocCount() == 0) {
        return;
    }

    TBinarizer binarizer(testDocStorage->GetEffectiveFactorCount(), categFeatures, floatFeatures, localExecutor);
    binarizer.SetupToIgnoreFeaturesAfter(learnFeatures);
    PrepareSlotsAfter(learnFeatures, testFeatures);
    binarizer.Binarize(allowNansOnlyInTest, testDocStorage, selectedDocIndices, clearPool, testFeatures);
    DumpMemUsage("Extract bools done");
}
