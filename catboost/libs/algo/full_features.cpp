#include "full_features.h"
#include "split.h"

#include <catboost/libs/helpers/mem_usage.h>
#include <catboost/libs/data/load_data.h>

#include <library/threading/local_executor/local_executor.h>
#include <util/generic/set.h>

const int LearnNotSet = -1;

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


static void AddReason(TVector<ui8>* hist,
                      const TDocumentStorage& docStorage,
                      const TVector<size_t>& docIndices,
                      int idx,
                      ENanMode nanMode,
                      bool nanInLearn,
                      const TVector<float>& featureBorder,
                      NPar::TLocalExecutor* localExecutor)
{
    ui8* histData = hist->data();
    const float* featureBorderData = featureBorder.data();
    const yssize_t featureBorderSize = featureBorder.ysize();
    bool hasNans = false;
    localExecutor->ExecRange([histData, featureBorderData, featureBorderSize, &hasNans, idx, nanMode, &docStorage, &docIndices] (int i) {
        const auto& featureVal = docStorage.Factors[idx][docIndices[i]];
        if (IsNan(featureVal)) {
            hasNans = true;
            histData[i] = nanMode == ENanMode::Min ? 0 : featureBorderSize;
        } else {
            histData[i] = LowerBound(featureBorderData, featureBorderData + featureBorderSize, featureVal) - featureBorderData;
        }
    }, NPar::TLocalExecutor::TExecRangeParams(0, docStorage.GetDocCount()).SetBlockSize(1000)
     , NPar::TLocalExecutor::WAIT_COMPLETE);
    CB_ENSURE(!hasNans || nanInLearn, "There are nans in test dataset (feature number " << idx << ") but there were not nans in learn dataset");
}

static bool IsRedundantFeature(const TDocumentStorage& docStorage, const TVector<size_t>& docIndices, int learnSampleCount, int featureIdx) {
    for (int i = 1; i < learnSampleCount; ++i) {
        if (docStorage.Factors[featureIdx][docIndices[i]] != docStorage.Factors[featureIdx][docIndices[0]]) {
            return false;
        }
    }
    return true;
}

template <typename T>
static void ClearVector(TVector<T>* dst) {
    static_assert(std::is_pod<T>::value, "T must be a pod");
    dst->clear();
    dst->shrink_to_fit();
}

static void ExtractBoolsFromDocInfo(const TVector<size_t>& docIndices,
                                    const THashSet<int>& categFeatures,
                                    const TVector<TFloatFeature>& floatFeatures,
                                    const TVector<int>& ignoredFeatures,
                                    int learnSampleCount,
                                    size_t oneHotMaxSize,
                                    ENanMode nanMode,
                                    bool allowClearPool,
                                    NPar::TLocalExecutor& localExecutor,
                                    TDocumentStorage* docStorage,
                                    TVector<TVector<ui8>>* hist,
                                    TVector<TVector<int>>* catFeaturesRemapped,
                                    TVector<TVector<int>>* oneHotValues,
                                    TVector<bool>* isOneHot) {
    THashSet<int> ignoredFeaturesSet(ignoredFeatures.begin(), ignoredFeatures.end());
    const auto featureCount = docStorage->GetFactorsCount();
    TVector<size_t> reasonTargetIdx(featureCount);
    size_t catFeatureIdx = 0;
    size_t floatFeatureIdx = 0;
    for (int featureIdx = 0; featureIdx < featureCount; ++featureIdx) {
        if (categFeatures.has(featureIdx)) {
            reasonTargetIdx[featureIdx] = catFeatureIdx;
            ++catFeatureIdx;
        } else {
            reasonTargetIdx[featureIdx] = floatFeatureIdx;
            ++floatFeatureIdx;
        }
    }

    catFeaturesRemapped->resize(catFeatureIdx, TVector<int>(docStorage->GetDocCount()));
    oneHotValues->resize(catFeatureIdx);
    isOneHot->resize(catFeatureIdx, false);
    hist->resize(floatFeatureIdx, TVector<ui8>(docStorage->GetDocCount()));

    const int BlockSize = 10;
    auto calcHistogramsInFeatureBlock = [&](int blockId) {
        int lastFeatureIdx = Min((blockId + 1) * BlockSize, (int)featureCount);
        for (int featureIdx = blockId * BlockSize; featureIdx  < lastFeatureIdx; ++featureIdx) {
            if (categFeatures.has(featureIdx)) {
                TVector<int>& dstRemapped = (*catFeaturesRemapped)[reasonTargetIdx[featureIdx]];
                TVector<int>& dstValues = (*oneHotValues)[reasonTargetIdx[featureIdx]];
                bool& dstIsOneHot = (*isOneHot)[reasonTargetIdx[featureIdx]];

                bool isRedundantFeature = false;
                if (learnSampleCount != LearnNotSet) {
                    isRedundantFeature = IsRedundantFeature(*docStorage, docIndices, learnSampleCount, featureIdx);
                    if (isRedundantFeature) {
                        MATRIXNET_INFO_LOG << "feature " << featureIdx << " is redundant categorical feature, skipping it" << Endl;
                    }
                }

                if (ignoredFeaturesSet.has(featureIdx) || isRedundantFeature) {
                    ClearVector(&dstRemapped);
                    ClearVector(&dstValues);
                } else {
                    if (learnSampleCount != LearnNotSet) {
                        using TCatFeaturesRemap = THashMap<int, int>;
                        TCatFeaturesRemap uniqueFeaturesRemap;
                        for (int i = 0; i < learnSampleCount; ++i) {
                            const auto val = ConvertFloatCatFeatureToIntHash(docStorage->Factors[featureIdx][docIndices[i]]);
                            TCatFeaturesRemap::insert_ctx ctx = nullptr;
                            TCatFeaturesRemap::iterator it = uniqueFeaturesRemap.find(val, ctx);
                            if (it == uniqueFeaturesRemap.end()) {
                              it = uniqueFeaturesRemap.emplace_direct(ctx, val, (int)uniqueFeaturesRemap.size());
                            }
                            dstRemapped[i] = it->second;
                        }
                        if (uniqueFeaturesRemap.size() <= oneHotMaxSize) {
                            dstIsOneHot = true;
                            for (int i = learnSampleCount; i < (int)docStorage->GetDocCount(); ++i) {
                                const auto val = ConvertFloatCatFeatureToIntHash(docStorage->Factors[featureIdx][docIndices[i]]);
                                TCatFeaturesRemap::iterator it = uniqueFeaturesRemap.find(val);
                                if (it != uniqueFeaturesRemap.end()) {
                                    dstRemapped[i] = it->second;
                                } else {
                                    dstRemapped[i] = static_cast<int>(uniqueFeaturesRemap.size());
                                }
                            }
                        } else {
                            // We store all hash values only for non one-hot features
                            dstIsOneHot = false;
                            for (int i = learnSampleCount; i < (int)docStorage->GetDocCount(); ++i) {
                                const auto val = ConvertFloatCatFeatureToIntHash(docStorage->Factors[featureIdx][docIndices[i]]);
                                TCatFeaturesRemap::insert_ctx ctx = nullptr;
                                TCatFeaturesRemap::iterator it = uniqueFeaturesRemap.find(val, ctx);
                                if (it == uniqueFeaturesRemap.end()) {
                                    it = uniqueFeaturesRemap.emplace_direct(ctx, val, (int)uniqueFeaturesRemap.size());
                                }
                                dstRemapped[i] = it->second;
                            }
                        }

                        dstValues.resize(uniqueFeaturesRemap.size());
                        for (const auto& kv : uniqueFeaturesRemap) {
                            dstValues[kv.second] = kv.first;
                        }
                    } else {
                        ClearVector(&dstRemapped);
                        ClearVector(&dstValues);
                    }
                }
            } else {
                const auto reasonIdx = reasonTargetIdx[featureIdx];
                TVector<ui8>& dst = hist->at(reasonIdx);
                if (ignoredFeaturesSet.has(featureIdx) || floatFeatures[reasonIdx].Borders.empty()) {
                    dst.clear();
                    dst.shrink_to_fit();
                } else {
                    AddReason(
                        &hist->at(reasonTargetIdx[featureIdx]),
                        *docStorage,
                        docIndices,
                        featureIdx,
                        nanMode,
                        floatFeatures[reasonIdx].HasNans,
                        floatFeatures[reasonIdx].Borders,
                        &localExecutor);
                    if (allowClearPool) {
                        docStorage->Factors[featureIdx].clear();
                        docStorage->Factors[featureIdx].shrink_to_fit();
                    }
                }
            }
        }
    };
    localExecutor.ExecRange(calcHistogramsInFeatureBlock,
                            0,
                            (int)(featureCount + BlockSize - 1) / BlockSize,
                            NPar::TLocalExecutor::WAIT_COMPLETE);
    DumpMemUsage("Extract bools done");
}

void PrepareAllFeaturesFromPermutedDocs(const TVector<size_t>& docIndices,
                                        const THashSet<int>& categFeatures,
                                        const TVector<TFloatFeature>& floatFeatures,
                                        const TVector<int>& ignoredFeatures,
                                        int learnSampleCount,
                                        size_t oneHotMaxSize,
                                        ENanMode nanMode,
                                        bool allowClearPool,
                                        NPar::TLocalExecutor& localExecutor,
                                        TDocumentStorage* docStorage,
                                        TAllFeatures* allFeatures) {
    if (docStorage->GetDocCount() == 0) {
        return;
    }

    ExtractBoolsFromDocInfo(docIndices,
                            categFeatures,
                            floatFeatures,
                            ignoredFeatures,
                            learnSampleCount,
                            oneHotMaxSize,
                            nanMode,
                            allowClearPool,
                            localExecutor,
                            docStorage,
                            &allFeatures->FloatHistograms,
                            &allFeatures->CatFeaturesRemapped,
                            &allFeatures->OneHotValues,
                            &allFeatures->IsOneHot);

    for (const auto& cf : allFeatures->CatFeaturesRemapped) {
        Y_ASSERT(cf.empty() || cf.size() == docStorage->GetDocCount());
    }
}

void PrepareAllFeatures(const THashSet<int>& categFeatures,
                        const TVector<TFloatFeature>& floatFeatures,
                        const TVector<int>& ignoredFeatures,
                        int learnSampleCount,
                        size_t oneHotMaxSize,
                        ENanMode nanMode,
                        bool allowClearPool,
                        NPar::TLocalExecutor& localExecutor,
                        TDocumentStorage* docStorage,
                        TAllFeatures* allFeatures)
{
    TVector<size_t> indices(docStorage->GetDocCount(), 0);
    std::iota(indices.begin(), indices.end(), 0);

    PrepareAllFeaturesFromPermutedDocs(
        indices,
        categFeatures,
        floatFeatures,
        ignoredFeatures,
        learnSampleCount,
        oneHotMaxSize,
        nanMode,
        allowClearPool,
        localExecutor,
        docStorage,
        allFeatures);
}

namespace {
    class TBinarizationHelper {
    public:
        int FeatureCount = 0;
        const THashSet<int>& CategFeatures;
        const TVector<TFloatFeature>& FloatFeatures;
        ENanMode NanMode;
        NPar::TLocalExecutor& LocalExecutor;

        THashSet<int> IgnoredFeatures;
        TVector<size_t> TypedFeatureIdx; // ReasonTargetIdx
        size_t CatFeatureCount = 0;
        size_t FloatFeatureCount = 0;

        const int BlockSize = 10;

        TBinarizationHelper(int featureCount,
                            const THashSet<int>& categFeatures,
                            const TVector<TFloatFeature>& floatFeatures,
                            const TVector<int>& ignoredFeatures,
                            ENanMode nanMode,
                            NPar::TLocalExecutor& localExecutor)
        : FeatureCount(featureCount)
        , CategFeatures(categFeatures)
        , FloatFeatures(floatFeatures)
        , NanMode(nanMode)
        , LocalExecutor(localExecutor)
        , IgnoredFeatures(ignoredFeatures.begin(), ignoredFeatures.end())
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

        /// Prepare `features` for binarization.
        void PrepareFeatures(TAllFeatures* features) {
            features->CatFeaturesRemapped.resize(CatFeatureCount);
            features->OneHotValues.resize(CatFeatureCount);
            features->IsOneHot.resize(CatFeatureCount, true);
            features->FloatHistograms.resize(FloatFeatureCount);
        }

        void PrepareTestFeatures(const TAllFeatures* learnFeatures, TAllFeatures* testFeatures) {
            PrepareFeatures(testFeatures);
            for (int featureIdx = 0; featureIdx < FeatureCount; ++featureIdx) {
                if (CategFeatures.has(featureIdx)) {
                    auto catFeatureIdx = TypedFeatureIdx[featureIdx];
                    if (learnFeatures->CatFeaturesRemapped[catFeatureIdx].empty()) {
                        IgnoredFeatures.insert(featureIdx);
                        continue;
                    }
                    testFeatures->OneHotValues[catFeatureIdx] = learnFeatures->OneHotValues[catFeatureIdx];
                    testFeatures->IsOneHot[catFeatureIdx] = learnFeatures->IsOneHot[catFeatureIdx];
                } else {
                    auto floatFeatureIdx = TypedFeatureIdx[featureIdx];
                    if (learnFeatures->FloatHistograms[floatFeatureIdx].empty()) {
                        IgnoredFeatures.insert(featureIdx);
                    }
                }
            }
        }

        /// Select all documents in range [0, docCount).
        class TSelectAll {
        public:
            TSelectAll(size_t docCount)
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
            TSelectIndices(const TVector<size_t>& indices)
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

        void Binarize(bool forLearn, bool clearPool, const TVector<size_t>& select, TDocumentStorage* docStorage, TAllFeatures* features) {

            auto binarizeBlockOfFeatures = [&](int blockId) {
                int lastFeatureIdx = Min((blockId + 1) * BlockSize, (int)FeatureCount);
                for (int featureIdx = blockId * BlockSize; featureIdx  < lastFeatureIdx; ++featureIdx) {
                    if (IgnoredFeatures.has(featureIdx)) {
                        // clear pool?
                        continue;
                    }
                    if (CategFeatures.has(featureIdx)) {
                        int catFeatureIdx = TypedFeatureIdx[featureIdx];
                        if (select.empty()) {
                            BinarizeCatFeature(featureIdx, catFeatureIdx, TSelectAll(docStorage->GetDocCount()), docStorage, features);
                        } else {
                            BinarizeCatFeature(featureIdx, catFeatureIdx, TSelectIndices(select), docStorage, features);
                        }
                    } else {
                        int floatFeatureIdx = TypedFeatureIdx[featureIdx];
                        if (FloatFeatures[floatFeatureIdx].Borders.empty()) {
                            continue;
                        }
                        bool hasNans;
                        if (select.empty()) {
                            hasNans = BinarizeFloatFeature(featureIdx, floatFeatureIdx, TSelectAll(docStorage->GetDocCount()), docStorage, features);
                        } else {
                            hasNans = BinarizeFloatFeature(featureIdx, floatFeatureIdx, TSelectIndices(select), docStorage, features);
                        }
                        if (hasNans) {
                            bool mayHaveNans = forLearn || FloatFeatures[floatFeatureIdx].HasNans;
                            CB_ENSURE(mayHaveNans, "There are nans in test dataset (feature number " << featureIdx << ") but there were not nans in learn dataset");
                        }
                        if (clearPool) {
                            ClearVector(&docStorage->Factors[featureIdx]);
                        }
                    }
                }
            };

            LocalExecutor.ExecRange(binarizeBlockOfFeatures,
                                    0,
                                    (int)(FeatureCount + BlockSize - 1) / BlockSize,
                                    NPar::TLocalExecutor::WAIT_COMPLETE);

        }

        template <typename Selector>
        void BinarizeCatFeature(int featureIdx, int catFeatureIdx, const Selector& select, TDocumentStorage* docStorage, TAllFeatures* features) {
            size_t docCount = select.GetDocCount();
            TVector<float>& src = docStorage->Factors[featureIdx];
            TVector<int>& dstRemapped = features->CatFeaturesRemapped[catFeatureIdx];
            TVector<int>& dstValues = features->OneHotValues[catFeatureIdx];
            bool dstIsOneHot = features->IsOneHot[catFeatureIdx];

            dstRemapped.resize(docCount);

            using TCatFeaturesRemap = THashMap<int, int>;
            TCatFeaturesRemap uniqueFeaturesRemap;
            if (dstValues.empty()) {
                // Processing learn data
                for (size_t i = 0; i < docCount; ++i) {
                    const auto val = ConvertFloatCatFeatureToIntHash(src[select(i)]);
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
                        const auto val = ConvertFloatCatFeatureToIntHash(src[select(i)]);
                        TCatFeaturesRemap::iterator it = uniqueFeaturesRemap.find(val);
                        if (it == uniqueFeaturesRemap.end()) {
                            dstRemapped[i] = static_cast<int>(uniqueFeaturesRemap.size());
                        } else {
                            dstRemapped[i] = it->second;
                        }
                    }
                } else {
                    for (size_t i = 0; i < docCount; ++i) {
                        const auto val = ConvertFloatCatFeatureToIntHash(src[select(i)]);
                        TCatFeaturesRemap::insert_ctx ctx = nullptr;
                        TCatFeaturesRemap::iterator it = uniqueFeaturesRemap.find(val, ctx);
                        if (it == uniqueFeaturesRemap.end()) {
                            int remap = static_cast<int>(uniqueFeaturesRemap.size());
                            dstValues.push_back(remap);
                            it = uniqueFeaturesRemap.emplace_direct(ctx, val, remap);
                            dstRemapped[i] = it->second;
                        } else {
                            dstRemapped[i] = it->second;
                        }
                    }
                }
            }
        }

        /// @return Whether the feature has NaNs
        template <typename Selector>
        bool BinarizeFloatFeature(int featureIdx, int floatFeatureIdx, const Selector& select, TDocumentStorage* docStorage, TAllFeatures* features) {
            size_t docCount = select.GetDocCount();
            TVector<float>& src = docStorage->Factors[featureIdx];
            TVector<ui8>& hist = features->FloatHistograms[floatFeatureIdx];

            hist.resize(docCount);

            ui8* histData = hist.data();
            const float* featureBorderData = FloatFeatures[floatFeatureIdx].Borders.data();
            const yssize_t featureBorderSize = FloatFeatures[floatFeatureIdx].Borders.ysize();
            bool hasNans = false;

            LocalExecutor.ExecRange([&] (int i) {
                const auto& featureVal = src[select(i)];
                if (IsNan(featureVal)) {
                    hasNans = true;
                    histData[i] = NanMode == ENanMode::Min ? 0 : featureBorderSize;
                } else {
                    histData[i] = LowerBound(featureBorderData, featureBorderData + featureBorderSize, featureVal) - featureBorderData;
                }
            }, NPar::TLocalExecutor::TExecRangeParams(0, docCount).SetBlockSize(1000)
            , NPar::TLocalExecutor::WAIT_COMPLETE);

            return hasNans;
        }
    };

    /// Remove all-same features and limit one-hot features.
    void  CleanupCatFeatures(bool removeRedundant, size_t oneHotMaxSize, TAllFeatures* features) {
        for (int catFeatureIdx = 0; catFeatureIdx < features->OneHotValues.ysize(); ++catFeatureIdx) {
            auto& oneHotValues = features->OneHotValues[catFeatureIdx];
            if (removeRedundant && oneHotValues.size() == 1) {
                ClearVector(&oneHotValues);
                ClearVector(&features->CatFeaturesRemapped[catFeatureIdx]);
            }
            if (oneHotValues.size() > oneHotMaxSize) {
                features->IsOneHot[catFeatureIdx] = false;
            }
        }
    }
}

void PrepareAllFeaturesLearn(const THashSet<int>& categFeatures,
                             const TVector<TFloatFeature>& floatFeatures,
                             const TVector<int>& ignoredFeatures,
                             bool ignoreRedundantCatFeatures,
                             size_t oneHotMaxSize,
                             ENanMode nanMode,
                             bool clearPool,
                             NPar::TLocalExecutor& localExecutor,
                             const TVector<size_t>& select,
                             TDocumentStorage* learnDocStorage,
                             TAllFeatures* learnFeatures) {
    if (learnDocStorage->GetDocCount() == 0) {
        return;
    }
    TBinarizationHelper binarizer(learnDocStorage->GetFactorsCount(), categFeatures, floatFeatures, ignoredFeatures, nanMode, localExecutor);
    binarizer.PrepareFeatures(learnFeatures);
    binarizer.Binarize(/*forLearn=*/true, clearPool, select, learnDocStorage, learnFeatures);
    CleanupCatFeatures(ignoreRedundantCatFeatures, oneHotMaxSize, learnFeatures);
    DumpMemUsage("Extract bools done");
}

void PrepareAllFeaturesTest(const THashSet<int>& categFeatures,
                            const TVector<TFloatFeature>& floatFeatures,
                            const TAllFeatures& learnFeatures,
                            ENanMode nanMode,
                            bool clearPool,
                            NPar::TLocalExecutor& localExecutor,
                            const TVector<size_t>& select,
                            TDocumentStorage* testDocStorage,
                            TAllFeatures* testFeatures) {
    if (testDocStorage->GetDocCount() == 0) {
        return;
    }
    TBinarizationHelper binarizer(testDocStorage->GetFactorsCount(), categFeatures, floatFeatures, /*ignoredFeatures=*/{}, nanMode, localExecutor);
    binarizer.PrepareTestFeatures(&learnFeatures, testFeatures);
    binarizer.Binarize(/*forLearn=*/false, clearPool, select, testDocStorage, testFeatures);
    DumpMemUsage("Extract bools done");
}
