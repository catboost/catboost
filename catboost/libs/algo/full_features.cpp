#include "full_features.h"
#include "split.h"

#include <catboost/libs/helpers/mem_usage.h>
#include <catboost/libs/data/load_data.h>

#include <library/threading/local_executor/local_executor.h>
#include <util/generic/set.h>

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

static void ClearVector(TVector<int>* dst) {
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
                        TSet<int> uniqueVals;
                        for (int i = 0; i < learnSampleCount; ++i) {
                            const auto val = ConvertFloatCatFeatureToIntHash(docStorage->Factors[featureIdx][docIndices[i]]);
//                          TODO(kirillovs): use THashMap here as the next step of cat feature hashes vector elimination
//                                           (i want to canonize less tests each commit)
//                          TCatFeaturesRemap::insert_ctx ctx = nullptr;
//                          TCatFeaturesRemap::iterator it = uniqueFeaturesRemap.find(val, ctx);
//                          if (it == uniqueFeaturesRemap.end()) {
//                              it = uniqueFeaturesRemap.emplace_direct(ctx, val, (int)uniqueFeaturesRemap.size());
//                          }
                            uniqueVals.insert(val);
                        }
                        if (uniqueVals.size() <= oneHotMaxSize) {
                            dstIsOneHot = true;
                        } else {
                            // We store all hash values only for non one-hot features
                            dstIsOneHot = false;
                            for (int i = learnSampleCount; i < (int)docStorage->GetDocCount(); ++i) {
                                const auto val = ConvertFloatCatFeatureToIntHash(docStorage->Factors[featureIdx][docIndices[i]]);
                                uniqueVals.insert(val);
                            }
                        }
                        for (auto& uv : uniqueVals) {
                            uniqueFeaturesRemap.emplace(uv, uniqueFeaturesRemap.size());
                        }
                        uniqueVals.clear();
                        for (int i = 0; i < (int)docStorage->GetDocCount(); ++i) {
                            const auto val = ConvertFloatCatFeatureToIntHash(docStorage->Factors[featureIdx][docIndices[i]]);
                            if (uniqueFeaturesRemap.has(val)) {
                                dstRemapped[i] = uniqueFeaturesRemap[val];
                            } else {
                                dstRemapped[i] = static_cast<int>(uniqueFeaturesRemap.size());
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
