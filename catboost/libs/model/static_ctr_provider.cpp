#include "static_ctr_provider.h"

#include <catboost/libs/helpers/exception.h>

struct TCompressedModelCtr {
    const TFeatureCombination* Projection;
    TVector<const TModelCtr*> ModelCtrs;
};

void TStaticCtrProvider::CalcCtrs(const TVector<TModelCtr>& neededCtrs,
                                  const TConstArrayRef<ui8>& binarizedFeatures,
                                  const TConstArrayRef<int>& hashedCatFeatures,
                                  size_t docCount,
                                  TArrayRef<float> result) {
    if (neededCtrs.empty()) {
        return;
    }
    TVector<TCompressedModelCtr> compressedModelCtrs;
    compressedModelCtrs.emplace_back(TCompressedModelCtr{&neededCtrs[0].Base.Projection, {&neededCtrs[0]}});
    for (size_t i = 1; i < neededCtrs.size(); ++i) {
        Y_ASSERT(neededCtrs[i - 1] < neededCtrs[i]); // needed ctrs should be sorted
        if (*(compressedModelCtrs.back().Projection) != neededCtrs[i].Base.Projection) {
            compressedModelCtrs.emplace_back(TCompressedModelCtr{&neededCtrs[i].Base.Projection, {}});
        }
        compressedModelCtrs.back().ModelCtrs.push_back(&neededCtrs[i]);
    }
    size_t samplesCount = docCount;
    TVector<ui64> ctrHashes(samplesCount);
    TVector<ui64> buckets(samplesCount);
    size_t resultIdx = 0;
    float* resultPtr = result.data();
    TVector<int> transposedCatFeatureIndexes;
    TVector<TBinFeatureIndexValue> binarizedIndexes;
    for (size_t i = 0; i < compressedModelCtrs.size(); ++i) {
        auto& proj = *compressedModelCtrs[i].Projection;
        binarizedIndexes.clear();
        transposedCatFeatureIndexes.clear();
        for (const auto feature : proj.CatFeatures) {
            transposedCatFeatureIndexes.push_back(CatFeatureIndex.at(feature));
        }
        for (const auto feature : proj.BinFeatures ) {
            binarizedIndexes.push_back(FloatFeatureIndexes.at(feature));
        }
        for (const auto feature : proj.OneHotFeatures ) {
            binarizedIndexes.push_back(OneHotFeatureIndexes.at(feature));
        }
        CalcHashes(binarizedFeatures, hashedCatFeatures, transposedCatFeatureIndexes, binarizedIndexes, docCount, &ctrHashes);
        for (size_t j = 0; j < compressedModelCtrs[i].ModelCtrs.size(); ++j) {
            auto& ctr = *compressedModelCtrs[i].ModelCtrs[j];
            auto& learnCtr = CtrData.LearnCtrs.at(ctr.Base);
            auto hashIndexResolver = learnCtr.GetIndexHashViewer();
            const ECtrType ctrType = ctr.Base.CtrType;
            auto ptrBuckets = buckets.data();
            for (size_t docId = 0; docId < samplesCount; ++docId) {
                ptrBuckets[docId] = hashIndexResolver.GetIndex(ctrHashes[docId]);
            }
            if (ctrType == ECtrType::BinarizedTargetMeanValue || ctrType == ECtrType::FloatTargetMeanValue) {
                const auto emptyVal = ctr.Calc(0.f, 0.f);
                auto ctrMean = learnCtr.GetTypedArrayRefForBlobData<TCtrMeanHistory>();
                for (size_t doc = 0; doc < samplesCount; ++doc) {
                    if (ptrBuckets[doc] != NCatboost::TDenseIndexHashView::NotFoundIndex) {
                        const TCtrMeanHistory& ctrMeanHistory = ctrMean[ptrBuckets[doc]];
                        resultPtr[doc + resultIdx] = ctr.Calc(ctrMeanHistory.Sum, ctrMeanHistory.Count);
                    } else {
                        resultPtr[doc + resultIdx] = emptyVal;
                    }
                }
            } else if (ctrType == ECtrType::Counter || ctrType == ECtrType::FeatureFreq) {
                TConstArrayRef<int> ctrTotal = learnCtr.GetTypedArrayRefForBlobData<int>();
                const int denominator = learnCtr.CounterDenominator;
                auto emptyVal = ctr.Calc(0, denominator);
                for (size_t doc = 0; doc < samplesCount; ++doc) {
                    if (ptrBuckets[doc] != NCatboost::TDenseIndexHashView::NotFoundIndex) {
                        resultPtr[doc + resultIdx] = ctr.Calc(ctrTotal[ptrBuckets[doc]], denominator);
                    } else {
                        resultPtr[doc + resultIdx] = emptyVal;
                    }
                }
            } else if (ctrType == ECtrType::Buckets) {
                auto ctrIntArray = learnCtr.GetTypedArrayRefForBlobData<int>();
                const int targetClassesCount = learnCtr.TargetClassesCount;
                auto emptyVal = ctr.Calc(0, 0);
                for (size_t doc = 0; doc < samplesCount; ++doc) {
                    if (ptrBuckets[doc] != NCatboost::TDenseIndexHashView::NotFoundIndex) {
                        int goodCount = 0;
                        int totalCount = 0;
                        auto ctrHistory = MakeArrayRef(ctrIntArray.data() + ptrBuckets[doc] * targetClassesCount, targetClassesCount);
                        goodCount = ctrHistory[ctr.TargetBorderIdx];
                        for (int classId = 0; classId < targetClassesCount; ++classId) {
                            totalCount += ctrHistory[classId];
                        }
                        resultPtr[doc + resultIdx] = ctr.Calc(goodCount, totalCount);
                    } else {
                        resultPtr[doc + resultIdx] = emptyVal;
                    }
                }
            } else {
                auto ctrIntArray = learnCtr.GetTypedArrayRefForBlobData<int>();
                const int targetClassesCount = learnCtr.TargetClassesCount;

                auto emptyVal = ctr.Calc(0, 0);
                if (targetClassesCount > 2) {
                    for (size_t doc = 0; doc < samplesCount; ++doc) {
                        int goodCount = 0;
                        int totalCount = 0;
                        if (ptrBuckets[doc] != NCatboost::TDenseIndexHashView::NotFoundIndex) {
                            auto ctrHistory = MakeArrayRef(ctrIntArray.data() + ptrBuckets[doc] * targetClassesCount, targetClassesCount);
                            for (int classId = 0; classId < ctr.TargetBorderIdx + 1; ++classId) {
                                totalCount += ctrHistory[classId];
                            }
                            for (int classId = ctr.TargetBorderIdx + 1; classId < targetClassesCount; ++classId) {
                                goodCount += ctrHistory[classId];
                            }
                            totalCount += goodCount;
                        }
                        resultPtr[doc + resultIdx] = ctr.Calc(goodCount, totalCount);
                    }
                } else {
                    for (size_t doc = 0; doc < samplesCount; ++doc) {
                        if (ptrBuckets[doc] != NCatboost::TDenseIndexHashView::NotFoundIndex) {
                            const int* ctrHistory = &ctrIntArray[ptrBuckets[doc] * 2];
                            resultPtr[doc + resultIdx] = ctr.Calc(ctrHistory[1], ctrHistory[0] + ctrHistory[1]);
                        } else {
                            resultPtr[doc + resultIdx] = emptyVal;
                        }
                    }
                }
            }
            resultIdx += docCount;
        }
    }
}

bool TStaticCtrProvider::HasNeededCtrs(const TVector<TModelCtr>& neededCtrs) const {
    for (const auto& ctr : neededCtrs) {
        if (!CtrData.LearnCtrs.has(ctr.Base)) {
            return false;
        }
    }
    return true;
}

void TStaticCtrProvider::SetupBinFeatureIndexes(const TVector<TFloatFeature> &floatFeatures,
                                                const TVector<TOneHotFeature> &oheFeatures,
                                                const TVector<TCatFeature> &catFeatures) {
    ui32 currentIndex = 0;
    FloatFeatureIndexes.clear();
    for (const auto& floatFeature : floatFeatures) {
        for (size_t borderIdx = 0; borderIdx < floatFeature.Borders.size(); ++borderIdx) {
            TBinFeatureIndexValue featureIdx{currentIndex, false, (ui8)(borderIdx + 1)};
            TFloatSplit split{floatFeature.FeatureIndex, floatFeature.Borders[borderIdx]};
            FloatFeatureIndexes[split] = featureIdx;
        }
        ++currentIndex;
    }
    OneHotFeatureIndexes.clear();
    for (const auto& oheFeature : oheFeatures) {
        for (int valueId = 0; valueId < oheFeature.Values.ysize(); ++valueId) {
            TBinFeatureIndexValue featureIdx{currentIndex, true, (ui8)(valueId + 1)};
            TOneHotSplit feature{oheFeature.CatFeatureIndex, oheFeature.Values[valueId]};
            OneHotFeatureIndexes[feature] = featureIdx;
        }
        ++currentIndex;
    }
    CatFeatureIndex.clear();
    for (const auto& catFeature : catFeatures) {
        const int prevSize = CatFeatureIndex.ysize();
        CatFeatureIndex[catFeature.FeatureIndex] = prevSize;
    }
}
