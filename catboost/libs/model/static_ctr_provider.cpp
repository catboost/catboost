#include "static_ctr_provider.h"
#include <catboost/libs/helpers/exception.h>

struct TCompressedModelCtr {
    const TProjection* Projection;
    yvector<const TModelCtr*> ModelCtrs;
};

void TStaticCtrProvider::CalcCtrs(const yvector<TModelCtr>& neededCtrs,
                                  const NArrayRef::TConstArrayRef<ui8>& binarizedFeatures,
                                  const NArrayRef::TConstArrayRef<int>& hashedCatFeatures,
                                  const IFeatureIndexProvider& binFeatureIndexProvider, size_t docCount,
                                  NArrayRef::TArrayRef<float> result) {
    if (neededCtrs.empty()) {
        return;
    }
    yvector<TCompressedModelCtr> compressedModelCtrs;
    compressedModelCtrs.emplace_back(TCompressedModelCtr{&neededCtrs[0].Projection, {&neededCtrs[0]}});
    for (size_t i = 1; i < neededCtrs.size(); ++i) {
        Y_ASSERT(neededCtrs[i - 1] < neededCtrs[i]); // needed ctrs should be sorted
        if (*(compressedModelCtrs.back().Projection) != neededCtrs[i].Projection) {
            compressedModelCtrs.emplace_back(TCompressedModelCtr{&neededCtrs[i].Projection, {}});
        }
        compressedModelCtrs.back().ModelCtrs.push_back(&neededCtrs[i]);
    }
    size_t samplesCount = docCount;
    yvector<ui64> ctrHashes(samplesCount);
    yvector<ui64> buckets(samplesCount);
    size_t resultIdx = 0;
    float* resultPtr = result.data();
    for (size_t i = 0; i < compressedModelCtrs.size(); ++i) {
        auto& proj = *compressedModelCtrs[i].Projection;
        CalcHashes(proj, binarizedFeatures, hashedCatFeatures, binFeatureIndexProvider, docCount, &ctrHashes);
        for (size_t j = 0; j < compressedModelCtrs[i].ModelCtrs.size(); ++j) {
            auto& ctr = *compressedModelCtrs[i].ModelCtrs[j];
            auto& learnCtr = CtrData.LearnCtrs.at(ctr);
            const ECtrType ctrType = ctr.CtrType;
            auto ptrBuckets = buckets.data();
            for (size_t docId = 0; docId < samplesCount; ++docId) {
                ptrBuckets[docId] = learnCtr.ResolveHashToIndex(ctrHashes[docId]);
            }
            if (ctrType == ECtrType::BinarizedTargetMeanValue || ctrType == ECtrType::FloatTargetMeanValue) {
                auto emptyVal = ctr.Calc(0.f, 0.f);
                for (size_t doc = 0; doc < samplesCount; ++doc) {
                    if (ptrBuckets[doc] != TCtrValueTable::UnknownHash) {
                        const TCtrMeanHistory& ctrMeanHistory = learnCtr.CtrMean[ptrBuckets[doc]];
                        resultPtr[doc + resultIdx] = ctr.Calc(ctrMeanHistory.Sum, ctrMeanHistory.Count);
                    } else {
                        resultPtr[doc + resultIdx] = emptyVal;
                    }
                }
            } else if (ctrType == ECtrType::Counter || ctrType == ECtrType::FeatureFreq) {
                int denominator = learnCtr.CounterDenominator;
                auto emptyVal = ctr.Calc(0, denominator);
                for (size_t doc = 0; doc < samplesCount; ++doc) {
                    if (ptrBuckets[doc] != TCtrValueTable::UnknownHash) {
                        resultPtr[doc + resultIdx] = ctr.Calc(learnCtr.CtrTotal[ptrBuckets[doc]], denominator);
                    } else {
                        resultPtr[doc + resultIdx] = emptyVal;
                    }
                }
            } else if (ctrType == ECtrType::Buckets) {
                auto emptyVal = ctr.Calc(0, 0);
                for (size_t doc = 0; doc < samplesCount; ++doc) {
                    if (ptrBuckets[doc] != TCtrValueTable::UnknownHash) {
                        int goodCount = 0;
                        int totalCount = 0;
                        const yvector<int>& ctrHistory = learnCtr.Ctr[ptrBuckets[doc]];
                        const int targetClassesCount = ctrHistory.ysize();
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
                const int targetClassesCount = learnCtr.Ctr[0].ysize();
                auto emptyVal = ctr.Calc(0, 0);
                if (targetClassesCount > 2) {
                    for (size_t doc = 0; doc < samplesCount; ++doc) {
                        int goodCount = 0;
                        int totalCount = 0;
                        if (ptrBuckets[doc] != TCtrValueTable::UnknownHash) {
                            const int* ctrHistory = learnCtr.Ctr[ptrBuckets[doc]].data();
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
                        if (ptrBuckets[doc] != TCtrValueTable::UnknownHash) {
                            const int* ctrHistory = &learnCtr.CtrTotal[ptrBuckets[doc] * 2];
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
