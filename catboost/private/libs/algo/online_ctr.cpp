#include "online_ctr.h"

#include "fold.h"
#include "index_hash_calcer.h"
#include "learn_context.h"
#include "scoring.h"
#include "tree_print.h"

#include <catboost/libs/helpers/array_subset.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/helpers/mem_usage.h>
#include <catboost/libs/helpers/resource_constrained_executor.h>
#include <catboost/libs/model/ctr_value_table.h>
#include <catboost/libs/model/model.h>

#include <library/cpp/threading/local_executor/local_executor.h>

#include <util/generic/bitops.h>
#include <util/generic/scope.h>
#include <util/generic/utility.h>
#include <util/generic/variant.h>
#include <util/generic/xrange.h>
#include <util/system/mem_info.h>
#include <util/system/type_name.h>
#include <util/thread/singleton.h>

#include <algorithm>
#include <numeric>

#if defined(HAVE_CUDA)
#include <catboost/cuda/cuda_lib/cuda_base.h>
#include <catboost/cuda/data/gpu_input_columns.h>
#include <catboost/cuda/gpu_data/kernel/gpu_final_ctr.cuh>
#include <catboost/cuda/gpu_data/kernel/gpu_input_factorize.cuh>
#include <catboost/cuda/gpu_data/kernel/gpu_input_utils.cuh>
#include <catboost/cuda/gpu_data/kernel/gpu_input_targets.cuh>
#include <cuda_runtime_api.h>
#endif


using namespace NCB;


void TOwnedOnlineCtr::DropEmptyData() {
    TVector<TProjection> emptyProjections;
    for (auto& projCtr : Data) {
        if (projCtr.second.Feature.empty()) {
            emptyProjections.emplace_back(projCtr.first);
        }
    }
    for (const auto& proj : emptyProjections) {
        Data.erase(proj);
    }
}


TConstArrayRef<ui8> TPrecomputedOnlineCtr::GetData(const TCtr& ctr, ui32 datasetIdx) const {
    Y_ASSERT(ctr.Projection.IsSingleCatFeature());
    const TOnlineCtrIdx onlineCtrIdx{
        SafeIntegerCast<i32>(ctr.Projection.CatFeatures[0]),
        ctr.CtrIdx,
        ctr.TargetBorderIdx,
        ctr.PriorIdx
    };

    const TQuantizedObjectsDataProvider& dataProvider
        = (datasetIdx == 0) ? *(Data.DataProviders.Learn) : *(Data.DataProviders.Test[datasetIdx - 1]);

    const IQuantizedFloatValuesHolder& column
        = **(dataProvider.GetNonPackedFloatFeature(Data.Meta.OnlineCtrIdxToFeatureIdx.at(onlineCtrIdx)));

    const TQuantizedFloatValuesHolder& columnImpl
        = dynamic_cast<const NCB::TQuantizedFloatValuesHolder&>(column);

    Y_ASSERT(columnImpl.GetBitsPerKey() == 8);
    TConstPtrArraySubset<ui8> arraySubset = columnImpl.GetArrayData<ui8>();
    Y_ASSERT(std::holds_alternative<TFullSubset<ui32>>(*arraySubset.GetSubsetIndexing()));
    return TConstArrayRef<ui8>(*arraySubset.GetSrc(), dataProvider.GetObjectCount());
}

namespace {
struct TBucketsView {
    size_t MaxElem = 0;
    size_t BorderCount = 0;
    TAtomicSharedPtr<TVector<ui8>> DataPtr;
    TArrayRef<int> Data;
    TArrayRef<int> BucketData;
    NCB::TScratchCache& ScratchCacheRef;

public:
    TBucketsView(size_t maxElem, size_t borderCount, NCB::TScratchCache* scratchCache)
        : MaxElem(maxElem)
        , BorderCount(borderCount)
        , DataPtr(scratchCache->GetScratchBlob())
        , ScratchCacheRef(*scratchCache)
    {
        Data = PrepareScratchBlob<int>(MaxElem * (BorderCount + 1), DataPtr.Get());
        BucketData = MakeArrayRef<int>(Data.data() + MaxElem, MaxElem * BorderCount);
    }
    ~TBucketsView() {
        ScratchCacheRef.ReleaseScratchBlob(DataPtr);
    }
    inline int& GetTotal(size_t i) {
        Y_ASSERT(i < MaxElem);
        return Data[i];
    }
    inline TArrayRef<int> GetBorders(size_t i) {
        Y_ASSERT(i < MaxElem);
        return TArrayRef<int>(BucketData.data() + BorderCount * i, BorderCount);
    }
};
}

void CalcNormalization(const TVector<float>& priors, TVector<float>* shift, TVector<float>* norm) {
    shift->yresize(priors.size());
    norm->yresize(priors.size());
    for (int i = 0; i < priors.ysize(); ++i) {
        float prior = priors[i];
        float left = Min(0.0f, prior);
        float right = Max(1.0f, prior);
        (*shift)[i] = -left;
        (*norm)[i] = (right - left);
    }
}


static void UpdateGoodCount(int curCount, ECtrType ctrType, int* goodCount) {
    if (ctrType == ECtrType::Buckets) {
        *goodCount = curCount;
    } else {
        *goodCount -= curCount;
    }
}

namespace {
    // Blocked 2-stage calculation director
    class TBlockedCalcer {
    public:
        explicit TBlockedCalcer(int blockSize)
            : BlockSize(blockSize) {
        }
        template <typename TCalc1, typename TCalc2>
        void Calc(TCalc1 calc1, TCalc2 calc2, int datasetIdx, int docCount) {
            for (int blockStart = 0; blockStart < docCount; blockStart += BlockSize) {
                const int nextBlockStart = Min<int>(docCount, blockStart + BlockSize);
                calc1(blockStart, nextBlockStart, datasetIdx);
                calc2(blockStart, nextBlockStart, datasetIdx);
            }
        }

    private:
        const int BlockSize;
    };
}

static void CalcOnlineCTRClasses(
    const TVector<size_t>& testOffsets,
    TConstArrayRef<ui64> enumeratedCatFeatures,
    size_t leafCount,
    const TVector<int>& permutedTargetClass,
    int targetClassesCount,
    int targetBorderCount,
    const TVector<float>& priors,
    int ctrBorderCount,
    ECtrType ctrType,
    ui32 ctrIdx,
    NCB::TScratchCache* scratchCache,
    IOnlineCtrProjectionDataWriter* writer) {

    TVector<float> shift;
    TVector<float> norm;
    CalcNormalization(priors, &shift, &norm);

    // ensure blocks have reasonable size
    const int blockSize = (1000 + targetBorderCount - 1) / targetBorderCount + 100;
    TVector<int> totalCountByDoc(blockSize);
    TVector<TVector<int>> goodCountByBorderByDoc(targetBorderCount, TVector<int>(blockSize));
    TBucketsView bv(leafCount, targetClassesCount, scratchCache);

    auto calcGoodCounts = [&](int blockStart, int nextBlockStart, int datasetIdx) {
        auto docOffset = datasetIdx ? testOffsets[datasetIdx - 1] : 0;
        for (int docId = blockStart; docId < nextBlockStart; ++docId) {
            const auto elemId = enumeratedCatFeatures[docOffset + docId];

            int goodCount = totalCountByDoc[docId - blockStart] = bv.GetTotal(elemId);
            auto bordersData = bv.GetBorders(elemId);
            for (int border = 0; border < targetBorderCount; ++border) {
                UpdateGoodCount(bordersData[border], ctrType, &goodCount);
                goodCountByBorderByDoc[border][docId - blockStart] = goodCount;
            }

            if (datasetIdx == 0) {
                ++bordersData[permutedTargetClass[docId]];
                ++bv.GetTotal(elemId);
            }
        }
    };

    auto calcCTRs = [&](int blockStart, int nextBlockStart, int datasetIdx) {
        for (int border = 0; border < targetBorderCount; ++border) {
            for (int prior = 0; prior < priors.ysize(); ++prior) {
                const float priorX = priors[prior];
                const float shiftX = shift[prior];
                const float normX = norm[prior];
                const int* goodCountData = goodCountByBorderByDoc[border].data();
                ui8* featureData = writer->GetDataBuffer(ctrIdx, border, prior, datasetIdx).data();
                for (int docId = blockStart; docId < nextBlockStart; ++docId) {
                    featureData[docId] = CalcCTR(
                        goodCountData[docId - blockStart],
                        totalCountByDoc[docId - blockStart],
                        priorX,
                        shiftX,
                        normX,
                        ctrBorderCount);
                }
            }
        }
    };

    TBlockedCalcer calcer(blockSize);
    const size_t learnSampleCount = testOffsets[0];
    calcer.Calc(calcGoodCounts, calcCTRs, 0, learnSampleCount);
    for (size_t testIdx = 0; testIdx < testOffsets.size() - 1; ++testIdx) {
        const size_t testSampleCount = testOffsets[testIdx + 1] - testOffsets[testIdx];
        calcer.Calc(calcGoodCounts, calcCTRs, testIdx + 1, testSampleCount);
    }
}

static void CalcStatsForEachBlock(
    const NPar::ILocalExecutor::TExecRangeParams& ctrParallelizationParams,
    TConstArrayRef<ui64> enumeratedCatFeatures,
    TConstArrayRef<int> permutedTargetClass,
    NPar::ILocalExecutor* localExecutor,
    TArrayRef<TVector<TCtrHistory>> perBlockCtrs
) {
    const int blockCount = ctrParallelizationParams.GetBlockCount();
    const int blockSize = ctrParallelizationParams.GetBlockSize();
    localExecutor->ExecRange(
        [&] (int blockIdx) {
            const int blockStart = blockSize * blockIdx;
            const int nextBlockStart = Min(blockStart + blockSize, ctrParallelizationParams.LastId);
            TArrayRef<TCtrHistory> blockCtrsRef(perBlockCtrs[blockIdx]);
            Fill(blockCtrsRef.begin(), blockCtrsRef.end(), TCtrHistory{0, 0});
            for (int docIdx : xrange(blockStart, nextBlockStart)) {
                ++blockCtrsRef[enumeratedCatFeatures[docIdx]].N[permutedTargetClass[docIdx]];
            }
        },
        0,
        blockCount,
        NPar::TLocalExecutor::WAIT_COMPLETE
    );
}

static void SumCtrsFromBlocks(
    const NPar::ILocalExecutor::TExecRangeParams& valueBlockParams,
    NPar::ILocalExecutor* localExecutor,
    TArrayRef<TVector<TCtrHistory>> perBlockCtrs,
    TArrayRef<TCtrHistory> ctrs
) {
    const int blockCount = valueBlockParams.GetBlockCount();
    const int blockSize = valueBlockParams.GetBlockSize();
    localExecutor->ExecRange(
        [&] (int blockIdx) {
            const int blockStart = blockIdx * blockSize;
            const int nextBlockStart = Min<int>(blockStart + blockSize, valueBlockParams.LastId);
            Fill(ctrs.data() + blockStart, ctrs.data() + nextBlockStart, TCtrHistory{0, 0});
            for (auto& blockCtrs : perBlockCtrs) {
                TArrayRef<TCtrHistory> blockCtrsRef(blockCtrs);
                for (int idx : xrange(blockStart, nextBlockStart)) {
                    const TCtrHistory ctrOffset = blockCtrsRef[idx];
                    blockCtrsRef[idx] = ctrs[idx];
                    ctrs[idx].N[0] += ctrOffset.N[0];
                    ctrs[idx].N[1] += ctrOffset.N[1];
                }
            }
        },
        0,
        blockCount,
        NPar::TLocalExecutor::WAIT_COMPLETE
    );
}

static void CalcQuantizedCtrs(
    const NPar::ILocalExecutor::TExecRangeParams& ctrParallelizationParams,
    TConstArrayRef<ui64> enumeratedCatFeatures,
    TConstArrayRef<int> permutedTargetClass,
    TConstArrayRef<float> priors,
    TConstArrayRef<float> shifts,
    TConstArrayRef<float> norms,
    int ctrBorderCount,
    NPar::ILocalExecutor* localExecutor,
    NCB::TScratchCache* scratchCache,
    TArrayRef<TVector<TCtrHistory>> perBlockCtrs,
    ui32 ctrIdx,
    IOnlineCtrProjectionDataWriter* writer
) {
    constexpr int BlockSize = 1000;
    TBlockedCalcer calcer(BlockSize);

    const int blockCount = ctrParallelizationParams.GetBlockCount();
    const int blockSize = ctrParallelizationParams.GetBlockSize();
    const int docCount = ctrParallelizationParams.LastId;
    localExecutor->ExecRange(
        [&] (int blockIdx) {
            const TArrayRef<TCtrHistory> ctrArrSimple(perBlockCtrs[blockIdx]);
            auto totalCountPtr = scratchCache->GetScratchBlob();
            Y_DEFER { scratchCache->ReleaseScratchBlob(totalCountPtr); };
            auto totalCount = PrepareScratchBlob<int>(2 * BlockSize, totalCountPtr.Get());
            auto goodCount = totalCount.data() + BlockSize;

            int docOffset = blockSize * blockIdx;
            auto calcGoodCount = [&, totalCount = totalCount](int blockStart, int nextBlockStart, int /*datasetIdx*/) {
                for (int docIdx : xrange(blockStart, nextBlockStart)) {
                    auto& elem = ctrArrSimple[enumeratedCatFeatures[docOffset + docIdx]].N;
                    goodCount[docIdx - blockStart] = elem[1];
                    totalCount[docIdx - blockStart] = elem[0] + elem[1];
                    ++elem[permutedTargetClass[docOffset + docIdx]];
                }
            };

            auto calcCtrs = [&, totalCount = totalCount](int blockStart, int nextBlockStart, int datasetIdx) {
                for (int priorIdx : xrange(priors.ysize())) {
                    const float prior = priors[priorIdx];
                    const float shift = shifts[priorIdx];
                    const float norm = norms[priorIdx];
                    const int borderCount = ctrBorderCount;
                    ui8* featureData = writer->GetDataBuffer(
                        ctrIdx,
                        /*targetBorderIdx*/ 0,
                        priorIdx,
                        datasetIdx).data();
                    for (auto docIdx : xrange(blockStart, nextBlockStart)) {
                        featureData[docOffset + docIdx] = CalcCTR(
                            goodCount[docIdx - blockStart],
                            totalCount[docIdx - blockStart],
                            prior,
                            shift,
                            norm,
                            borderCount);
                    }
                }
            };

            calcer.Calc(
                calcGoodCount,
                calcCtrs,
                /*datasetIdx*/ 0,
                Min(blockSize, docCount - blockSize * blockIdx));
        },
        0,
        blockCount,
        NPar::TLocalExecutor::WAIT_COMPLETE
    );
}

static void CalcOnlineCTRSimple(
    const TVector<size_t>& testOffsets,
    TConstArrayRef<ui64> enumeratedCatFeatures,
    size_t uniqueValuesCount,
    const TVector<int>& permutedTargetClass,
    const TVector<float>& priors,
    int ctrBorderCount,
    ui32 ctrIdx,
    NPar::ILocalExecutor* localExecutor,
    NCB::TScratchCache* scratchCache,
    IOnlineCtrProjectionDataWriter* writer) {

    const int learnSampleCount = testOffsets[0];
    NPar::ILocalExecutor::TExecRangeParams ctrParallelizationParams(0, learnSampleCount);
    ctrParallelizationParams.SetBlockCount(localExecutor->GetThreadCount() + 1);

    const int bigBlockCount = ctrParallelizationParams.GetBlockCount();
    TVector<TVector<TCtrHistory>> perBlockCtrs;
    ResizeRank2(bigBlockCount, uniqueValuesCount, perBlockCtrs);
    CalcStatsForEachBlock(ctrParallelizationParams, enumeratedCatFeatures, permutedTargetClass, localExecutor, perBlockCtrs);

    NPar::ILocalExecutor::TExecRangeParams valueBlockParams(0, uniqueValuesCount);
    valueBlockParams.SetBlockSize(1 + 1000 / (localExecutor->GetThreadCount() + 1));

    TVector<TCtrHistory> ctrsForTest;
    ctrsForTest.yresize(uniqueValuesCount);
    SumCtrsFromBlocks(valueBlockParams, localExecutor, perBlockCtrs, ctrsForTest);

    TVector<float> shifts;
    TVector<float> norms;
    CalcNormalization(priors, &shifts, &norms);

    CalcQuantizedCtrs(
        ctrParallelizationParams,
        enumeratedCatFeatures,
        permutedTargetClass,
        priors,
        shifts,
        norms,
        ctrBorderCount,
        localExecutor,
        scratchCache,
        perBlockCtrs,
        ctrIdx,
        writer);

    constexpr int BlockSize = 1000;
    const TArrayRef<TCtrHistory> ctrArrSimple(ctrsForTest);
    auto totalCountPtr = scratchCache->GetScratchBlob();
    Y_DEFER { scratchCache->ReleaseScratchBlob(totalCountPtr); };
    auto totalCount = NCB::PrepareScratchBlob<int>(2 * BlockSize, totalCountPtr.Get());
    auto goodCount = totalCount.data() + BlockSize;

    auto calcGoodCount = [&, totalCount = totalCount](int blockStart, int nextBlockStart, int datasetIdx) {
        Y_ASSERT(datasetIdx >= 1);
        auto docOffset = testOffsets[datasetIdx - 1];
        for (int docIdx : xrange(blockStart, nextBlockStart)) {
            auto& elem = ctrArrSimple[enumeratedCatFeatures[docOffset + docIdx]].N;
            goodCount[docIdx - blockStart] = elem[1];
            totalCount[docIdx - blockStart] = elem[0] + elem[1];
        }
    };

    auto calcCtrs = [&, totalCount = totalCount](int blockStart, int nextBlockStart, int datasetIdx) {
        for (int priorIdx : xrange(priors.ysize())) {
            const float prior = priors[priorIdx];
            const float shift = shifts[priorIdx];
            const float norm = norms[priorIdx];
            ui8* featureData = writer->GetDataBuffer(
                ctrIdx,
                /*targetBorderIdx*/ 0,
                priorIdx,
                datasetIdx).data();
            for (int docIdx : xrange(blockStart, nextBlockStart)) {
                featureData[docIdx] = CalcCTR(
                    goodCount[docIdx - blockStart],
                    totalCount[docIdx - blockStart],
                    prior,
                    shift,
                    norm,
                    ctrBorderCount);
            }
        }
    };

    TBlockedCalcer calcer(BlockSize);

    for (size_t testIdx = 0; testIdx < testOffsets.size() - 1; ++testIdx) {
        const size_t testSampleCount = testOffsets[testIdx + 1] - testOffsets[testIdx];
        calcer.Calc(calcGoodCount, calcCtrs, testIdx + 1, testSampleCount);
    }
}

static void CalcOnlineCTRMean(
    const TVector<size_t>& testOffsets,
    TConstArrayRef<ui64> enumeratedCatFeatures,
    size_t leafCount,
    const TVector<int>& permutedTargetClass,
    int targetBorderCount,
    const TVector<float>& priors,
    int ctrBorderCount,
    ui32 ctrIdx,
    NCB::TScratchCache* scratchCache,
    IOnlineCtrProjectionDataWriter* writer) {

    TVector<float> shift;
    TVector<float> norm;
    CalcNormalization(priors, &shift, &norm);

    const int blockSize = 1000;
    TVector<float> sum(blockSize);
    TVector<int> count(blockSize);
    auto ctrArrMeanPtr = scratchCache->GetScratchBlob();
    Y_DEFER { scratchCache->ReleaseScratchBlob(ctrArrMeanPtr); };
    auto ctrArrMean = NCB::PrepareScratchBlob<TCtrMeanHistory>(leafCount, ctrArrMeanPtr.Get());

    auto calcCount = [&](int blockStart, int nextBlockStart, int datasetIdx) {
        auto docOffset = datasetIdx == 0 ? 0 : testOffsets[datasetIdx - 1];
        for (int docId = blockStart; docId < nextBlockStart; ++docId) {
            TCtrMeanHistory& elem = ctrArrMean[enumeratedCatFeatures[docOffset + docId]];
            sum[docId - blockStart] = elem.Sum;
            count[docId - blockStart] = elem.Count;
            if (docOffset == 0) {
                elem.Add(static_cast<float>(permutedTargetClass[docId]) / targetBorderCount);
            }
        }
    };

    auto calcCTRs = [&](int blockStart, int nextBlockStart, int datasetIdx) {
        for (int prior = 0; prior < priors.ysize(); ++prior) {
            const float priorX = priors[prior];
            const float shiftX = shift[prior];
            const float normX = norm[prior];
            ui8* featureData = writer->GetDataBuffer(
                ctrIdx,
                /*targetBorderIdx*/ 0,
                prior,
                datasetIdx).data();
            for (int docId = blockStart; docId < nextBlockStart; ++docId) {
                featureData[docId] = CalcCTR(
                    sum[docId - blockStart],
                    count[docId - blockStart],
                    priorX,
                    shiftX,
                    normX,
                    ctrBorderCount);
            }
        }
    };

    TBlockedCalcer calcer(blockSize);
    const size_t learnSampleCount = testOffsets[0];
    calcer.Calc(calcCount, calcCTRs, 0, learnSampleCount);
    for (size_t testIdx = 0; testIdx < testOffsets.size() - 1; ++testIdx) {
        const size_t testSampleCount = testOffsets[testIdx + 1] - testOffsets[testIdx];
        calcer.Calc(calcCount, calcCTRs, testIdx + 1, testSampleCount);
    }
}

static void CalcOnlineCTRCounter(
    const TVector<size_t>& testOffsets,
    const TVector<int>& counterCTRTotal,
    TConstArrayRef<ui64> enumeratedCatFeatures,
    int denominator,
    const TVector<float>& priors,
    int ctrBorderCount,
    ui32 ctrIdx,
    NCB::TScratchCache* scratchCache,
    IOnlineCtrProjectionDataWriter* writer) {

    TVector<float> shift;
    TVector<float> norm;
    CalcNormalization(priors, &shift, &norm);

    constexpr int BlockSize = 1000;
    auto ctrTotalPtr = scratchCache->GetScratchBlob();
    Y_DEFER { scratchCache->ReleaseScratchBlob(ctrTotalPtr); };
    auto ctrTotal = PrepareScratchBlob<int>(BlockSize, ctrTotalPtr.Get());

    auto calcTotal = [&, ctrTotal = ctrTotal](int blockStart, int nextBlockStart, int datasetIdx) {
        auto docOffset = datasetIdx ? testOffsets[datasetIdx - 1] : 0;
        for (int docId = blockStart; docId < nextBlockStart; ++docId) {
            const auto elemId = enumeratedCatFeatures[docOffset + docId];
            ctrTotal[docId - blockStart] = counterCTRTotal[elemId];
        }
    };

    auto calcCTRs = [&, ctrTotal = ctrTotal](int blockStart, int nextBlockStart, int datasetIdx) {
        for (int prior = 0; prior < priors.ysize(); ++prior) {
            const float priorX = priors[prior];
            const float shiftX = shift[prior];
            const float normX = norm[prior];
            const int totalCount = denominator;
            const int borderCount = ctrBorderCount;
            ui8* featureData = writer->GetDataBuffer(
                ctrIdx,
                /*targetBorderIdx*/ 0,
                prior,
                datasetIdx).data();
            for (int docId = blockStart; docId < nextBlockStart; ++docId) {
                featureData[docId] = CalcCTR(
                    ctrTotal[docId - blockStart],
                    totalCount,
                    priorX,
                    shiftX,
                    normX,
                    borderCount);
            }
        }
    };

    TBlockedCalcer calcer(BlockSize);
    const size_t learnSampleCount = testOffsets[0];
    calcer.Calc(calcTotal, calcCTRs, 0, learnSampleCount);
    for (size_t testIdx = 0; testIdx < testOffsets.size() - 1; ++testIdx) {
        const size_t testSampleCount = testOffsets[testIdx + 1] - testOffsets[testIdx];
        calcer.Calc(calcTotal, calcCTRs, testIdx + 1, testSampleCount);
    }
}

static inline void CountOnlineCTRTotal(
    TConstArrayRef<ui64> hashArr,
    int sampleCount,
    TVector<int>* counterCTRTotal) {

    for (int sampleIdx = 0; sampleIdx < sampleCount; ++sampleIdx) {
        const auto elemId = hashArr[sampleIdx];
        ++(*counterCTRTotal)[elemId];
    }
}

template <typename TValueType>
static void CopyCatColumnToHash(
    const IQuantizedCatValuesHolder& catColumn,
    const TFeaturesArraySubsetIndexing& featuresSubsetIndexing,
    NPar::ILocalExecutor* localExecutor,
    TValueType* hashArrView
) {
    TCloningParams cloningParams;
    cloningParams.SubsetIndexing = &featuresSubsetIndexing;
    auto cloned = catColumn.CloneWithNewSubsetIndexing(
        cloningParams,
        localExecutor
    );
    dynamic_cast<const IQuantizedCatValuesHolder*>(cloned.Get())->ParallelForEachBlock(
        localExecutor,
        [hashArrView] (size_t blockStartIdx, auto blockView) {
            auto hashArrViewPtr = hashArrView + blockStartIdx;
            for (auto i : xrange(blockView.size())) {
                hashArrViewPtr[i] = blockView[i] + 1;
            }
        },
        512);
}


void ComputeOnlineCTRs(
    const TTrainingDataProviders& data,
    const TProjection& proj,
    const TCtrHelper& ctrHelper,
    const NCB::TFeaturesArraySubsetIndexing& foldLearnPermutationFeaturesSubset,
    const TVector<TVector<int>>& foldLearnTargetClass,
    const TVector<int>& foldTargetClassesCount,
    const NCatboostOptions::TCatFeatureParams& catFeatureParams,
    NPar::ILocalExecutor* localExecutor,
    NCB::TScratchCache* scratchCache,
    IOnlineCtrProjectionDataWriter* writer) {

    const auto& ctrInfo = ctrHelper.GetCtrInfo(proj);
    writer->AllocateData(ctrInfo.size());
    size_t learnSampleCount = data.Learn->GetObjectCount();
    const TVector<size_t>& testOffsets = data.CalcTestOffsets();
    size_t totalSampleCount = learnSampleCount + data.GetTestSampleCount();

    const auto& quantizedFeaturesInfo = *data.Learn->ObjectsData->GetQuantizedFeaturesInfo();

    auto hashArrPtr = scratchCache->GetScratchBlob();
    Y_DEFER { scratchCache->ReleaseScratchBlob(hashArrPtr); };
    auto hashArr = NCB::GrowScratchBlob<ui64>(totalSampleCount, hashArrPtr.Get());
    auto rehashHashVal = scratchCache->GetScratchHash();
    Y_DEFER { scratchCache->ReleaseScratchHash(rehashHashVal); };

    if (proj.IsSingleCatFeature()) {
        // Shortcut for simple ctrs
        auto catFeatureIdx = TCatFeatureIdx((ui32)proj.CatFeatures[0]);

        TArrayRef<ui64> hashArrView(hashArr);
        if (learnSampleCount > 0) {
            CopyCatColumnToHash(
                **data.Learn->ObjectsData->GetCatFeature(*catFeatureIdx),
                foldLearnPermutationFeaturesSubset,
                localExecutor,
                hashArrView.data()
            );
        }
        for (size_t docOffset = learnSampleCount, testIdx = 0;
             docOffset < totalSampleCount && testIdx < data.Test.size();
             ++testIdx)
        {
            const size_t testSampleCount = data.Test[testIdx]->GetObjectCount();
            CopyCatColumnToHash(
                **data.Test[testIdx]->ObjectsData->GetCatFeature(*catFeatureIdx),
                data.Test[testIdx]->ObjectsData->GetFeaturesArraySubsetIndexing(),
                localExecutor,
                hashArrView.data() + docOffset
            );
            docOffset += testSampleCount;
        }
        rehashHashVal->MakeEmpty(
            quantizedFeaturesInfo.GetUniqueValuesCounts(TCatFeatureIdx(proj.CatFeatures[0])).OnLearnOnly
        );
    } else {
        ParallelFill<ui64>(/*fillValue*/0, /*blockSize*/Nothing(), localExecutor, MakeArrayRef(hashArr));
        CalcHashes(
            proj,
            *data.Learn->ObjectsData,
            foldLearnPermutationFeaturesSubset,
            nullptr,
            hashArr.begin(),
            hashArr.begin() + learnSampleCount,
            localExecutor);
        for (size_t docOffset = learnSampleCount, testIdx = 0;
             docOffset < totalSampleCount && testIdx < data.Test.size();
             ++testIdx)
        {
            const size_t testSampleCount = data.Test[testIdx]->GetObjectCount();
            CalcHashes(
                proj,
                *data.Test[testIdx]->ObjectsData,
                data.Test[testIdx]->ObjectsData->GetFeaturesArraySubsetIndexing(),
                nullptr,
                hashArr.begin() + docOffset,
                hashArr.begin() + docOffset + testSampleCount,
                localExecutor);
            docOffset += testSampleCount;
        }
        size_t approxBucketsCount = 1;
        for (auto cf : proj.CatFeatures) {
            approxBucketsCount *= quantizedFeaturesInfo.GetUniqueValuesCounts(TCatFeatureIdx(cf)).OnLearnOnly;
            if (approxBucketsCount > learnSampleCount) {
                break;
            }
        }
        rehashHashVal->MakeEmpty(Min(learnSampleCount, approxBucketsCount));
    }
    ui64 topSize = catFeatureParams.CtrLeafCountLimit;
    if (proj.IsSingleCatFeature() && catFeatureParams.StoreAllSimpleCtrs) {
        topSize = Max<ui64>();
    }
    auto leafCount = ComputeReindexHash(
        topSize,
        rehashHashVal.Get(),
        hashArr.begin(),
        hashArr.begin() + learnSampleCount);

    TOnlineCtrUniqValuesCounts uniqValuesCounts;
    uniqValuesCounts.CounterCount = uniqValuesCounts.Count = leafCount;

    for (size_t docOffset = learnSampleCount, testIdx = 0;
         docOffset < totalSampleCount && testIdx < data.Test.size();
         ++testIdx)
    {
        const size_t testSampleCount = data.Test[testIdx]->GetObjectCount();
        leafCount = UpdateReindexHash(
            rehashHashVal.Get(),
            hashArr.begin() + docOffset,
            hashArr.begin() + docOffset + testSampleCount);
        docOffset += testSampleCount;
    }

    TVector<int> counterCTRTotal;
    int counterCTRDenominator = 0;
    if (AnyOf(
            ctrInfo.begin(),
            ctrInfo.end(),
            [] (const auto& info) { return info.Type == ECtrType::Counter; }))
    {
        counterCTRTotal.resize(leafCount);
        int sampleCount = learnSampleCount;
        if (catFeatureParams.CounterCalcMethod == ECounterCalc::Full) {
            uniqValuesCounts.CounterCount = leafCount;
            sampleCount = hashArr.size();
        }
        CountOnlineCTRTotal(hashArr, sampleCount, &counterCTRTotal);
        counterCTRDenominator = *MaxElement(counterCTRTotal.begin(), counterCTRTotal.end());
    }
    writer->SetUniqValuesCounts(uniqValuesCounts);

    localExecutor->ExecRange(
        [&] (ui32 ctrIdx) {
            const ECtrType ctrType = ctrInfo[ctrIdx].Type;
            const ui32 classifierId = ctrInfo[ctrIdx].TargetClassifierIdx;
            int targetClassesCount = foldTargetClassesCount[classifierId];

            const ui32 targetBorderCount = GetTargetBorderCount(ctrInfo[ctrIdx], targetClassesCount);
            const ui32 ctrBorderCount = ctrInfo[ctrIdx].BorderCount;
            const auto& priors = ctrInfo[ctrIdx].Priors;
            writer->AllocateCtrData(ctrIdx, targetBorderCount, priors.size());

            if (ctrType == ECtrType::Borders && targetClassesCount == SIMPLE_CLASSES_COUNT) {
                CalcOnlineCTRSimple(
                    testOffsets,
                    hashArr,
                    leafCount,
                    foldLearnTargetClass[classifierId],
                    priors,
                    ctrBorderCount,
                    ctrIdx,
                    localExecutor,
                    scratchCache,
                    writer);

            } else if (ctrType == ECtrType::BinarizedTargetMeanValue) {
                CalcOnlineCTRMean(
                    testOffsets,
                    hashArr,
                    leafCount,
                    foldLearnTargetClass[classifierId],
                    targetClassesCount - 1,
                    priors,
                    ctrBorderCount,
                    ctrIdx,
                    scratchCache,
                    writer);

            } else if (ctrType == ECtrType::Buckets ||
                    (ctrType == ECtrType::Borders && targetClassesCount > SIMPLE_CLASSES_COUNT)) {
                CalcOnlineCTRClasses(
                    testOffsets,
                    hashArr,
                    leafCount,
                    foldLearnTargetClass[classifierId],
                    targetClassesCount,
                    GetTargetBorderCount(ctrInfo[ctrIdx], targetClassesCount),
                    priors,
                    ctrBorderCount,
                    ctrType,
                    ctrIdx,
                    scratchCache,
                    writer);
            } else {
                Y_ASSERT(ctrType == ECtrType::Counter);
                CalcOnlineCTRCounter(
                    testOffsets,
                    counterCTRTotal,
                    hashArr,
                    counterCTRDenominator,
                    priors,
                    ctrBorderCount,
                    ctrIdx,
                    scratchCache,
                    writer);
            }
        },
        0,
        ctrInfo.ysize(),
        NPar::TLocalExecutor::WAIT_COMPLETE);
}


class TOnlineCtrPerProjectionDataWriter final : public IOnlineCtrProjectionDataWriter {
public:
    TOnlineCtrPerProjectionDataWriter(
        TConstArrayRef<NCB::TIndexRange<size_t>> datasetsObjectRanges,
        TOnlineCtrPerProjectionData* dst
    )
        : DatasetsObjectRanges(datasetsObjectRanges)
        , Dst(dst)
    {}

    void SetUniqValuesCounts(const TOnlineCtrUniqValuesCounts& uniqValuesCounts) override {
        Dst->UniqValuesCounts = uniqValuesCounts;
    }

    void AllocateData(size_t ctrCount) override {
        Dst->Feature.resize(ctrCount);
    }

    // call after AllocateData has been called
    void AllocateCtrData(size_t ctrIdx, size_t targetBorderCount, size_t priorCount) override {
        Dst->Feature[ctrIdx].SetSizes(priorCount, targetBorderCount);
        if (!DatasetsObjectRanges.empty()) {
            auto totalSampleCount = DatasetsObjectRanges.back().End;
            for (auto targetBorderIdx : xrange(targetBorderCount)) {
                for (auto priorIdx : xrange(priorCount)) {
                    Dst->Feature[ctrIdx][targetBorderIdx][priorIdx].yresize(totalSampleCount);
                }
            }
        }
    }

    TArrayRef<ui8> GetDataBuffer(int ctrIdx, int targetBorderIdx, int priorIdx, int datasetIdx) override {
        ui8* allDataPtr = Dst->Feature[ctrIdx][targetBorderIdx][priorIdx].data();
        return TArrayRef<ui8>(
            allDataPtr + DatasetsObjectRanges[datasetIdx].Begin,
            allDataPtr + DatasetsObjectRanges[datasetIdx].End
        );
    }

private:
    TConstArrayRef<NCB::TIndexRange<size_t>> DatasetsObjectRanges;
    TOnlineCtrPerProjectionData* Dst;
};


void ComputeOnlineCTRs(
    const TTrainingDataProviders& data,
    const TFold& fold,
    const TProjection& proj,
    TLearnContext* ctx,
    TOwnedOnlineCtr* onlineCtrStorage) {

    TOnlineCtrPerProjectionDataWriter onlineCtrWriter(
        onlineCtrStorage->DatasetsObjectRanges,
        &(onlineCtrStorage->Data[proj])
    );

    ComputeOnlineCTRs(
        data,
        proj,
        ctx->CtrsHelper,
        fold.LearnPermutationFeaturesSubset,
        fold.LearnTargetClass,
        fold.TargetClassesCount,
        ctx->Params.CatFeatureParams.Get(),
        ctx->LocalExecutor,
        &ctx->ScratchCache,
        &onlineCtrWriter
    );
}

void CalcFinalCtrsImpl(
    const ECtrType ctrType,
    const ui64 ctrLeafCountLimit,
    const TVector<int>& targetClass,
    TConstArrayRef<float> targets,
    const ui32 totalSampleCount,
    int targetClassesCount,
    TVector<ui64>* hashArr,
    TCtrValueTable* result) {

    Y_ASSERT(hashArr->size() == (size_t)totalSampleCount);

    size_t leafCount = 0;
    {
        TDenseHash<ui64, ui32> tmpHash;
        leafCount = ComputeReindexHash(
            ctrLeafCountLimit,
            &tmpHash,
            hashArr->begin(),
            hashArr->begin() + totalSampleCount);
        auto hashIndexBuilder = result->GetIndexHashBuilder(leafCount);
        for (const auto& kv : tmpHash) {
            hashIndexBuilder.SetIndex(kv.first, kv.second);
        }
    }

    TArrayRef<int> ctrIntArray;
    TArrayRef<TCtrMeanHistory> ctrMean;
    if (ctrType == ECtrType::BinarizedTargetMeanValue || ctrType == ECtrType::FloatTargetMeanValue) {
        ctrMean = result->AllocateBlobAndGetArrayRef<TCtrMeanHistory>(leafCount);
    } else if (ctrType == ECtrType::Counter || ctrType == ECtrType::FeatureFreq) {
        ctrIntArray = result->AllocateBlobAndGetArrayRef<int>(leafCount);
        result->CounterDenominator = 0;
    } else {
        result->TargetClassesCount = targetClassesCount;
        ctrIntArray = result->AllocateBlobAndGetArrayRef<int>(leafCount * targetClassesCount);
    }


    int targetBorderCount = targetClassesCount - 1;
    auto hashArrPtr = hashArr->data();
    for (ui32 z = 0; z < totalSampleCount; ++z) {
        const ui64 elemId = hashArrPtr[z];
        if (ctrType == ECtrType::BinarizedTargetMeanValue) {
            TCtrMeanHistory& elem = ctrMean[elemId];
            elem.Add(static_cast<float>(targetClass[z]) / targetBorderCount);
        } else if (ctrType == ECtrType::Counter || ctrType == ECtrType::FeatureFreq) {
            ++ctrIntArray[elemId];
        } else if (ctrType == ECtrType::FloatTargetMeanValue) {
            TCtrMeanHistory& elem = ctrMean[elemId];
            elem.Add(targets[z]);
        } else {
            TArrayRef<int> elem = MakeArrayRef(
                ctrIntArray.data() + targetClassesCount * elemId,
                targetClassesCount);
            ++elem[targetClass[z]];
        }
    }

    if (ctrType == ECtrType::Counter) {
        result->CounterDenominator = *MaxElement(ctrIntArray.begin(), ctrIntArray.end());
    }
    if (ctrType == ECtrType::FeatureFreq) {
        result->CounterDenominator = static_cast<int>(totalSampleCount);
    }
}

#if defined(HAVE_CUDA)
namespace {
    template <class T>
    class TCudaDeviceArray final {
    public:
        TCudaDeviceArray() = default;

        explicit TCudaDeviceArray(ui64 size) {
            if (size) {
                CUDA_SAFE_CALL(cudaMalloc(&Ptr, static_cast<size_t>(size) * sizeof(T)));
            }
        }

        TCudaDeviceArray(const TCudaDeviceArray&) = delete;
        TCudaDeviceArray& operator=(const TCudaDeviceArray&) = delete;

        TCudaDeviceArray(TCudaDeviceArray&& other) noexcept
            : Ptr(other.Ptr)
        {
            other.Ptr = nullptr;
        }

        TCudaDeviceArray& operator=(TCudaDeviceArray&& other) noexcept {
            if (this != &other) {
                Reset();
                Ptr = other.Ptr;
                other.Ptr = nullptr;
            }
            return *this;
        }

        ~TCudaDeviceArray() {
            Reset();
        }

        T* Get() const noexcept {
            return Ptr;
        }

    private:
        void Reset() noexcept {
            if (Ptr) {
                cudaFree(Ptr);
                Ptr = nullptr;
            }
        }

    private:
        T* Ptr = nullptr;
    };

    template <class T>
    static void CopyHostToDevice(const T* srcHost, T* dstDev, ui64 size, cudaStream_t stream) {
        if (size == 0) {
            return;
        }
        NCudaLib::TMemcpyTracker::Instance().RecordMemcpyAsync(
            /*dst*/ dstDev,
            /*src*/ srcHost,
            sizeof(T) * size,
            cudaMemcpyHostToDevice
        );
        CUDA_SAFE_CALL(cudaMemcpyAsync(dstDev, srcHost, sizeof(T) * size, cudaMemcpyHostToDevice, stream));
    }

    template <class T>
    static void CopyDeviceToHost(const T* srcDev, T* dstHost, ui64 size, cudaStream_t stream) {
        if (size == 0) {
            return;
        }
        NCudaLib::TMemcpyTracker::Instance().RecordMemcpyAsync(
            /*dst*/ dstHost,
            /*src*/ srcDev,
            sizeof(T) * size,
            cudaMemcpyDeviceToHost
        );
        CUDA_SAFE_CALL(cudaMemcpyAsync(dstHost, srcDev, sizeof(T) * size, cudaMemcpyDeviceToHost, stream));
    }

    static bool TryCalcFinalCtrsOnGpu(
        const ECtrType ctrType,
        const TProjection& projection,
        const TDatasetDataForFinalCtrs& datasetDataForFinalCtrs,
        const NCB::TPerfectHashedToHashedCatValuesMap& perfectHashedToHashedCatValuesMap,
        int targetBorderClassifierIdx,
        ui64 ctrLeafCountLimit,
        bool storeAllSimpleCtr,
        ECounterCalc counterCalcMethod,
        TCtrValueTable* result
    ) {
        const bool requireGpuFinalCtr = !datasetDataForFinalCtrs.Targets.Defined();
        const auto bail = [&] (TStringBuf reason) -> bool {
            if (requireGpuFinalCtr) {
                CB_ENSURE(
                    false,
                    TStringBuilder()
                        << "Final CTR GPU path failed: " << reason
                        << " (ctrType=" << static_cast<int>(ctrType) << ", projection=" << projection << ")"
                );
            }
            return false;
        };

        if (projection.CatFeatures.empty() && projection.OneHotFeatures.empty()) {
            return bail("projection has no categorical/one-hot features");
        }

        ui32 learnSampleCount = datasetDataForFinalCtrs.Data.Learn->GetObjectCount();
        ui32 totalSampleCount = learnSampleCount;
        if (ctrType == ECtrType::Counter && counterCalcMethod == ECounterCalc::Full) {
            totalSampleCount += datasetDataForFinalCtrs.Data.GetTestSampleCount();
        }

        if (projection.IsSingleCatFeature() && storeAllSimpleCtr) {
            ctrLeafCountLimit = Max<ui64>();
        }

        const auto& learnObjectsDataProvider = *datasetDataForFinalCtrs.Data.Learn->ObjectsData;
        const auto quantizedFeaturesInfo = learnObjectsDataProvider.GetQuantizedFeaturesInfo();
        if (!quantizedFeaturesInfo) {
            return bail("QuantizedFeaturesInfo is null");
        }

        struct TBinFeatureGpuData {
            int FloatFeatureIdx = 0;
            int SplitIdx = 0;
            float BorderValue = 0.0f;
            bool UseRawFloatValues = false;
        };
        TVector<TBinFeatureGpuData> binFeaturesGpu;
        binFeaturesGpu.reserve(projection.BinFeatures.size());

        TVector<const NCB::TGpuExternalCatValuesHolder*> learnCatHolders;
        learnCatHolders.reserve(projection.CatFeatures.size());

        i32 deviceId = -1;
        for (const int featureIdx : projection.CatFeatures) {
            const auto& holder = **(learnObjectsDataProvider.GetCatFeature(featureIdx));
            const auto* gpuHolder = dynamic_cast<const NCB::TGpuExternalCatValuesHolder*>(&holder);
            if (!gpuHolder) {
                return bail(TStringBuilder() << "cat feature holder is not GPU-backed (featureIdx=" << featureIdx << ")");
            }
            if (deviceId == -1) {
                deviceId = gpuHolder->GetDeviceId();
            } else if (deviceId != gpuHolder->GetDeviceId()) {
                return bail("multiple devices in categorical features projection");
            }
            learnCatHolders.push_back(gpuHolder);
        }

        for (const auto& feature : projection.OneHotFeatures) {
            const int featureIdx = feature.CatFeatureIdx;
            const auto& holder = **(learnObjectsDataProvider.GetCatFeature(featureIdx));
            const auto* gpuHolder = dynamic_cast<const NCB::TGpuExternalCatValuesHolder*>(&holder);
            if (!gpuHolder) {
                return bail(TStringBuilder() << "one-hot feature holder is not GPU-backed (featureIdx=" << featureIdx << ")");
            }
            if (deviceId == -1) {
                deviceId = gpuHolder->GetDeviceId();
            } else if (deviceId != gpuHolder->GetDeviceId()) {
                return bail("multiple devices in one-hot features projection");
            }
        }

        for (const auto& feature : projection.BinFeatures) {
            const int floatFeatureIdx = feature.FloatFeature;
            const auto& holder = **(learnObjectsDataProvider.GetFloatFeature(floatFeatureIdx));
            const int splitIdx = feature.SplitIdx;
            if (const auto* gpuHolder = dynamic_cast<const NCB::TGpuExternalFloatValuesHolder*>(&holder)) {
                const int featureDeviceId = gpuHolder->GetColumnDesc().DeviceId;
                if (deviceId == -1) {
                    deviceId = featureDeviceId;
                } else if (deviceId != featureDeviceId) {
                    return bail("multiple devices in float bin features projection");
                }

                const auto& borders = quantizedFeaturesInfo->GetBorders(TFloatFeatureIdx(static_cast<ui32>(floatFeatureIdx)));
                if ((splitIdx < 0) || (static_cast<ui32>(splitIdx) >= borders.size())) {
                    return bail("float split index is out of bounds for borders");
                }
                binFeaturesGpu.push_back(TBinFeatureGpuData{floatFeatureIdx, splitIdx, borders[splitIdx], /*UseRawFloatValues*/ true});
            } else if (const auto* qHolder = dynamic_cast<const NCB::TQuantizedFloatValuesHolder*>(&holder)) {
                Y_UNUSED(qHolder);
                if (splitIdx < 0) {
                    return bail("float split index is negative");
                }
                // We'll read the already-quantized 8-bit bins from host and update hashes on device.
                // This is used for projections that depend on non-raw float features (e.g. precomputed).
                binFeaturesGpu.push_back(TBinFeatureGpuData{floatFeatureIdx, splitIdx, 0.0f, /*UseRawFloatValues*/ false});
            } else {
                if (!datasetDataForFinalCtrs.Targets.Defined()) {
                    CB_ENSURE(
                        false,
                        TStringBuilder()
                            << "Final CTR GPU path does not support float feature holder type for bin feature "
                            << "(floatFeatureIdx=" << floatFeatureIdx << ", splitIdx=" << splitIdx << "): "
                            << TypeName(typeid(holder))
                    );
                }
                return false;
            }
        }
        if (deviceId < 0) {
            return bail("could not determine device id for projection");
        }

        NCudaLib::SetDevice(deviceId);
        const cudaStream_t stream = 0;

        const NCB::TGpuInputTargets* learnGpuTargets = nullptr;
        if (const auto* gpuObjects = dynamic_cast<const NCB::TGpuInputQuantizedObjectsDataProvider*>(datasetDataForFinalCtrs.Data.Learn->ObjectsData.Get())) {
            const auto& gpuTargets = gpuObjects->GetGpuTargets();
            if (gpuTargets.TargetCount > 0) {
                for (const auto& t : gpuTargets.Targets) {
                    if (t.DeviceId != deviceId) {
                        return bail("GPU target device id mismatch");
                    }
                }
                learnGpuTargets = &gpuTargets;
            }
        }

        // Copy bin->hashedCatValue maps to device (small, per-feature).
        TVector<TCudaDeviceArray<ui32>> binToHashMapsDev;
        binToHashMapsDev.reserve(projection.CatFeatures.size());
        TVector<const ui32*> binToHashPtrsDev;
        binToHashPtrsDev.reserve(projection.CatFeatures.size());

        for (const int featureIdx : projection.CatFeatures) {
            CB_ENSURE(
                (ui32)featureIdx < perfectHashedToHashedCatValuesMap.size(),
                "perfectHashedToHashedCatValuesMap is out of bounds for cat feature " << featureIdx
            );
            const auto& mapping = perfectHashedToHashedCatValuesMap[featureIdx];
            auto mappingDev = TCudaDeviceArray<ui32>(mapping.size());
            if (!mapping.empty()) {
                CopyHostToDevice(mapping.data(), mappingDev.Get(), mapping.size(), stream);
            }
            binToHashPtrsDev.push_back(mappingDev.Get());
            binToHashMapsDev.push_back(std::move(mappingDev));
        }

        // Build projection hash for learn and (optionally) test datasets on device.
        TCudaDeviceArray<ui64> hashesDev(totalSampleCount);
        if (totalSampleCount) {
            CUDA_SAFE_CALL(cudaMemsetAsync(hashesDev.Get(), 0, static_cast<size_t>(totalSampleCount) * sizeof(ui64), stream));
        }

        ui32 offset = 0;
        for (ui32 f = 0; f < learnCatHolders.size(); ++f) {
            const ui32* bins = learnCatHolders[f]->GetDeviceBinsPtr();
            CB_ENSURE(bins != nullptr, "GPU bins pointer is null for cat feature " << projection.CatFeatures[f]);
            NKernel::UpdateHashesFromCatFeature(
                bins,
                learnSampleCount,
                binToHashPtrsDev[f],
                hashesDev.Get() + offset,
                stream
            );
        }

        for (const auto& feature : binFeaturesGpu) {
            const auto& holder = **(learnObjectsDataProvider.GetFloatFeature(feature.FloatFeatureIdx));
            if (feature.UseRawFloatValues) {
                const auto* gpuHolder = dynamic_cast<const NCB::TGpuExternalFloatValuesHolder*>(&holder);
                CB_ENSURE(gpuHolder != nullptr, "Float feature holder is not GPU-backed");
                const auto& column = gpuHolder->GetColumnDesc();
                CB_ENSURE(column.DeviceId == deviceId, "Unexpected device id mismatch for float feature");
                CB_ENSURE(column.Data != 0, "GPU float feature pointer is null");
                CB_ENSURE(column.StrideBytes > 0, "Invalid stride for GPU float feature");

                TCudaDeviceArray<float> valuesDev(learnSampleCount);
                if (learnSampleCount) {
                    const void* src = reinterpret_cast<const void*>(static_cast<uintptr_t>(column.Data));
                    NKernel::CopyStridedGpuInputToFloat(
                        src,
                        column.StrideBytes,
                        learnSampleCount,
                        static_cast<NKernel::EGpuInputDType>(static_cast<ui8>(column.DType)),
                        valuesDev.Get(),
                        stream
                    );
                    NKernel::UpdateHashesFromFloatSplit(
                        valuesDev.Get(),
                        learnSampleCount,
                        feature.BorderValue,
                        hashesDev.Get() + offset,
                        stream
                    );
                }
            } else {
                const auto* qHolder = dynamic_cast<const NCB::TQuantizedFloatValuesHolder*>(&holder);
                CB_ENSURE(qHolder != nullptr, "Float feature holder is not CPU-quantized");
                CB_ENSURE(qHolder->GetBitsPerKey() == 8, "Only 8-bit quantized floats are supported for final CTR hashing");
                const ui32 threshold = SafeIntegerCast<ui32>(feature.SplitIdx + 1);
                CB_ENSURE(threshold <= 255, "Float split index is too large for 8-bit binarized feature");

                TConstPtrArraySubset<ui8> arraySubset = qHolder->GetArrayData<ui8>();
                CB_ENSURE(
                    std::holds_alternative<TFullSubset<ui32>>(*arraySubset.GetSubsetIndexing()),
                    "Only full subset indexing is supported for CPU-quantized float features in final CTR hashing"
                );
                const ui8* binsHost = *arraySubset.GetSrc();
                CB_ENSURE(binsHost != nullptr, "CPU-quantized float bins pointer is null");

                TCudaDeviceArray<ui8> binsDev(learnSampleCount);
                if (learnSampleCount) {
                    CopyHostToDevice(binsHost, binsDev.Get(), learnSampleCount, stream);
                }
                NKernel::UpdateHashesFromBinarizedSplit(
                    binsDev.Get(),
                    learnSampleCount,
                    static_cast<ui8>(threshold),
                    hashesDev.Get() + offset,
                    stream
                );
            }
        }

        for (const auto& feature : projection.OneHotFeatures) {
            const int featureIdx = feature.CatFeatureIdx;
            const auto& holder = **(learnObjectsDataProvider.GetCatFeature(featureIdx));
            const auto* gpuHolder = dynamic_cast<const NCB::TGpuExternalCatValuesHolder*>(&holder);
            CB_ENSURE(gpuHolder != nullptr, "One-hot feature holder is not GPU-backed");
            CB_ENSURE(gpuHolder->GetDeviceId() == deviceId, "Unexpected device id mismatch for one-hot feature");
            const ui32* bins = gpuHolder->GetDeviceBinsPtr();
            CB_ENSURE(bins != nullptr, "GPU bins pointer is null for one-hot feature " << featureIdx);
            NKernel::UpdateHashesFromOneHotFeature(
                bins,
                learnSampleCount,
                SafeIntegerCast<ui32>(feature.Value),
                hashesDev.Get() + offset,
                stream
            );
        }

        offset += learnSampleCount;

        if (totalSampleCount > learnSampleCount) {
            for (const auto& testDataPtr : datasetDataForFinalCtrs.Data.Test) {
                const auto& testObjectsDataProvider = *testDataPtr->ObjectsData;
                const ui32 testCount = testDataPtr->GetObjectCount();

                for (ui32 f = 0; f < projection.CatFeatures.size(); ++f) {
                    const int featureIdx = projection.CatFeatures[f];
                    const auto& holder = **(testObjectsDataProvider.GetCatFeature(featureIdx));
                    const auto* gpuHolder = dynamic_cast<const NCB::TGpuExternalCatValuesHolder*>(&holder);
                    if (!gpuHolder || gpuHolder->GetDeviceId() != deviceId) {
                        return false;
                    }
                    const ui32* bins = gpuHolder->GetDeviceBinsPtr();
                    CB_ENSURE(bins != nullptr, "GPU bins pointer is null for test cat feature " << featureIdx);
                    NKernel::UpdateHashesFromCatFeature(
                        bins,
                        testCount,
                        binToHashPtrsDev[f],
                        hashesDev.Get() + offset,
                        stream
                    );
                }

                for (const auto& feature : binFeaturesGpu) {
                    const auto& holder = **(testObjectsDataProvider.GetFloatFeature(feature.FloatFeatureIdx));
                    if (feature.UseRawFloatValues) {
                        const auto* gpuHolder = dynamic_cast<const NCB::TGpuExternalFloatValuesHolder*>(&holder);
                        if (!gpuHolder || (gpuHolder->GetColumnDesc().DeviceId != deviceId)) {
                            return false;
                        }
                        const auto& column = gpuHolder->GetColumnDesc();
                        if ((column.Data == 0) || (column.StrideBytes == 0)) {
                            return false;
                        }
                        TCudaDeviceArray<float> valuesDev(testCount);
                        if (testCount) {
                            const void* src = reinterpret_cast<const void*>(static_cast<uintptr_t>(column.Data));
                            NKernel::CopyStridedGpuInputToFloat(
                                src,
                                column.StrideBytes,
                                testCount,
                                static_cast<NKernel::EGpuInputDType>(static_cast<ui8>(column.DType)),
                                valuesDev.Get(),
                                stream
                            );
                            NKernel::UpdateHashesFromFloatSplit(
                                valuesDev.Get(),
                                testCount,
                                feature.BorderValue,
                                hashesDev.Get() + offset,
                                stream
                            );
                        }
                    } else {
                        const auto* qHolder = dynamic_cast<const NCB::TQuantizedFloatValuesHolder*>(&holder);
                        if (!qHolder || (qHolder->GetBitsPerKey() != 8)) {
                            return false;
                        }
                        const ui32 threshold = SafeIntegerCast<ui32>(feature.SplitIdx + 1);
                        if (threshold > 255) {
                            return false;
                        }
                        TConstPtrArraySubset<ui8> arraySubset = qHolder->GetArrayData<ui8>();
                        if (!std::holds_alternative<TFullSubset<ui32>>(*arraySubset.GetSubsetIndexing())) {
                            return false;
                        }
                        const ui8* binsHost = *arraySubset.GetSrc();
                        if (!binsHost) {
                            return false;
                        }
                        TCudaDeviceArray<ui8> binsDev(testCount);
                        if (testCount) {
                            CopyHostToDevice(binsHost, binsDev.Get(), testCount, stream);
                        }
                        NKernel::UpdateHashesFromBinarizedSplit(
                            binsDev.Get(),
                            testCount,
                            static_cast<ui8>(threshold),
                            hashesDev.Get() + offset,
                            stream
                        );
                    }
                }

                for (const auto& feature : projection.OneHotFeatures) {
                    const int featureIdx = feature.CatFeatureIdx;
                    const auto& holder = **(testObjectsDataProvider.GetCatFeature(featureIdx));
                    const auto* gpuHolder = dynamic_cast<const NCB::TGpuExternalCatValuesHolder*>(&holder);
                    if (!gpuHolder || (gpuHolder->GetDeviceId() != deviceId)) {
                        return false;
                    }
                    const ui32* bins = gpuHolder->GetDeviceBinsPtr();
                    if (!bins) {
                        return false;
                    }
                    NKernel::UpdateHashesFromOneHotFeature(
                        bins,
                        testCount,
                        SafeIntegerCast<ui32>(feature.Value),
                        hashesDev.Get() + offset,
                        stream
                    );
                }

                offset += testCount;
            }
        }

        CB_ENSURE(offset == totalSampleCount, "Unexpected final CTR hash size mismatch");

        // Factorize hashes on device: ranks per object + unique hash values and their counts.
        TCudaDeviceArray<ui32> ranksDev(totalSampleCount);
        TCudaDeviceArray<ui64> uniqueHashesDev(totalSampleCount);
        TCudaDeviceArray<ui32> countsDev(totalSampleCount);
        TCudaDeviceArray<ui32> uniqueCountDev(1);

        NKernel::FactorizeStridedGpuInputToUnique(
            hashesDev.Get(),
            sizeof(ui64),
            totalSampleCount,
            NKernel::EGpuInputDType::UInt64,
            ranksDev.Get(),
            uniqueHashesDev.Get(),
            countsDev.Get(),
            uniqueCountDev.Get(),
            stream
        );

        ui32 uniqueCount = 0;
        CopyDeviceToHost(uniqueCountDev.Get(), &uniqueCount, 1, stream);
        CUDA_SAFE_CALL(cudaStreamSynchronize(stream));

        TVector<ui64> uniqueHashes;
        TVector<ui32> counts;
        uniqueHashes.yresize(uniqueCount);
        counts.yresize(uniqueCount);
        if (uniqueCount) {
            CopyDeviceToHost(uniqueHashesDev.Get(), uniqueHashes.data(), uniqueCount, stream);
            CopyDeviceToHost(countsDev.Get(), counts.data(), uniqueCount, stream);
            CUDA_SAFE_CALL(cudaStreamSynchronize(stream));
        }

        ui32 leafCount = uniqueCount;
        TVector<ui32> bucketCounts = counts;
        const ui32* ranksForAccum = ranksDev.Get();
        TCudaDeviceArray<ui32> remapDev;
        TCudaDeviceArray<ui32> remappedRanksDev;

        if ((ctrLeafCountLimit != Max<ui64>()) && (uniqueCount > ctrLeafCountLimit)) {
            leafCount = SafeIntegerCast<ui32>(ctrLeafCountLimit);
            CB_ENSURE(leafCount > 0, "ctr_leaf_count_limit must be positive");

            TVector<ui32> topIdx(uniqueCount);
            std::iota(topIdx.begin(), topIdx.end(), 0);
            std::nth_element(
                topIdx.begin(),
                topIdx.begin() + leafCount,
                topIdx.end(),
                [&bucketCounts] (ui32 lhs, ui32 rhs) {
                    return bucketCounts[lhs] > bucketCounts[rhs];
                }
            );
            topIdx.resize(leafCount);

            TVector<ui32> remapHost(uniqueCount, leafCount - 1);
            TVector<ui64> bucketHashByIndex;
            bucketHashByIndex.yresize(leafCount);
            for (ui32 newIdx = 0; newIdx < leafCount; ++newIdx) {
                const ui32 oldIdx = topIdx[newIdx];
                remapHost[oldIdx] = newIdx;
                bucketHashByIndex[newIdx] = uniqueHashes[oldIdx];
            }

            TVector<ui32> aggregatedCounts(leafCount, 0);
            for (ui32 oldIdx = 0; oldIdx < uniqueCount; ++oldIdx) {
                aggregatedCounts[remapHost[oldIdx]] += bucketCounts[oldIdx];
            }
            bucketCounts = std::move(aggregatedCounts);

            remapDev = TCudaDeviceArray<ui32>(uniqueCount);
            if (uniqueCount) {
                CopyHostToDevice(remapHost.data(), remapDev.Get(), uniqueCount, stream);
            }
            remappedRanksDev = TCudaDeviceArray<ui32>(totalSampleCount);
            NKernel::RemapIndices(ranksDev.Get(), totalSampleCount, remapDev.Get(), remappedRanksDev.Get(), stream);
            ranksForAccum = remappedRanksDev.Get();

            auto hashIndexBuilder = result->GetIndexHashBuilder(leafCount);
            for (ui32 i = 0; i < leafCount; ++i) {
                hashIndexBuilder.SetIndex(bucketHashByIndex[i], i);
            }
        } else {
            auto hashIndexBuilder = result->GetIndexHashBuilder(uniqueCount);
            for (ui32 i = 0; i < uniqueCount; ++i) {
                hashIndexBuilder.SetIndex(uniqueHashes[i], i);
            }
        }

        if (ctrType == ECtrType::BinarizedTargetMeanValue || ctrType == ECtrType::FloatTargetMeanValue) {
            TCudaDeviceArray<float> sumsDev(leafCount);
            if (leafCount) {
                CUDA_SAFE_CALL(cudaMemsetAsync(sumsDev.Get(), 0, static_cast<size_t>(leafCount) * sizeof(float), stream));
            }

            if (ctrType == ECtrType::BinarizedTargetMeanValue) {
                if (datasetDataForFinalCtrs.LearnTargetClass.Defined() && datasetDataForFinalCtrs.TargetClassesCount.Defined()) {
                    const int targetClassesCount = (**datasetDataForFinalCtrs.TargetClassesCount)[targetBorderClassifierIdx];
                    const int targetBorderCount = targetClassesCount - 1;
                    CB_ENSURE(targetBorderCount > 0, "Invalid target border count for BinarizedTargetMeanValue");

                    const auto& targetClassHost = (**datasetDataForFinalCtrs.LearnTargetClass)[targetBorderClassifierIdx];
                    CB_ENSURE(targetClassHost.size() == learnSampleCount, "targetClass size mismatch");

                    TCudaDeviceArray<ui32> targetClassDev(learnSampleCount);
                    if (learnSampleCount) {
                        CopyHostToDevice(reinterpret_cast<const ui32*>(targetClassHost.data()), targetClassDev.Get(), learnSampleCount, stream);
                    }

                    NKernel::AccumulateBinarizedTargetSumByIndex(
                        ranksForAccum,
                        targetClassDev.Get(),
                        learnSampleCount,
                        1.0f / static_cast<float>(targetBorderCount),
                        sumsDev.Get(),
                        stream
                    );
                } else {
                    CB_ENSURE(learnGpuTargets != nullptr, "GPU targets are not available for target-dependent CTR");
                    CB_ENSURE(
                        datasetDataForFinalCtrs.TargetClassifiers.Defined() && (*datasetDataForFinalCtrs.TargetClassifiers != nullptr),
                        "TargetClassifiers is not set for target-dependent CTR"
                    );
                    const auto& targetClassifier = (**datasetDataForFinalCtrs.TargetClassifiers)[targetBorderClassifierIdx];
                    const ui32 targetId = targetClassifier.GetTargetId();
                    CB_ENSURE(targetId < learnGpuTargets->TargetCount, "GPU target id is out of range");

                    const auto& targetColumn = learnGpuTargets->Targets[targetId];
                    CB_ENSURE(targetColumn.DeviceId == deviceId, "Target device id mismatch");
                    CB_ENSURE(targetColumn.Data != 0, "GPU target pointer is null");
                    CB_ENSURE(targetColumn.StrideBytes > 0, "Invalid stride for GPU target");

                    const auto& bordersHost = targetClassifier.GetBorders();
                    const ui32 borderCount = SafeIntegerCast<ui32>(bordersHost.size());
                    CB_ENSURE(borderCount > 0, "Invalid target border count for BinarizedTargetMeanValue");

                    TCudaDeviceArray<float> targetsDev(learnSampleCount);
                    if (learnSampleCount) {
                        const void* srcTarget = reinterpret_cast<const void*>(static_cast<uintptr_t>(targetColumn.Data));
                        NKernel::CopyStridedGpuInputToFloat(
                            srcTarget,
                            targetColumn.StrideBytes,
                            learnSampleCount,
                            static_cast<NKernel::EGpuInputDType>(static_cast<ui8>(targetColumn.DType)),
                            targetsDev.Get(),
                            stream
                        );
                    }

                    TCudaDeviceArray<float> bordersDev(borderCount);
                    if (borderCount) {
                        CopyHostToDevice(bordersHost.data(), bordersDev.Get(), borderCount, stream);
                    }

                    TCudaDeviceArray<ui32> targetClassDev(learnSampleCount);
                    if (learnSampleCount) {
                        NKernel::BinarizeToUi32(
                            targetsDev.Get(),
                            learnSampleCount,
                            bordersDev.Get(),
                            borderCount,
                            targetClassDev.Get(),
                            stream
                        );
                    }

                    NKernel::AccumulateBinarizedTargetSumByIndex(
                        ranksForAccum,
                        targetClassDev.Get(),
                        learnSampleCount,
                        1.0f / static_cast<float>(borderCount),
                        sumsDev.Get(),
                        stream
                    );
                }
            } else {
                if (datasetDataForFinalCtrs.Targets.Defined()) {
                    CB_ENSURE(
                        datasetDataForFinalCtrs.TargetClassifiers.Defined() && (*datasetDataForFinalCtrs.TargetClassifiers != nullptr),
                        "TargetClassifiers is not set for FloatTargetMeanValue CTR"
                    );
                    const auto targetId = (**datasetDataForFinalCtrs.TargetClassifiers)[targetBorderClassifierIdx].GetTargetId();
                    const auto targetsHost = (*datasetDataForFinalCtrs.Targets)[targetId];
                    CB_ENSURE(targetsHost.size() == learnSampleCount, "targets size mismatch");

                    TCudaDeviceArray<float> targetsDev(learnSampleCount);
                    if (learnSampleCount) {
                        CopyHostToDevice(targetsHost.data(), targetsDev.Get(), learnSampleCount, stream);
                    }
                    NKernel::AccumulateFloatByIndex(
                        ranksForAccum,
                        targetsDev.Get(),
                        learnSampleCount,
                        sumsDev.Get(),
                        stream
                    );
                } else {
                    CB_ENSURE(learnGpuTargets != nullptr, "GPU targets are not available for FloatTargetMeanValue CTR");
                    CB_ENSURE(
                        datasetDataForFinalCtrs.TargetClassifiers.Defined() && (*datasetDataForFinalCtrs.TargetClassifiers != nullptr),
                        "TargetClassifiers is not set for FloatTargetMeanValue CTR"
                    );
                    const auto targetId = (**datasetDataForFinalCtrs.TargetClassifiers)[targetBorderClassifierIdx].GetTargetId();
                    CB_ENSURE(targetId < learnGpuTargets->TargetCount, "GPU target id is out of range");

                    const auto& targetColumn = learnGpuTargets->Targets[targetId];
                    CB_ENSURE(targetColumn.DeviceId == deviceId, "Target device id mismatch");
                    CB_ENSURE(targetColumn.Data != 0, "GPU target pointer is null");
                    CB_ENSURE(targetColumn.StrideBytes > 0, "Invalid stride for GPU target");

                    TCudaDeviceArray<float> targetsDev(learnSampleCount);
                    if (learnSampleCount) {
                        const void* srcTarget = reinterpret_cast<const void*>(static_cast<uintptr_t>(targetColumn.Data));
                        NKernel::CopyStridedGpuInputToFloat(
                            srcTarget,
                            targetColumn.StrideBytes,
                            learnSampleCount,
                            static_cast<NKernel::EGpuInputDType>(static_cast<ui8>(targetColumn.DType)),
                            targetsDev.Get(),
                            stream
                        );
                    }
                    NKernel::AccumulateFloatByIndex(
                        ranksForAccum,
                        targetsDev.Get(),
                        learnSampleCount,
                        sumsDev.Get(),
                        stream
                    );
                }
            }

            TVector<float> sumsHost;
            sumsHost.yresize(leafCount);
            if (leafCount) {
                CopyDeviceToHost(sumsDev.Get(), sumsHost.data(), leafCount, stream);
                CUDA_SAFE_CALL(cudaStreamSynchronize(stream));
            }

            auto meanHist = result->AllocateBlobAndGetArrayRef<TCtrMeanHistory>(leafCount);
            for (ui32 i = 0; i < leafCount; ++i) {
                meanHist[i].Sum = sumsHost[i];
                meanHist[i].Count = SafeIntegerCast<int>(bucketCounts[i]);
            }
        } else if (ctrType == ECtrType::Counter || ctrType == ECtrType::FeatureFreq) {
            auto ctrIntArray = result->AllocateBlobAndGetArrayRef<int>(leafCount);
            int maxCount = 0;
            for (ui32 i = 0; i < leafCount; ++i) {
                const int c = SafeIntegerCast<int>(bucketCounts[i]);
                ctrIntArray[i] = c;
                maxCount = Max(maxCount, c);
            }
            if (ctrType == ECtrType::Counter) {
                result->CounterDenominator = maxCount;
            } else {
                result->CounterDenominator = SafeIntegerCast<int>(totalSampleCount);
            }
        } else {
            int targetClassesCount = 0;
            TCudaDeviceArray<ui32> targetClassDev(learnSampleCount);

            if (datasetDataForFinalCtrs.LearnTargetClass.Defined() && datasetDataForFinalCtrs.TargetClassesCount.Defined()) {
                targetClassesCount = (**datasetDataForFinalCtrs.TargetClassesCount)[targetBorderClassifierIdx];
                CB_ENSURE(targetClassesCount > 0, "Invalid target classes count");

                const auto& targetClassHost = (**datasetDataForFinalCtrs.LearnTargetClass)[targetBorderClassifierIdx];
                CB_ENSURE(targetClassHost.size() == learnSampleCount, "targetClass size mismatch");

                if (learnSampleCount) {
                    CopyHostToDevice(reinterpret_cast<const ui32*>(targetClassHost.data()), targetClassDev.Get(), learnSampleCount, stream);
                }
            } else {
                CB_ENSURE(learnGpuTargets != nullptr, "GPU targets are not available for target-dependent CTR");
                CB_ENSURE(
                    datasetDataForFinalCtrs.TargetClassifiers.Defined() && (*datasetDataForFinalCtrs.TargetClassifiers != nullptr),
                    "TargetClassifiers is not set for target-dependent CTR"
                );

                const auto& targetClassifier = (**datasetDataForFinalCtrs.TargetClassifiers)[targetBorderClassifierIdx];
                targetClassesCount = targetClassifier.GetClassesCount();
                CB_ENSURE(targetClassesCount > 0, "Invalid target classes count");

                const ui32 targetId = targetClassifier.GetTargetId();
                CB_ENSURE(targetId < learnGpuTargets->TargetCount, "GPU target id is out of range");

                const auto& targetColumn = learnGpuTargets->Targets[targetId];
                CB_ENSURE(targetColumn.DeviceId == deviceId, "Target device id mismatch");
                CB_ENSURE(targetColumn.Data != 0, "GPU target pointer is null");
                CB_ENSURE(targetColumn.StrideBytes > 0, "Invalid stride for GPU target");

                const auto& bordersHost = targetClassifier.GetBorders();
                const ui32 borderCount = SafeIntegerCast<ui32>(bordersHost.size());
                CB_ENSURE(
                    targetClassesCount == SafeIntegerCast<int>(borderCount + 1),
                    "Target classifier borders size mismatch"
                );

                TCudaDeviceArray<float> targetsDev(learnSampleCount);
                if (learnSampleCount) {
                    const void* srcTarget = reinterpret_cast<const void*>(static_cast<uintptr_t>(targetColumn.Data));
                    NKernel::CopyStridedGpuInputToFloat(
                        srcTarget,
                        targetColumn.StrideBytes,
                        learnSampleCount,
                        static_cast<NKernel::EGpuInputDType>(static_cast<ui8>(targetColumn.DType)),
                        targetsDev.Get(),
                        stream
                    );
                }

                TCudaDeviceArray<float> bordersDev(borderCount);
                if (borderCount) {
                    CopyHostToDevice(bordersHost.data(), bordersDev.Get(), borderCount, stream);
                }

                if (learnSampleCount) {
                    NKernel::BinarizeToUi32(
                        targetsDev.Get(),
                        learnSampleCount,
                        bordersDev.Get(),
                        borderCount,
                        targetClassDev.Get(),
                        stream
                    );
                }
            }

            const ui64 countsSize = static_cast<ui64>(leafCount) * static_cast<ui64>(targetClassesCount);

            TCudaDeviceArray<ui32> classCountsDev(countsSize);
            if (countsSize) {
                CUDA_SAFE_CALL(cudaMemsetAsync(classCountsDev.Get(), 0, static_cast<size_t>(countsSize) * sizeof(ui32), stream));
            }

            NKernel::AccumulateClassCountsByIndex(
                ranksForAccum,
                targetClassDev.Get(),
                learnSampleCount,
                SafeIntegerCast<ui32>(targetClassesCount),
                classCountsDev.Get(),
                stream
            );

            TVector<ui32> classCountsHost;
            classCountsHost.yresize(countsSize);
            if (countsSize) {
                CopyDeviceToHost(classCountsDev.Get(), classCountsHost.data(), countsSize, stream);
                CUDA_SAFE_CALL(cudaStreamSynchronize(stream));
            }

            result->TargetClassesCount = targetClassesCount;
            auto ctrIntArray = result->AllocateBlobAndGetArrayRef<int>(countsSize);
            for (ui64 i = 0; i < countsSize; ++i) {
                ctrIntArray[SafeIntegerCast<size_t>(i)] = SafeIntegerCast<int>(classCountsHost[SafeIntegerCast<size_t>(i)]);
            }
        }

        return true;
    }
}
#endif


static void CalcFinalCtrs(
    const ECtrType ctrType,
    const TProjection& projection,
    const TDatasetDataForFinalCtrs& datasetDataForFinalCtrs,
    const NCB::TFeaturesArraySubsetIndexing& learnFeaturesSubsetIndexing,
    const NCB::TPerfectHashedToHashedCatValuesMap& perfectHashedToHashedCatValuesMap,
    int targetBorderClassifierIdx,
    ui64 ctrLeafCountLimit,
    bool storeAllSimpleCtr,
    ECounterCalc counterCalcMethod,
    TCtrValueTable* result,
    NPar::ILocalExecutor* localExecutor) {

#if defined(HAVE_CUDA)
    if (TryCalcFinalCtrsOnGpu(
            ctrType,
            projection,
            datasetDataForFinalCtrs,
            perfectHashedToHashedCatValuesMap,
            targetBorderClassifierIdx,
            ctrLeafCountLimit,
            storeAllSimpleCtr,
            counterCalcMethod,
            result))
    {
        return;
    }
#endif

    ui32 learnSampleCount = datasetDataForFinalCtrs.Data.Learn->GetObjectCount();
    ui32 totalSampleCount = learnSampleCount;
    if (ctrType == ECtrType::Counter && counterCalcMethod == ECounterCalc::Full) {
        totalSampleCount += datasetDataForFinalCtrs.Data.GetTestSampleCount();
    }
    TVector<ui64> hashArr(totalSampleCount);
    CalcHashes(
        projection,
        *datasetDataForFinalCtrs.Data.Learn->ObjectsData,
        learnFeaturesSubsetIndexing,
        &perfectHashedToHashedCatValuesMap,
        hashArr.begin(),
        hashArr.begin() + learnSampleCount,
        localExecutor
    );
    if (totalSampleCount > learnSampleCount) {
        ui64* testHashBegin = hashArr.begin() + learnSampleCount;
        for (const auto& testDataPtr : datasetDataForFinalCtrs.Data.Test) {
            ui64* testHashEnd = testHashBegin + testDataPtr->GetObjectCount();
            CalcHashes(
                projection,
                *testDataPtr->ObjectsData,
                testDataPtr->ObjectsData->GetFeaturesArraySubsetIndexing(),
                &perfectHashedToHashedCatValuesMap,
                testHashBegin,
                testHashEnd,
                localExecutor);
            testHashBegin = testHashEnd;
        }
    }

    if (projection.IsSingleCatFeature() && storeAllSimpleCtr) {
        ctrLeafCountLimit = Max<ui64>();
    }
    CalcFinalCtrsImpl(
        ctrType,
        ctrLeafCountLimit,
        NeedTargetClassifier(ctrType) ?
            (**datasetDataForFinalCtrs.LearnTargetClass)[targetBorderClassifierIdx] : TVector<int>(),
        (*datasetDataForFinalCtrs.Targets)[(**datasetDataForFinalCtrs.TargetClassifiers)[targetBorderClassifierIdx].GetTargetId()],
        totalSampleCount,
        NeedTargetClassifier(ctrType) ?
            (**datasetDataForFinalCtrs.TargetClassesCount)[targetBorderClassifierIdx] : 0,
        &hashArr,
        result
    );
}

static ui64 EstimateCalcFinalCtrsCpuRamUsage(
    const ECtrType ctrType,
    const TTrainingDataProviders& data,
    int targetClassesCount,
    ui64 ctrLeafCountLimit,
    ECounterCalc counterCalcMethod) {

    ui64 cpuRamUsageEstimate = 0;

    ui32 totalSampleCount = data.Learn->GetObjectCount();
    if (ctrType == ECtrType::Counter && counterCalcMethod == ECounterCalc::Full) {
        totalSampleCount += data.GetTestSampleCount();
    }
    // for hashArr in CalcFinalCtrs
    cpuRamUsageEstimate += sizeof(ui64)*totalSampleCount;

    ui64 reindexHashRamLimit =
        sizeof(TDenseHash<ui64,ui32>::value_type)*FastClp2(totalSampleCount*2);

    // data for temporary vector to calc top
    ui64 computeReindexHashTopRamLimit = (ctrLeafCountLimit < totalSampleCount) ?
        sizeof(std::pair<ui64, ui32>)*totalSampleCount : 0;

    // CalcFinalCtrsImpl stage 1
    ui64 computeReindexHashRamLimit = reindexHashRamLimit + computeReindexHashTopRamLimit;

    ui64 reindexHashAfterComputeSizeLimit = Min<ui64>(ctrLeafCountLimit, totalSampleCount);

    ui64 reindexHashAfterComputeRamLimit =
        sizeof(TDenseHash<ui64,ui32>::value_type)*FastClp2(reindexHashAfterComputeSizeLimit*2);


    ui64 indexBucketsRamLimit = (sizeof(NCatboost::TBucket) *
         NCatboost::TDenseIndexHashBuilder::GetProperBucketsCount(reindexHashAfterComputeSizeLimit));

    // CalcFinalCtrsImplstage 2
    ui64 buildingHashIndexRamLimit = reindexHashAfterComputeRamLimit + indexBucketsRamLimit;

    ui64 ctrBlobRamLimit = 0;
    if (ctrType == ECtrType::BinarizedTargetMeanValue || ctrType == ECtrType::FloatTargetMeanValue) {
        ctrBlobRamLimit = sizeof(TCtrMeanHistory)*reindexHashAfterComputeSizeLimit;
    } else if (ctrType == ECtrType::Counter || ctrType == ECtrType::FeatureFreq) {
        ctrBlobRamLimit = sizeof(int)*reindexHashAfterComputeSizeLimit;
    } else {
        ctrBlobRamLimit = sizeof(int)*(reindexHashAfterComputeSizeLimit * targetClassesCount);
    }

    // CalcFinalCtrsImplstage 3
    ui64 fillingCtrBlobRamLimit = indexBucketsRamLimit + ctrBlobRamLimit;

    // max usage is max of CalcFinalCtrsImpl 3 stages
    cpuRamUsageEstimate += Max(computeReindexHashRamLimit, buildingHashIndexRamLimit, fillingCtrBlobRamLimit);

    return cpuRamUsageEstimate;
}

void CalcFinalCtrsAndSaveToModel(
    ui64 cpuRamLimit,
    const THashMap<TFeatureCombination, TProjection>& featureCombinationToProjectionMap,
    const TDatasetDataForFinalCtrs& datasetDataForFinalCtrs,
    const NCB::TPerfectHashedToHashedCatValuesMap& perfectHashedToHashedCatValuesMap,
    ui64 ctrLeafCountLimit,
    bool storeAllSimpleCtrs,
    ECounterCalc counterCalcMethod,
    const TVector<TModelCtrBase>& usedCtrBases,
    std::function<void(TCtrValueTable&& table)>&& asyncCtrValueTableCallback,
    NPar::ILocalExecutor* localExecutor) {

    CATBOOST_DEBUG_LOG << "Started parallel calculation of " << usedCtrBases.size() << " unique ctrs" << Endl;

    TMaybe<TFeaturesArraySubsetIndexing> permutedLearnFeaturesSubsetIndexing;
    const TFeaturesArraySubsetIndexing* learnFeaturesSubsetIndexing = nullptr;
    if (datasetDataForFinalCtrs.LearnPermutation) {
        permutedLearnFeaturesSubsetIndexing = Compose(
            datasetDataForFinalCtrs.Data.Learn->ObjectsData->GetFeaturesArraySubsetIndexing(),
            **datasetDataForFinalCtrs.LearnPermutation);
        learnFeaturesSubsetIndexing = &*permutedLearnFeaturesSubsetIndexing;
    } else {
        learnFeaturesSubsetIndexing =
            &datasetDataForFinalCtrs.Data.Learn->ObjectsData->GetFeaturesArraySubsetIndexing();
    }


    ui64 cpuRamUsage = NMemInfo::GetMemInfo().RSS;
    OutputWarningIfCpuRamUsageOverLimit(cpuRamUsage, cpuRamLimit);

    {
        NCB::TResourceConstrainedExecutor finalCtrExecutor(
            "CPU RAM",
            cpuRamLimit - Min(cpuRamLimit, cpuRamUsage),
            true,
            localExecutor);

        const auto& layout = *datasetDataForFinalCtrs.Data.Learn->MetaInfo.FeaturesLayout;

        auto ctrTableGenerator = [&] (const TModelCtrBase& ctr) -> TCtrValueTable {
            TCtrValueTable resTable;
            CalcFinalCtrs(
                ctr.CtrType,
                featureCombinationToProjectionMap.at(ctr.Projection),
                datasetDataForFinalCtrs,
                *learnFeaturesSubsetIndexing,
                perfectHashedToHashedCatValuesMap,
                ctr.TargetBorderClassifierIdx,
                ctrLeafCountLimit,
                storeAllSimpleCtrs,
                counterCalcMethod,
                &resTable,
                localExecutor);
            resTable.ModelCtrBase = ctr;
            CATBOOST_DEBUG_LOG << "Finished CTR: " << ctr.CtrType << " "
                << BuildDescription(layout, ctr.Projection) << Endl;
            return resTable;
        };

        const auto getTargetClassesCountForEstimation = [&] (const TModelCtrBase& ctr) -> int {
            if (!NeedTargetClassifier(ctr.CtrType)) {
                return 0;
            }
            if (datasetDataForFinalCtrs.TargetClassesCount.Defined() && (*datasetDataForFinalCtrs.TargetClassesCount != nullptr)) {
                return (**datasetDataForFinalCtrs.TargetClassesCount)[ctr.TargetBorderClassifierIdx];
            }
            CB_ENSURE(
                datasetDataForFinalCtrs.TargetClassifiers.Defined() && (*datasetDataForFinalCtrs.TargetClassifiers != nullptr),
                "TargetClassifiers is not set for target-dependent CTRs"
            );
            return (**datasetDataForFinalCtrs.TargetClassifiers)[ctr.TargetBorderClassifierIdx].GetClassesCount();
        };

        for (const auto& ctr : usedCtrBases) {
            finalCtrExecutor.Add(
                {
                    EstimateCalcFinalCtrsCpuRamUsage(
                        ctr.CtrType,
                        datasetDataForFinalCtrs.Data,
                        getTargetClassesCountForEstimation(ctr),
                        ctrLeafCountLimit,
                        counterCalcMethod
                    ),
                    [&asyncCtrValueTableCallback, &ctrTableGenerator, &ctr] () {
                        auto table = ctrTableGenerator(ctr);
                        asyncCtrValueTableCallback(std::move(table));
                    }
                }
            );
        }

        finalCtrExecutor.ExecTasks();
    }

    CATBOOST_DEBUG_LOG << "CTR calculation finished" << Endl;
}
