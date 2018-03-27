#pragma once

#include "grid_creator.h"

#include <catboost/cuda/utils/cpu_random.h>
#include <catboost/cuda/utils/compression_helpers.h>
#include <catboost/cuda/cuda_lib/helpers.h>
#include <catboost/libs/options/binarization_options.h>
#include <library/grid_creator/binarization.h>
#include <util/system/types.h>
#include <util/generic/vector.h>
#include <util/random/shuffle.h>

namespace NCatboostCuda {
    void GroupSamples(const TVector<TGroupId>& qid, TVector<TVector<ui32>>* qdata);

    inline TVector<TVector<ui32>> GroupSamples(const TVector<TGroupId>& qid) {
        TVector<TVector<ui32>> qdata;
        GroupSamples(qid, &qdata);
        return qdata;
    }

    inline TVector<ui32> ComputeGroupOffsets(const TVector<TVector<ui32>>& queries) {
        TVector<ui32> offsets;
        ui32 cursor = 0;
        for (const auto& query : queries) {
            offsets.push_back(cursor);
            cursor += query.size();
        }
        return offsets;
    }

    inline TVector<ui32> ComputeGroupSizes(const TVector<TVector<ui32>>& gdata) {
        TVector<ui32> sizes;
        sizes.resize(gdata.size());
        for (ui32 i = 0; i < gdata.size(); ++i) {
            sizes[i] = gdata[i].size();
        }
        return sizes;
    }

    template <class TIndicesType>
    inline void Shuffle(ui64 seed, ui32 blockSize, ui32 sampleCount, TVector<TIndicesType>* orderPtr) {
        TRandom rng(seed);
        rng.Advance(10);
        auto& order = *orderPtr;
        order.resize(sampleCount);
        std::iota(order.begin(), order.end(), 0);

        if (blockSize == 1) {
            ::Shuffle(order.begin(), order.begin() + sampleCount, rng);
        } else {
            const auto blocksCount = static_cast<ui32>(NHelpers::CeilDivide(order.size(), blockSize));
            TVector<ui32> blocks(blocksCount);
            std::iota(blocks.begin(), blocks.end(), 0);
            ::Shuffle(blocks.begin(), blocks.end(), rng);

            ui32 cursor = 0;
            for (ui32 i = 0; i < blocksCount; ++i) {
                const ui32 blockStart = blocks[i] * blockSize;
                const ui32 blockEnd = Min<ui32>(blockStart + blockSize, order.size());
                for (ui32 j = blockStart; j < blockEnd; ++j) {
                    order[cursor++] = j;
                }
            }
        }
    }

    template <class TIndicesType>
    inline void QueryConsistentShuffle(ui64 seed, ui32 blockSize, const TVector<TGroupId>& queryIds, TVector<TIndicesType>* orderPtr) {
        auto grouppedQueries = GroupSamples(queryIds);
        auto offsets = ComputeGroupOffsets(grouppedQueries);
        TVector<ui32> order;
        Shuffle(seed, blockSize, grouppedQueries.size(), &order);
        auto& docwiseOrder = *orderPtr;
        docwiseOrder.resize(queryIds.size());

        ui32 cursor = 0;
        for (ui32 i = 0; i < order.size(); ++i) {
            const auto queryid = order[i];
            ui32 queryOffset = offsets[queryid];
            ui32 querySize = grouppedQueries[queryid].size();
            for (ui32 doc = 0; doc < querySize; ++doc) {
                docwiseOrder[cursor++] = queryOffset + doc;
            }
        }
    }

    template <class T>
    TVector<T> SampleVector(const TVector<T>& vec,
                            ui32 size,
                            ui64 seed) {
        TRandom random(seed);
        TVector<T> result(size);
        for (ui32 i = 0; i < size; ++i) {
            result[i] = vec[(random.NextUniformL() % vec.size())];
        }
        return result;
    };

    inline ui32 GetSampleSizeForBorderSelectionType(ui32 vecSize,
                                                    EBorderSelectionType borderSelectionType) {
        switch (borderSelectionType) {
            case EBorderSelectionType::MinEntropy:
            case EBorderSelectionType::MaxLogSum:
                return Min<ui32>(vecSize, 100000);
            default:
                return vecSize;
        }
    };

    template <class T>
    void ApplyPermutation(const TVector<ui64>& order,
                          TVector<T>& data) {
        if (data.size()) {
            TVector<T> tmp(data.begin(), data.end());
            for (ui32 i = 0; i < order.size(); ++i) {
                data[i] = tmp[order[i]];
            }
        }
    };

    inline TVector<float> BuildBorders(const TVector<float>& floatFeature,
                                       const ui32 seed,
                                       const NCatboostOptions::TBinarizationOptions& config) {
        TOnCpuGridBuilderFactory gridBuilderFactory;
        ui32 sampleSize = GetSampleSizeForBorderSelectionType(floatFeature.size(),
                                                              config.BorderSelectionType);
        if (sampleSize < floatFeature.size()) {
            auto sampledValues = SampleVector(floatFeature, sampleSize, TRandom::GenerateSeed(seed));
            return TBordersBuilder(gridBuilderFactory, sampledValues)(config);
        } else {
            return TBordersBuilder(gridBuilderFactory, floatFeature)(config);
        }
    };

    //this routine assumes NanMode = Min/Max means we have nans in float-values
    template <class TBinType = ui32>
    inline TVector<TBinType> BinarizeLine(const float* values,
                                          const ui64 valuesCount,
                                          const ENanMode nanMode,
                                          const TVector<float>& borders) {
        TVector<TBinType> result(valuesCount);

        NPar::TLocalExecutor::TExecRangeParams params(0, (int)valuesCount);
        params.SetBlockSize(16384);

        NPar::LocalExecutor().ExecRange([&](int blockIdx) {
            NPar::LocalExecutor().BlockedLoopBody(params, [&](int i) {
                float value = values[i];
                if (IsNan(value)) {
                    CB_ENSURE(nanMode != ENanMode::Forbidden, "Error, NaNs for current feature are forbidden. (All NaNs are forbidden or test has NaNs and learn not)");
                    result[i] = nanMode == ENanMode::Min ? 0 : borders.size();
                } else {
                    ui32 bin = Binarize<TBinType>(borders, values[i]);
                    if (nanMode == ENanMode::Min) {
                        result[i] = bin + 1;
                    } else {
                        result[i] = bin;
                    }
                }
            })(blockIdx);
        },
                                        0, params.GetBlockCount(), NPar::TLocalExecutor::WAIT_COMPLETE);

        return result;
    }
}
