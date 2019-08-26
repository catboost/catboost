#pragma once

#include "quantization.h"

#include <util/generic/utility.h>
#include <util/generic/vector.h>

#include <util/stream/labeled.h>
#include <util/system/platform.h>
#include <util/system/types.h>
#include <util/system/yassert.h>

#include <algorithm>
#include <functional>
#include <limits>
#include <type_traits>

#include <library/sse/sse.h>

namespace NCB::NModelEvaluation {

    using TTreeCalcFunction = std::function<void(
        const TObliviousTrees& obliviousTrees,
        const TCPUEvaluatorQuantizedData*,
        size_t docCountInBlock,
        TCalcerIndexType* __restrict indexesVec,
        size_t treeStart,
        size_t treeEnd,
        double* __restrict results)>;


    TTreeCalcFunction GetCalcTreesFunction(
        const TObliviousTrees& trees,
        size_t docCountInBlock,
        bool calcIndexesOnly = false);

    template <class X>
    inline X* GetAligned(X* val) {
        uintptr_t off = ((uintptr_t)val) & 0xf;
        val = (X*)((ui8*)val - off + 0x10);
        return val;
    }

    template <bool isQuantizedFeaturesData = false,
        typename TFloatFeatureAccessor, typename TCatFeatureAccessor, typename TFunctor>
    inline void ProcessDocsInBlocks(
        const TObliviousTrees& trees,
        const TIntrusivePtr<ICtrProvider>& ctrProvider,
        TFloatFeatureAccessor floatFeatureAccessor,
        TCatFeatureAccessor catFeaturesAccessor,
        size_t docCount,
        size_t blockSize,
        TFunctor callback,
        const NCB::NModelEvaluation::TFeatureLayout* featureInfo
    ) {
        const size_t binSlots = blockSize * trees.GetEffectiveBinaryFeaturesBucketsCount();

        TCPUEvaluatorQuantizedData quantizedData;
        if (binSlots < 65536) { // 65KB of stack maximum
            quantizedData.QuantizedData = NCB::TMaybeOwningArrayHolder<ui8>::CreateNonOwning(
                MakeArrayRef(GetAligned((ui8*)(alloca(binSlots + 0x20))), binSlots));
        } else {
            TVector<ui8> binFeaturesHolder;
            binFeaturesHolder.yresize(binSlots);
            quantizedData.QuantizedData = NCB::TMaybeOwningArrayHolder<ui8>::CreateOwning(std::move(binFeaturesHolder));
        }
        if constexpr (!isQuantizedFeaturesData) {
            TVector<ui32> transposedHash(blockSize * trees.GetUsedCatFeaturesCount());
            TVector<float> ctrs(trees.GetUsedModelCtrs().size() * blockSize);
            for (size_t blockStart = 0; blockStart < docCount; blockStart += blockSize) {
                const auto docCountInBlock = Min(blockSize, docCount - blockStart);
                BinarizeFeatures(
                    trees,
                    ctrProvider,
                    floatFeatureAccessor,
                    catFeaturesAccessor,
                    blockStart,
                    blockStart + docCountInBlock,
                    &quantizedData,
                    transposedHash,
                    ctrs,
                    featureInfo
                );
                callback(docCountInBlock, &quantizedData);
            }
        } else {
            for (size_t blockStart = 0; blockStart < docCount; blockStart += blockSize) {
                const auto docCountInBlock = Min(blockSize, docCount - blockStart);
                AssignFeatureBins(
                    trees,
                    floatFeatureAccessor,
                    catFeaturesAccessor,
                    blockStart,
                    blockStart + docCountInBlock,
                    &quantizedData
                );
                callback(docCountInBlock, &quantizedData);
            }
        }
    }

    template <typename T>
    void Transpose2DArray(
        TConstArrayRef<T> srcArray, // assume values are laid row by row
        size_t srcRowCount,
        size_t srcColumnCount,
        TArrayRef<T> dstArray
    ) {
        Y_ASSERT(srcArray.size() == srcRowCount * srcColumnCount);
        Y_ASSERT(srcArray.size() == dstArray.size());
        for (size_t srcRowIndex = 0; srcRowIndex < srcRowCount; ++srcRowIndex) {
            for (size_t srcColumnIndex = 0; srcColumnIndex < srcColumnCount; ++srcColumnIndex) {
                dstArray[srcColumnIndex * srcRowCount + srcRowIndex] =
                    srcArray[srcRowIndex * srcColumnCount + srcColumnIndex];
            }
        }
    }

    template <bool IsQuantizedFeaturesData = false, typename TFloatFeatureAccessor, typename TCatFeatureAccessor>
    inline void CalcLeafIndexesGeneric(
        const TObliviousTrees& trees,
        const TIntrusivePtr<ICtrProvider>& ctrProvider,
        TFloatFeatureAccessor floatFeatureAccessor,
        TCatFeatureAccessor catFeaturesAccessor,
        size_t docCount,
        size_t treeStart,
        size_t treeEnd,
        TArrayRef<TCalcerIndexType> treeLeafIndexes,
        const NCB::NModelEvaluation::TFeatureLayout* featureInfo
    ) {
        Y_ASSERT(treeEnd >= treeStart);
        const size_t treeCount = treeEnd - treeStart;
        Y_ASSERT(trees.GetFirstLeafOffsets().size() >= treeEnd);
        CB_ENSURE(treeLeafIndexes.size() == docCount * treeCount,
                  "`treeLeafIndexes` size is insufficient: "
                      LabeledOutput(treeLeafIndexes.size(), docCount * treeCount));
        std::fill(treeLeafIndexes.begin(), treeLeafIndexes.end(), 0);
        const size_t blockSize = Min(FORMULA_EVALUATION_BLOCK_SIZE, docCount);
        TCalcerIndexType* indexesWritePtr = treeLeafIndexes.data();

        auto calcTrees = GetCalcTreesFunction(trees, blockSize, true);

        if (docCount == 1) {
            ProcessDocsInBlocks<IsQuantizedFeaturesData>(
                trees, ctrProvider, floatFeatureAccessor, catFeaturesAccessor, docCount, blockSize,
                [&](size_t docCountInBlock, const TCPUEvaluatorQuantizedData* quantizedData) {
                    calcTrees(
                        trees,
                        quantizedData,
                        docCountInBlock,
                        indexesWritePtr,
                        treeStart,
                        treeEnd,
                        nullptr
                    );
                },
                featureInfo
            );
            return;
        }
        TVector<TCalcerIndexType> tmpLeafIndexHolder(blockSize * treeCount);
        TCalcerIndexType* transposedLeafIndexesPtr = tmpLeafIndexHolder.data();
        ProcessDocsInBlocks<IsQuantizedFeaturesData>(trees, ctrProvider, floatFeatureAccessor, catFeaturesAccessor,
                                                     docCount, blockSize,
                                                     [&](size_t docCountInBlock,
                                                         const TCPUEvaluatorQuantizedData* quantizedData) {
                                                         calcTrees(
                                                             trees,
                                                             quantizedData,
                                                             docCountInBlock,
                                                             transposedLeafIndexesPtr,
                                                             treeStart,
                                                             treeEnd,
                                                             nullptr
                                                         );
                                                         const size_t indexCountInBlock = docCountInBlock * treeCount;
                                                         Transpose2DArray<TCalcerIndexType>(
                                                             {transposedLeafIndexesPtr, indexCountInBlock},
                                                             treeCount,
                                                             docCountInBlock,
                                                             {indexesWritePtr, indexCountInBlock}
                                                         );
                                                         indexesWritePtr += indexCountInBlock;
                                                     },
                                                     featureInfo
        );
    }
}
