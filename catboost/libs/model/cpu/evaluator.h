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

#include <library/cpp/sse/sse.h>

namespace NCB::NModelEvaluation {

    using TTreeCalcFunction = std::function<void(
        const TModelTrees& ModelTrees,
        const TCPUEvaluatorQuantizedData*,
        size_t docCountInBlock,
        TCalcerIndexType* __restrict indexesVec,
        size_t treeStart,
        size_t treeEnd,
        double* __restrict results)>;


    TTreeCalcFunction GetCalcTreesFunction(
        const TModelTrees& trees,
        size_t docCountInBlock,
        bool calcIndexesOnly = false);

    template <class X>
    inline X* GetAligned(X* val) {
        uintptr_t off = ((uintptr_t)val) & 0xf;
        val = (X*)((ui8*)val - off + 0x10);
        return val;
    }

    template <
        typename TFloatFeatureAccessor,
        typename TCatFeatureAccessor,
        typename TFunctor
    >
    inline void ProcessDocsInBlocks(
        const TModelTrees& trees,
        const TIntrusivePtr<ICtrProvider>& ctrProvider,
        TFloatFeatureAccessor floatFeatureAccessor,
        TCatFeatureAccessor catFeaturesAccessor,
        size_t docCount,
        size_t blockSize,
        TFunctor callback,
        const NCB::NModelEvaluation::TFeatureLayout* featureInfo
    ) {
        ProcessDocsInBlocks(
            trees,
            ctrProvider,
            TIntrusivePtr<TTextProcessingCollection>(),
            floatFeatureAccessor,
            catFeaturesAccessor,
            [](TFeaturePosition, size_t) -> TStringBuf {
                CB_ENSURE_INTERNAL(
                    false,
                    "Trying to access text data from model.Calc() interface which has no text features"
                );
                return "Undefined";
            },
            docCount,
            blockSize,
            callback,
            featureInfo
        );
    }

    template <
        typename TFloatFeatureAccessor,
        typename TCatFeatureAccessor,
        typename TTextFeatureAccessor,
        typename TFunctor
    >
    inline void ProcessDocsInBlocks(
        const TModelTrees& trees,
        const TIntrusivePtr<ICtrProvider>& ctrProvider,
        const TIntrusivePtr<TTextProcessingCollection>& textProcessingCollection,
        TFloatFeatureAccessor floatFeatureAccessor,
        TCatFeatureAccessor catFeaturesAccessor,
        TTextFeatureAccessor textFeatureAccessor,
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

        TVector<ui32> transposedHash(blockSize * trees.GetUsedCatFeaturesCount());
        TVector<float> ctrs(trees.GetUsedModelCtrs().size() * blockSize);
        TVector<float> estimatedFeatures;
        if (textProcessingCollection) {
            // TODO(d-kruchinin): replace to GetUsedEstimatedFeatures.size() after creation TrimFeatures
            estimatedFeatures = TVector<float>(textProcessingCollection->TotalNumberOfOutputFeatures() * blockSize);
        }

        for (size_t blockStart = 0; blockStart < docCount; blockStart += blockSize) {
            const auto docCountInBlock = Min(blockSize, docCount - blockStart);
            BinarizeFeatures(
                trees,
                ctrProvider,
                textProcessingCollection,
                floatFeatureAccessor,
                catFeaturesAccessor,
                textFeatureAccessor,
                blockStart,
                blockStart + docCountInBlock,
                &quantizedData,
                transposedHash,
                ctrs,
                estimatedFeatures,
                featureInfo
            );
            callback(docCountInBlock, &quantizedData);
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

    template <typename TFloatFeatureAccessor, typename TCatFeatureAccessor>
    inline void CalcLeafIndexesGeneric(
        const TModelTrees& trees,
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
        CB_ENSURE(
            trees.GetTextFeatures().empty(),
            "Leaf indexes calculation is not implemented for models with text features"
        );
        std::fill(treeLeafIndexes.begin(), treeLeafIndexes.end(), 0);
        const size_t blockSize = Min(FORMULA_EVALUATION_BLOCK_SIZE, docCount);
        TCalcerIndexType* indexesWritePtr = treeLeafIndexes.data();

        auto calcTrees = GetCalcTreesFunction(trees, blockSize, true);

        if (docCount == 1) {
            ProcessDocsInBlocks(
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
        ProcessDocsInBlocks(
            trees,
            ctrProvider,
            floatFeatureAccessor,
            catFeaturesAccessor,
            docCount,
            blockSize,
            [&](size_t docCountInBlock, const TCPUEvaluatorQuantizedData* quantizedData) {
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
