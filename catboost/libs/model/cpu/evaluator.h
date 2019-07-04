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

    template <class TFloatFeatureAccessor, class TCatFeatureAccessor, bool IsQuantizedFeaturesData = false>
    class TLeafIndexCalcer : public NCB::NModelEvaluation::ILeafIndexCalcer {
    public:
        TLeafIndexCalcer(
            const TFullModel& model,
            TFloatFeatureAccessor floatFeatureAccessor,
            TCatFeatureAccessor catFeatureAccessor,
            size_t docCount,
            size_t treeStrat,
            size_t treeEnd
        )
            : Model(model), FloatFeatureAccessor(floatFeatureAccessor), CatFeatureAccessor(catFeatureAccessor),
              DocCount(docCount), TreeStart(treeStrat), TreeEnd(treeEnd) {
            CalcNextBatch();
        }

        bool Next() override {
            ++CurrDocIndex;
            if (CurrDocIndex < DocCount) {
                if (CurrDocIndex == CurrBatchStart + CurrBatchSize) {
                    CalcNextBatch();
                }
                return true;
            } else {
                return false;
            }
        }

        bool CanGet() const override {
            return CurrDocIndex < DocCount;
        }

        TVector<TCalcerIndexType> Get() const override {
            const auto treeCount = TreeEnd - TreeStart;
            const auto docIndexInBatch = CurrDocIndex - CurrBatchStart;
            TVector<TCalcerIndexType> result;
            result.reserve(treeCount);
            for (size_t treeNum = 0; treeNum < treeCount; ++treeNum) {
                result.push_back(CurrentBatchLeafIndexes[docIndexInBatch + treeNum * CurrBatchSize]);
            }
            return result;
        }

    private:
        void CalcNextBatch() {
            Y_ASSERT(CurrDocIndex == CurrBatchStart + CurrBatchSize);
            CurrBatchStart += CurrBatchSize;
            CurrBatchSize = Min(DocCount - CurrDocIndex, FORMULA_EVALUATION_BLOCK_SIZE);
            const size_t batchResultSize = (TreeEnd - TreeStart) * CurrBatchSize;
            CurrentBatchLeafIndexes.resize(batchResultSize);
            auto calcTrees = GetCalcTreesFunction(*Model.ObliviousTrees, CurrBatchSize, true);
            ProcessDocsInBlocks<IsQuantizedFeaturesData>(
                *Model.ObliviousTrees,
                Model.CtrProvider,
                [this](const TFeaturePosition& position, size_t index) -> float {
                    return FloatFeatureAccessor(position, CurrBatchStart + index);
                },
                [this](const TFeaturePosition& position, size_t index) -> ui32 {
                    return CatFeatureAccessor(position, CurrBatchStart + index);
                },
                CurrBatchSize,
                CurrBatchSize, [&](size_t docCountInBlock, const TCPUEvaluatorQuantizedData* quantizedData) {
                    Y_ASSERT(docCountInBlock == CurrBatchSize);
                    calcTrees(
                        *Model.ObliviousTrees,
                        quantizedData,
                        docCountInBlock,
                        CurrentBatchLeafIndexes.data(),
                        TreeStart,
                        TreeEnd,
                        nullptr
                    );
                },
                nullptr
            );
        }

    private:
        const TFullModel& Model;
        TFloatFeatureAccessor FloatFeatureAccessor;
        TCatFeatureAccessor CatFeatureAccessor;
        TVector<TCalcerIndexType> CurrentBatchLeafIndexes;

        const size_t DocCount = 0;
        const size_t TreeStart = 0;
        const size_t TreeEnd = 0;

        size_t CurrBatchStart = 0;
        size_t CurrBatchSize = 0;
        size_t CurrDocIndex = 0;
    };


    template <class TFloatFeatureAccessor, class TCatFeatureAccessor>
    THolder<NCB::NModelEvaluation::ILeafIndexCalcer> MakeLeafIndexCalcer(
        const TFullModel& model,
        const TFloatFeatureAccessor& floatAccessor,
        const TCatFeatureAccessor& catAccessor,
        size_t docCount,
        size_t treeStart,
        size_t treeEnd
    ) {
        return new TLeafIndexCalcer<TFloatFeatureAccessor, TCatFeatureAccessor>(
            model, floatAccessor, catAccessor, docCount, treeStart, treeEnd);
    }

    template <typename TCatFeatureValue = TStringBuf,
        class = typename std::enable_if<
            std::is_same_v<TCatFeatureValue, TStringBuf> || std::is_integral_v<TCatFeatureValue>>::type>
    THolder<NCB::NModelEvaluation::ILeafIndexCalcer> MakeLeafIndexCalcer(
        const TFullModel& model,
        TConstArrayRef<TConstArrayRef<float>> floatFeatures,
        TConstArrayRef<TVector<TCatFeatureValue>> catFeatures,
        size_t treeStrat,
        size_t treeEnd
    ) {
        CB_ENSURE(floatFeatures.empty() || catFeatures.empty() || floatFeatures.size() == catFeatures.size());
        const size_t docCount = Max(floatFeatures.size(), catFeatures.size());
        return MakeLeafIndexCalcer(
            model,
            [floatFeatures](const TFeaturePosition& position, size_t index) -> float {
                return floatFeatures[index][position.Index];
            },
            [catFeatures](const TFeaturePosition& position, size_t index) -> int {
                if constexpr (std::is_integral_v<TCatFeatureValue>) {
                    return catFeatures[index][position.Index];
                } else {
                    return CalcCatFeatureHash(catFeatures[index][position.Index]);
                }
            },
            docCount,
            treeStrat,
            treeEnd
        );
    }

    template <typename TCatFeatureValue = TStringBuf>
    THolder<NCB::NModelEvaluation::ILeafIndexCalcer> MakeLeafIndexCalcer(
        const TFullModel& model,
        TConstArrayRef<TConstArrayRef<float>> floatFeatures,
        TConstArrayRef<TVector<TCatFeatureValue>> catFeatures
    ) {
        return MakeLeafIndexCalcer(model, floatFeatures, catFeatures, 0, model.GetTreeCount());
    }
}
