#include "evaluator.h"

#include <library/cpp/sse/sse.h>

#include <util/generic/algorithm.h>
#include <util/stream/format.h>
#include <util/system/compiler.h>

#include <cstring>

namespace NCB::NModelEvaluation {

    constexpr size_t SSE_BLOCK_SIZE = 16;
    static_assert(SSE_BLOCK_SIZE * 8 == FORMULA_EVALUATION_BLOCK_SIZE);

    template <bool NeedXorMask, size_t START_BLOCK, typename TIndexType>
    Y_FORCE_INLINE void CalcIndexesBasic(
            const ui8* __restrict binFeatures,
            size_t docCountInBlock,
            TIndexType* __restrict indexesVec,
            const TRepackedBin* __restrict treeSplitsCurPtr,
            int curTreeSize) {
        if (START_BLOCK * SSE_BLOCK_SIZE >= docCountInBlock) {
            return;
        }
        for (int depth = 0; depth < curTreeSize; ++depth) {
            const ui8 borderVal = (ui8)(treeSplitsCurPtr[depth].SplitIdx);

            const auto featureId = treeSplitsCurPtr[depth].FeatureIndex;
            const ui8* __restrict binFeaturePtr = &binFeatures[featureId * docCountInBlock];
            const ui8 xorMask = treeSplitsCurPtr[depth].XorMask;
            if constexpr (NeedXorMask) {
                Y_PREFETCH_READ(binFeaturePtr, 3);
                Y_PREFETCH_WRITE(indexesVec, 3);
                #if defined(__clang__) && !defined(_ubsan_enabled_)
                #pragma clang loop vectorize_width(16)
                #endif
                for (size_t docId = START_BLOCK * SSE_BLOCK_SIZE; docId < docCountInBlock; ++docId) {
                    indexesVec[docId] |= ((binFeaturePtr[docId] ^ xorMask) >= borderVal) << depth;
                }
            } else {
                Y_PREFETCH_READ(binFeaturePtr, 3);
                Y_PREFETCH_WRITE(indexesVec, 3);
                #if defined(__clang__) && !defined(_ubsan_enabled_)
                #pragma clang loop vectorize_width(16)
                #endif
                for (size_t docId = START_BLOCK * SSE_BLOCK_SIZE; docId < docCountInBlock; ++docId) {
                    indexesVec[docId] |= ((binFeaturePtr[docId]) >= borderVal) << depth;
                }
            }
        }
    }

    #ifdef _sse3_

    template <bool NeedXorMask, size_t SSEBlockCount, int curTreeSize>
    Y_FORCE_INLINE void CalcIndexesSseDepthed(
            const ui8* __restrict binFeatures,
            size_t docCountInBlock,
            ui8* __restrict indexesVec,
            const TRepackedBin* __restrict treeSplitsCurPtr) {
        if constexpr (SSEBlockCount == 0) {
            CalcIndexesBasic<NeedXorMask, 0>(binFeatures, docCountInBlock, indexesVec, treeSplitsCurPtr, curTreeSize);
            return;
        }
    #define _mm_cmpge_epu8(a, b) _mm_cmpeq_epi8(_mm_max_epu8((a), (b)), (a))
    #define LOAD_16_DOC_HISTS(reg, binFeaturesPtr16) \
            const __m128i val##reg = _mm_lddqu_si128((const __m128i *)(binFeaturesPtr16));
    #define UPDATE_16_DOC_BINS(reg) \
            reg = _mm_or_si128(reg, _mm_and_si128(_mm_cmpge_epu8(val##reg, borderValVec), mask));

    #define LOAD_AND_UPDATE_16_DOCUMENT_BITS_XORED(reg, binFeaturesPtr16) \
            LOAD_16_DOC_HISTS(reg, binFeaturesPtr16);\
            reg = _mm_or_si128(reg, _mm_and_si128(_mm_cmpge_epu8(_mm_xor_si128(val##reg, xorMaskVec), borderValVec), mask));
    #define STORE_16_DOCS_RESULT(reg, addr) ((addr), reg);
        for (size_t regId = 0; regId < SSEBlockCount; regId += 2) {
            __m128i v0 = _mm_setzero_si128();
            __m128i v1 = _mm_setzero_si128();
            __m128i mask = _mm_set1_epi8(0x01);
            for (int depth = 0; depth < curTreeSize; ++depth) {
                const ui8 *__restrict binFeaturePtr = binFeatures + treeSplitsCurPtr[depth].FeatureIndex * docCountInBlock + SSE_BLOCK_SIZE * regId;
                const __m128i borderValVec = _mm_set1_epi8(treeSplitsCurPtr[depth].SplitIdx);
                if constexpr (!NeedXorMask) {
                    LOAD_16_DOC_HISTS(v0, binFeaturePtr);
                    if (regId + 1 < SSEBlockCount) {
                        LOAD_16_DOC_HISTS(v1, binFeaturePtr + SSE_BLOCK_SIZE);
                        UPDATE_16_DOC_BINS(v1);
                    }
                    UPDATE_16_DOC_BINS(v0);
                } else {
                    const __m128i xorMaskVec = _mm_set1_epi8(treeSplitsCurPtr[depth].XorMask);
                    LOAD_AND_UPDATE_16_DOCUMENT_BITS_XORED(v0, binFeaturePtr);
                    if (regId + 1 < SSEBlockCount) {
                        LOAD_AND_UPDATE_16_DOCUMENT_BITS_XORED(v1, binFeaturePtr + SSE_BLOCK_SIZE);
                    }
                }
                mask = _mm_slli_epi16(mask, 1);
            }
            _mm_storeu_si128((__m128i *)(indexesVec + SSE_BLOCK_SIZE * regId), v0);
            if (regId + 1 < SSEBlockCount) {
                _mm_storeu_si128((__m128i *)(indexesVec + SSE_BLOCK_SIZE * regId + SSE_BLOCK_SIZE), v1);
            }
        }
        if constexpr (SSEBlockCount != 8) {
            CalcIndexesBasic<NeedXorMask, SSEBlockCount>(binFeatures, docCountInBlock, indexesVec, treeSplitsCurPtr, curTreeSize);
        }
    #undef _mm_cmpge_epu8
    #undef LOAD_16_DOC_HISTS
    #undef UPDATE_16_DOC_BINS
    #undef LOAD_AND_UPDATE_16_DOCUMENT_BITS_XORED
    #undef STORE_16_DOCS_RESULT
    }

    template <bool NeedXorMask, size_t SSEBlockCount>
    static void CalcIndexesSse(
        const ui8* __restrict binFeatures,
        size_t docCountInBlock,
        ui8* __restrict indexesVec,
        const TRepackedBin* __restrict treeSplitsCurPtr,
        const int curTreeSize) {
        switch (curTreeSize)
        {
        case 1:
            CalcIndexesSseDepthed<NeedXorMask, SSEBlockCount, 1>(binFeatures, docCountInBlock, indexesVec, treeSplitsCurPtr);
            break;
        case 2:
            CalcIndexesSseDepthed<NeedXorMask, SSEBlockCount, 2>(binFeatures, docCountInBlock, indexesVec, treeSplitsCurPtr);
            break;
        case 3:
            CalcIndexesSseDepthed<NeedXorMask, SSEBlockCount, 3>(binFeatures, docCountInBlock, indexesVec, treeSplitsCurPtr);
            break;
        case 4:
            CalcIndexesSseDepthed<NeedXorMask, SSEBlockCount, 4>(binFeatures, docCountInBlock, indexesVec, treeSplitsCurPtr);
            break;
        case 5:
            CalcIndexesSseDepthed<NeedXorMask, SSEBlockCount, 5>(binFeatures, docCountInBlock, indexesVec, treeSplitsCurPtr);
            break;
        case 6:
            CalcIndexesSseDepthed<NeedXorMask, SSEBlockCount, 6>(binFeatures, docCountInBlock, indexesVec, treeSplitsCurPtr);
            break;
        case 7:
            CalcIndexesSseDepthed<NeedXorMask, SSEBlockCount, 7>(binFeatures, docCountInBlock, indexesVec, treeSplitsCurPtr);
            break;
        case 8:
            CalcIndexesSseDepthed<NeedXorMask, SSEBlockCount, 8>(binFeatures, docCountInBlock, indexesVec, treeSplitsCurPtr);
            break;
        default:
            break;
        }
    }

    #endif

    template <typename TIndexType>
    Y_FORCE_INLINE void CalculateLeafValues(const size_t docCountInBlock, const double* __restrict treeLeafPtr, const TIndexType* __restrict indexesPtr, double* __restrict writePtr) {
        Y_PREFETCH_READ(treeLeafPtr, 3);
        Y_PREFETCH_READ(treeLeafPtr + 128, 3);
        const auto docCountInBlock4 = (docCountInBlock | 0x3) ^ 0x3;
        for (size_t docId = 0; docId < docCountInBlock4; docId += 4) {
            writePtr[0] += treeLeafPtr[indexesPtr[0]];
            writePtr[1] += treeLeafPtr[indexesPtr[1]];
            writePtr[2] += treeLeafPtr[indexesPtr[2]];
            writePtr[3] += treeLeafPtr[indexesPtr[3]];
            writePtr += 4;
            indexesPtr += 4;
        }
        for (size_t docId = docCountInBlock4; docId < docCountInBlock; ++docId) {
            *writePtr += treeLeafPtr[*indexesPtr];
            ++writePtr;
            ++indexesPtr;
        }
    }

    #ifdef _sse3_
    template <int SSEBlockCount>
    Y_FORCE_INLINE static void GatherAddLeafSSE(const double* __restrict treeLeafPtr, const ui8* __restrict indexesPtr, __m128d* __restrict writePtr) {
        _mm_prefetch((const char*)(treeLeafPtr + 64), _MM_HINT_T2);

        for (size_t blockId = 0; blockId < SSEBlockCount; ++blockId) {
    #define GATHER_LEAFS(subBlock) const __m128d additions##subBlock = _mm_set_pd(treeLeafPtr[indexesPtr[subBlock * 2 + 1]], treeLeafPtr[indexesPtr[subBlock * 2 + 0]]);
    #define ADD_LEAFS(subBlock) writePtr[subBlock] = _mm_add_pd(writePtr[subBlock], additions##subBlock);

            GATHER_LEAFS(0);
            GATHER_LEAFS(1);
            GATHER_LEAFS(2);
            GATHER_LEAFS(3);
            ADD_LEAFS(0);
            ADD_LEAFS(1);
            ADD_LEAFS(2);
            ADD_LEAFS(3);

            GATHER_LEAFS(4);
            GATHER_LEAFS(5);
            GATHER_LEAFS(6);
            GATHER_LEAFS(7);
            ADD_LEAFS(4);
            ADD_LEAFS(5);
            ADD_LEAFS(6);
            ADD_LEAFS(7);
            writePtr += 8;
            indexesPtr += 16;
        }
    #undef GATHER_LEAFS
    #undef ADD_LEAFS
    }

    template <int SSEBlockCount>
    Y_FORCE_INLINE void CalculateLeafValues4(
        const size_t docCountInBlock,
        const double* __restrict treeLeafPtr0,
        const double* __restrict treeLeafPtr1,
        const double* __restrict treeLeafPtr2,
        const double* __restrict treeLeafPtr3,
        const ui8* __restrict indexesPtr0,
        const ui8* __restrict indexesPtr1,
        const ui8* __restrict indexesPtr2,
        const ui8* __restrict indexesPtr3,
        double* __restrict writePtr)
    {
        const auto docCountInBlock16 = SSEBlockCount * 16;
        if constexpr (SSEBlockCount > 0) {
            _mm_prefetch((const char*)(writePtr), _MM_HINT_T2);
            GatherAddLeafSSE<SSEBlockCount>(treeLeafPtr0, indexesPtr0, (__m128d*)writePtr);
            GatherAddLeafSSE<SSEBlockCount>(treeLeafPtr1, indexesPtr1, (__m128d*)writePtr);
            GatherAddLeafSSE<SSEBlockCount>(treeLeafPtr2, indexesPtr2, (__m128d*)writePtr);
            GatherAddLeafSSE<SSEBlockCount>(treeLeafPtr3, indexesPtr3, (__m128d*)writePtr);
        }
        if constexpr (SSEBlockCount != 8) {
            indexesPtr0 += SSE_BLOCK_SIZE * SSEBlockCount;
            indexesPtr1 += SSE_BLOCK_SIZE * SSEBlockCount;
            indexesPtr2 += SSE_BLOCK_SIZE * SSEBlockCount;
            indexesPtr3 += SSE_BLOCK_SIZE * SSEBlockCount;
            writePtr += SSE_BLOCK_SIZE * SSEBlockCount;
            for (size_t docId = docCountInBlock16; docId < docCountInBlock; ++docId) {
                *writePtr = *writePtr + treeLeafPtr0[*indexesPtr0] + treeLeafPtr1[*indexesPtr1] + treeLeafPtr2[*indexesPtr2] + treeLeafPtr3[*indexesPtr3];
                ++writePtr;
                ++indexesPtr0;
                ++indexesPtr1;
                ++indexesPtr2;
                ++indexesPtr3;
            }
        }
    }
    #endif

    template <typename TIndexType>
    Y_FORCE_INLINE void CalculateLeafValuesMulti(const size_t docCountInBlock, const double* __restrict leafPtr, const TIndexType* __restrict indexesVec, const int approxDimension, double* __restrict writePtr) {
        for (size_t docId = 0; docId < docCountInBlock; ++docId) {
            const double* __restrict leafValuePtr = leafPtr + indexesVec[docId] * approxDimension;
            for (int classId = 0; classId < approxDimension; ++classId) {
                writePtr[classId] += leafValuePtr[classId];
            }
            writePtr += approxDimension;
        }
    }

    template <bool IsSingleClassModel, bool NeedXorMask, int SSEBlockCount, bool CalcLeafIndexesOnly = false>
    Y_FORCE_INLINE void CalcTreesBlockedImpl(
        const TModelTrees& trees,
        const TModelTrees::TForApplyData& applyData,
        const ui8* __restrict binFeatures,
        const size_t docCountInBlock,
        TCalcerIndexType* __restrict indexesVecUI32,
        size_t treeStart,
        const size_t treeEnd,
        double* __restrict resultsPtr) {
        const TRepackedBin* __restrict treeSplitsCurPtr =
            trees.GetRepackedBins().data() + trees.GetModelTreeData()->GetTreeStartOffsets()[treeStart];

        ui8* __restrict indexesVec = (ui8*)indexesVecUI32;
        const double* __restrict treeLeafPtr = trees.GetModelTreeData()->GetLeafValues().data();
        const size_t* __restrict firstLeafOffsetsPtr = applyData.TreeFirstLeafOffsets.data();
    #ifdef _sse3_
        bool allTreesAreShallow = AllOf(
            trees.GetModelTreeData()->GetTreeSizes().begin() + treeStart,
            trees.GetModelTreeData()->GetTreeSizes().begin() + treeEnd,
            [](int depth) { return depth <= 8; }
        );
        if (IsSingleClassModel && !CalcLeafIndexesOnly && allTreesAreShallow) {
            auto alignedResultsPtr = resultsPtr;
            TVector<double> resultsTmpArray;
            const size_t neededMemory = docCountInBlock * trees.GetDimensionsCount() * sizeof(double);
            if ((uintptr_t)alignedResultsPtr % sizeof(__m128d) != 0) {
                if (neededMemory < 2048) {
                    alignedResultsPtr = GetAligned((double*)alloca(neededMemory + 0x20));
                } else {
                    resultsTmpArray.yresize(docCountInBlock * trees.GetDimensionsCount());
                    alignedResultsPtr = resultsTmpArray.data();
                }
                memset(alignedResultsPtr, 0, neededMemory);
            }
            auto treeEnd4 = treeStart + (((treeEnd - treeStart) | 0x3) ^ 0x3);
            for (size_t treeId = treeStart; treeId < treeEnd4; treeId += 4) {
                memset(indexesVec, 0, sizeof(ui32) * docCountInBlock);
                CalcIndexesSse<NeedXorMask, SSEBlockCount>(binFeatures, docCountInBlock, indexesVec + docCountInBlock * 0,
                                                           treeSplitsCurPtr, trees.GetModelTreeData()->GetTreeSizes()[treeId]);
                treeSplitsCurPtr += trees.GetModelTreeData()->GetTreeSizes()[treeId];
                CalcIndexesSse<NeedXorMask, SSEBlockCount>(binFeatures, docCountInBlock, indexesVec + docCountInBlock * 1,
                                                           treeSplitsCurPtr, trees.GetModelTreeData()->GetTreeSizes()[treeId + 1]);
                treeSplitsCurPtr += trees.GetModelTreeData()->GetTreeSizes()[treeId + 1];
                CalcIndexesSse<NeedXorMask, SSEBlockCount>(binFeatures, docCountInBlock, indexesVec + docCountInBlock * 2,
                                                           treeSplitsCurPtr, trees.GetModelTreeData()->GetTreeSizes()[treeId + 2]);
                treeSplitsCurPtr += trees.GetModelTreeData()->GetTreeSizes()[treeId + 2];
                CalcIndexesSse<NeedXorMask, SSEBlockCount>(binFeatures, docCountInBlock, indexesVec + docCountInBlock * 3,
                                                           treeSplitsCurPtr, trees.GetModelTreeData()->GetTreeSizes()[treeId + 3]);
                treeSplitsCurPtr += trees.GetModelTreeData()->GetTreeSizes()[treeId + 3];

                CalculateLeafValues4<SSEBlockCount>(
                    docCountInBlock,
                    treeLeafPtr + firstLeafOffsetsPtr[treeId + 0],
                    treeLeafPtr + firstLeafOffsetsPtr[treeId + 1],
                    treeLeafPtr + firstLeafOffsetsPtr[treeId + 2],
                    treeLeafPtr + firstLeafOffsetsPtr[treeId + 3],
                    indexesVec + docCountInBlock * 0,
                    indexesVec + docCountInBlock * 1,
                    indexesVec + docCountInBlock * 2,
                    indexesVec + docCountInBlock * 3,
                    alignedResultsPtr
                );
            }
            if (alignedResultsPtr != resultsPtr) {
                memcpy(resultsPtr, alignedResultsPtr, neededMemory);
            }
            treeStart = treeEnd4;
        }
#endif
        for (size_t treeId = treeStart; treeId < treeEnd; ++treeId) {
            auto curTreeSize = trees.GetModelTreeData()->GetTreeSizes()[treeId];
            memset(indexesVec, 0, sizeof(ui32) * docCountInBlock);
#ifdef _sse3_
            if (!CalcLeafIndexesOnly && curTreeSize <= 8) {
                CalcIndexesSse<NeedXorMask, SSEBlockCount>(binFeatures, docCountInBlock, indexesVec, treeSplitsCurPtr,
                                                           curTreeSize);
                if constexpr (IsSingleClassModel) { // single class model
                    CalculateLeafValues(docCountInBlock, treeLeafPtr + firstLeafOffsetsPtr[treeId], indexesVec, resultsPtr);
                } else { // multiclass model
                    CalculateLeafValuesMulti(docCountInBlock, treeLeafPtr + firstLeafOffsetsPtr[treeId], indexesVec,
                                             trees.GetDimensionsCount(), resultsPtr);
                }
            } else {
#else
            {
#endif
                CalcIndexesBasic<NeedXorMask, 0>(binFeatures, docCountInBlock, indexesVecUI32, treeSplitsCurPtr,
                                                 curTreeSize);
                if constexpr (CalcLeafIndexesOnly) {
                    indexesVecUI32 += docCountInBlock;
                    indexesVec += sizeof(ui32) * docCountInBlock;
                } else {
                    if constexpr (IsSingleClassModel) { // single class model
                        CalculateLeafValues(docCountInBlock, treeLeafPtr + firstLeafOffsetsPtr[treeId],
                                            indexesVecUI32, resultsPtr);
                    } else { // multiclass model
                        CalculateLeafValuesMulti(docCountInBlock, treeLeafPtr + firstLeafOffsetsPtr[treeId],
                                                 indexesVecUI32, trees.GetDimensionsCount(), resultsPtr);
                    }
                }
            }
            treeSplitsCurPtr += curTreeSize;
        }
    }

    template <bool IsSingleClassModel, bool NeedXorMask, bool CalcLeafIndexesOnly = false>
    Y_FORCE_INLINE void CalcTreesBlocked(
        const TModelTrees& trees,
        const TModelTrees::TForApplyData& applyData,
        const TCPUEvaluatorQuantizedData* quantizedData,
        size_t docCountInBlock,
        TCalcerIndexType* __restrict indexesVec,
        size_t treeStart,
        size_t treeEnd,
        double* __restrict resultsPtr) {
        const ui8* __restrict binFeatures = quantizedData->QuantizedData.data();
        switch (docCountInBlock / SSE_BLOCK_SIZE) {
            case 0:
                CalcTreesBlockedImpl<IsSingleClassModel, NeedXorMask, 0, CalcLeafIndexesOnly>(
                    trees, applyData, binFeatures, docCountInBlock, indexesVec, treeStart, treeEnd, resultsPtr);
                break;
            case 1:
                CalcTreesBlockedImpl<IsSingleClassModel, NeedXorMask, 1, CalcLeafIndexesOnly>(
                    trees, applyData, binFeatures, docCountInBlock, indexesVec, treeStart, treeEnd, resultsPtr);
                break;
            case 2:
                CalcTreesBlockedImpl<IsSingleClassModel, NeedXorMask, 2, CalcLeafIndexesOnly>(
                    trees, applyData, binFeatures, docCountInBlock, indexesVec, treeStart, treeEnd, resultsPtr);
                break;
            case 3:
                CalcTreesBlockedImpl<IsSingleClassModel, NeedXorMask, 3, CalcLeafIndexesOnly>(
                    trees, applyData, binFeatures, docCountInBlock, indexesVec, treeStart, treeEnd, resultsPtr);
                break;
            case 4:
                CalcTreesBlockedImpl<IsSingleClassModel, NeedXorMask, 4, CalcLeafIndexesOnly>(
                    trees, applyData, binFeatures, docCountInBlock, indexesVec, treeStart, treeEnd, resultsPtr);
                break;
            case 5:
                CalcTreesBlockedImpl<IsSingleClassModel, NeedXorMask, 5, CalcLeafIndexesOnly>(
                    trees, applyData, binFeatures, docCountInBlock, indexesVec, treeStart, treeEnd, resultsPtr);
                break;
            case 6:
                CalcTreesBlockedImpl<IsSingleClassModel, NeedXorMask, 6, CalcLeafIndexesOnly>(
                    trees, applyData, binFeatures, docCountInBlock, indexesVec, treeStart, treeEnd, resultsPtr);
                break;
            case 7:
                CalcTreesBlockedImpl<IsSingleClassModel, NeedXorMask, 7, CalcLeafIndexesOnly>(
                    trees, applyData, binFeatures, docCountInBlock, indexesVec, treeStart, treeEnd, resultsPtr);
                break;
            case 8:
                CalcTreesBlockedImpl<IsSingleClassModel, NeedXorMask, 8, CalcLeafIndexesOnly>(
                    trees, applyData, binFeatures, docCountInBlock, indexesVec, treeStart, treeEnd, resultsPtr);
                break;
            default:
                CB_ENSURE(false, "Unexpected number of SSE blocks");
        }
    }

    template <bool IsSingleClassModel, bool NeedXorMask, bool calcIndexesOnly = false>
    inline void CalcTreesSingleDocImpl(
        const TModelTrees& trees,
        const TModelTrees::TForApplyData& ,
        const TCPUEvaluatorQuantizedData* quantizedData,
        size_t,
        TCalcerIndexType* __restrict indexesVec,
        size_t treeStart,
        size_t treeEnd,
        double* __restrict results) {
        const ui8* __restrict binFeatures = quantizedData->QuantizedData.data();
        Y_ASSERT(calcIndexesOnly || (results && AllOf(results, results + trees.GetDimensionsCount(),
                                                      [](double value) { return value == 0.0; })));
        const TRepackedBin* __restrict treeSplitsCurPtr =
            trees.GetRepackedBins().data() + trees.GetModelTreeData()->GetTreeStartOffsets()[treeStart];
        const double* __restrict treeLeafPtr = trees.GetFirstLeafPtrForTree(treeStart);
        for (size_t treeId = treeStart; treeId < treeEnd; ++treeId) {
            const auto curTreeSize = trees.GetModelTreeData()->GetTreeSizes()[treeId];
            TCalcerIndexType index = 0;
            for (int depth = 0; depth < curTreeSize; ++depth) {
                const ui8 borderVal = (ui8)(treeSplitsCurPtr[depth].SplitIdx);
                const ui32 featureIndex = (treeSplitsCurPtr[depth].FeatureIndex);
                if constexpr (NeedXorMask) {
                    const ui8 xorMask = (ui8)(treeSplitsCurPtr[depth].XorMask);
                    index |= ((binFeatures[featureIndex] ^ xorMask) >= borderVal) << depth;
                } else {
                    index |= (binFeatures[featureIndex] >= borderVal) << depth;
                }
            }
            if constexpr (calcIndexesOnly) {
                *indexesVec++ = index;
            } else {
                if constexpr (IsSingleClassModel) { // single class model
                    results[0] += treeLeafPtr[index];
                } else { // multiclass model
                    const double* __restrict leafValuePtr = treeLeafPtr + index * trees.GetDimensionsCount();
                    for (int classId = 0; classId < (int)trees.GetDimensionsCount(); ++classId) {
                        results[classId] += leafValuePtr[classId];
                    }
                }
                treeLeafPtr += (1ull << curTreeSize) * trees.GetDimensionsCount();
            }
            treeSplitsCurPtr += curTreeSize;
        }
    }

    template <bool NeedXorMask>
    Y_FORCE_INLINE void CalcIndexesNonSymmetric(
        const TModelTrees& trees,
        const ui8* __restrict binFeatures,
        const size_t firstDocId,
        const size_t docCountInBlock,
        const size_t treeId,
        TCalcerIndexType* __restrict indexesVec
    ) {
        const TRepackedBin* __restrict treeSplitsPtr = trees.GetRepackedBins().data();
        const TNonSymmetricTreeStepNode* __restrict treeStepNodes = trees.GetModelTreeData()->GetNonSymmetricStepNodes().data();
        std::fill(indexesVec + firstDocId, indexesVec + docCountInBlock, trees.GetModelTreeData()->GetTreeStartOffsets()[treeId]);
        if (binFeatures == nullptr) {
            return;
        }
        size_t countStopped = 0;
        while (countStopped != docCountInBlock - firstDocId) {
            countStopped = 0;
            for (size_t docId = firstDocId; docId < docCountInBlock; ++docId) {
                const TNonSymmetricTreeStepNode* __restrict stepNode = treeStepNodes + indexesVec[docId];
                const TRepackedBin split = treeSplitsPtr[indexesVec[docId]];
                ui8 featureValue = binFeatures[split.FeatureIndex * docCountInBlock + docId];
                if constexpr (NeedXorMask) {
                    featureValue ^= split.XorMask;
                }
                const auto diff = (featureValue >= split.SplitIdx) ? stepNode->RightSubtreeDiff
                                                                   : stepNode->LeftSubtreeDiff;
                countStopped += (diff == 0);
                indexesVec[docId] += diff;
            }
        }
    }
#if defined(_sse4_1_)
    template <bool IsSingleClassModel, bool NeedXorMask, bool CalcLeafIndexesOnly = false>
    inline void CalcNonSymmetricTrees(
        const TModelTrees& trees,
        const TModelTrees::TForApplyData& applyData,
        const TCPUEvaluatorQuantizedData* quantizedData,
        size_t docCountInBlock,
        TCalcerIndexType* __restrict indexes,
        size_t treeStart,
        size_t treeEnd,
        double* __restrict resultsPtr
    ) {
        const ui8* __restrict binFeaturesI = quantizedData->QuantizedData.data();
        const TRepackedBin* __restrict treeSplitsPtr = trees.GetRepackedBins().data();
        const i32* __restrict treeStepNodes = reinterpret_cast<const i32*>(trees.GetModelTreeData()->GetNonSymmetricStepNodes().data());
        const ui32* __restrict nonSymmetricNodeIdToLeafIdPtr = trees.GetModelTreeData()->GetNonSymmetricNodeIdToLeafId().data();
        const double* __restrict leafValuesPtr = trees.GetModelTreeData()->GetLeafValues().data();
        for (size_t treeId = treeStart; treeId < treeEnd; ++treeId) {
            const ui32 treeStartIndex = trees.GetModelTreeData()->GetTreeStartOffsets()[treeId];
            __m128i* indexesVec = reinterpret_cast<__m128i*>(indexes);
            size_t docId = 0;
            // handle special case of model containing only empty splits
            for (; binFeaturesI != nullptr && docId + 8 <= docCountInBlock; docId += 8, indexesVec+=2) {
                const ui8* __restrict binFeatures = binFeaturesI + docId;
                __m128i index0 = _mm_set1_epi32(treeStartIndex);
                __m128i index1 = _mm_set1_epi32(treeStartIndex);
                __m128i diffs0, diffs1;
                do {
                    const TRepackedBin splits[8] = {
                        treeSplitsPtr[_mm_extract_epi32(index0, 0)],
                        treeSplitsPtr[_mm_extract_epi32(index0, 1)],
                        treeSplitsPtr[_mm_extract_epi32(index0, 2)],
                        treeSplitsPtr[_mm_extract_epi32(index0, 3)],
                        treeSplitsPtr[_mm_extract_epi32(index1, 0)],
                        treeSplitsPtr[_mm_extract_epi32(index1, 1)],
                        treeSplitsPtr[_mm_extract_epi32(index1, 2)],
                        treeSplitsPtr[_mm_extract_epi32(index1, 3)]
                    };
                    const __m128i zeroes = _mm_setzero_si128();
                    diffs0 = _mm_unpacklo_epi16(
                        _mm_hadd_epi16(
                            _mm_and_si128(
                                _mm_xor_si128(
                                    _mm_cmplt_epi32(
                                        _mm_xor_si128(
                                            _mm_setr_epi32(
                                                binFeatures[splits[0].FeatureIndex * docCountInBlock + 0],
                                                binFeatures[splits[1].FeatureIndex * docCountInBlock + 1],
                                                binFeatures[splits[2].FeatureIndex * docCountInBlock + 2],
                                                binFeatures[splits[3].FeatureIndex * docCountInBlock + 3]
                                            ),
                                            _mm_setr_epi32(
                                                splits[0].XorMask,
                                                splits[1].XorMask,
                                                splits[2].XorMask,
                                                splits[3].XorMask
                                            )
                                        ),
                                        _mm_setr_epi32(
                                            splits[0].SplitIdx,
                                            splits[1].SplitIdx,
                                            splits[2].SplitIdx,
                                            splits[3].SplitIdx
                                        )
                                   ),
                                   _mm_set1_epi32(0xffff0000)
                               ),
                                _mm_setr_epi32(
                                    treeStepNodes[_mm_extract_epi32(index0, 0)],
                                    treeStepNodes[_mm_extract_epi32(index0, 1)],
                                    treeStepNodes[_mm_extract_epi32(index0, 2)],
                                    treeStepNodes[_mm_extract_epi32(index0, 3)]
                                )
                            ),
                            zeroes
                        ),
                        zeroes
                    );
                    diffs1 = _mm_unpacklo_epi16(
                        _mm_hadd_epi16(
                            _mm_and_si128(
                                _mm_xor_si128(
                                    _mm_cmplt_epi32(
                                        _mm_xor_si128(
                                            _mm_setr_epi32(
                                                binFeatures[splits[4].FeatureIndex * docCountInBlock + 4],
                                                binFeatures[splits[5].FeatureIndex * docCountInBlock + 5],
                                                binFeatures[splits[6].FeatureIndex * docCountInBlock + 6],
                                                binFeatures[splits[7].FeatureIndex * docCountInBlock + 7]
                                            ),
                                            _mm_setr_epi32(
                                                splits[4].XorMask,
                                                splits[5].XorMask,
                                                splits[6].XorMask,
                                                splits[7].XorMask
                                            )
                                        ),
                                        _mm_setr_epi32(
                                            splits[4].SplitIdx,
                                            splits[5].SplitIdx,
                                            splits[6].SplitIdx,
                                            splits[7].SplitIdx
                                        )
                                    ),
                                    _mm_set1_epi32(0xffff0000)
                                ),
                                _mm_setr_epi32(
                                    treeStepNodes[_mm_extract_epi32(index1, 0)],
                                    treeStepNodes[_mm_extract_epi32(index1, 1)],
                                    treeStepNodes[_mm_extract_epi32(index1, 2)],
                                    treeStepNodes[_mm_extract_epi32(index1, 3)]
                                )
                            ),
                            zeroes
                        ),
                        zeroes
                    );
                    index0 = _mm_add_epi32(
                        index0,
                        diffs0
                    );
                    index1 = _mm_add_epi32(
                        index1,
                        diffs1
                    );
                } while (!_mm_testz_si128(diffs0, _mm_cmpeq_epi32(diffs0, diffs0)) || !_mm_testz_si128(diffs1, _mm_cmpeq_epi32(diffs1, diffs1)));
                _mm_storeu_si128(indexesVec, index0);
                _mm_storeu_si128(indexesVec + 1, index1);
            }
            if (docId < docCountInBlock) {
                CalcIndexesNonSymmetric<NeedXorMask>(trees, binFeaturesI, docId, docCountInBlock, treeId, indexes);
            }
            if constexpr (CalcLeafIndexesOnly) {
                const auto* __restrict firstLeafOffsetsPtr = applyData.TreeFirstLeafOffsets.data();
                const auto approxDimension = trees.GetDimensionsCount();
                for (docId = 0; docId < docCountInBlock; ++docId) {
                    Y_ASSERT((nonSymmetricNodeIdToLeafIdPtr[indexes[docId]] - firstLeafOffsetsPtr[treeId]) % approxDimension == 0);
                    indexes[docId] = ((nonSymmetricNodeIdToLeafIdPtr[indexes[docId]] - firstLeafOffsetsPtr[treeId]) / approxDimension);
                }
                indexes += docCountInBlock;
            } else if constexpr (IsSingleClassModel) {
                for (docId = 0; docId + 8 <= docCountInBlock; docId+=8) {
                    resultsPtr[docId + 0] += leafValuesPtr[nonSymmetricNodeIdToLeafIdPtr[indexes[docId + 0]]];
                    resultsPtr[docId + 1] += leafValuesPtr[nonSymmetricNodeIdToLeafIdPtr[indexes[docId + 1]]];
                    resultsPtr[docId + 2] += leafValuesPtr[nonSymmetricNodeIdToLeafIdPtr[indexes[docId + 2]]];
                    resultsPtr[docId + 3] += leafValuesPtr[nonSymmetricNodeIdToLeafIdPtr[indexes[docId + 3]]];
                    resultsPtr[docId + 4] += leafValuesPtr[nonSymmetricNodeIdToLeafIdPtr[indexes[docId + 4]]];
                    resultsPtr[docId + 5] += leafValuesPtr[nonSymmetricNodeIdToLeafIdPtr[indexes[docId + 5]]];
                    resultsPtr[docId + 6] += leafValuesPtr[nonSymmetricNodeIdToLeafIdPtr[indexes[docId + 6]]];
                    resultsPtr[docId + 7] += leafValuesPtr[nonSymmetricNodeIdToLeafIdPtr[indexes[docId + 7]]];
                }
                for (; docId < docCountInBlock; ++docId) {
                    resultsPtr[docId] += leafValuesPtr[nonSymmetricNodeIdToLeafIdPtr[indexes[docId]]];
                }
            } else {
                const auto approxDim = trees.GetDimensionsCount();
                auto* __restrict resultWritePtr = resultsPtr;
                for (docId = 0; docId < docCountInBlock; ++docId) {
                    const ui32 firstValueIdx = nonSymmetricNodeIdToLeafIdPtr[indexes[docId]];
                    for (int classId = 0; classId < (int)approxDim; ++classId, ++resultWritePtr) {
                        *resultWritePtr += leafValuesPtr[firstValueIdx + classId];
                    }
                }
            }
        }
    }

#else
    template <bool IsSingleClassModel, bool NeedXorMask, bool CalcLeafIndexesOnly = false>
    inline void CalcNonSymmetricTrees(
        const TModelTrees& trees,
        const TModelTrees::TForApplyData& applyData,
        const TCPUEvaluatorQuantizedData* quantizedData,
        size_t docCountInBlock,
        TCalcerIndexType* __restrict indexesVec,
        size_t treeStart,
        size_t treeEnd,
        double* __restrict resultsPtr
    ) {
        const ui8* __restrict binFeatures = quantizedData->QuantizedData.data();
        for (size_t treeId = treeStart; treeId < treeEnd; ++treeId) {
            CalcIndexesNonSymmetric<NeedXorMask>(trees, binFeatures, 0, docCountInBlock, treeId, indexesVec);
            for (size_t docId = 0; docId < docCountInBlock; ++docId) {
                indexesVec[docId] = trees.GetModelTreeData()->GetNonSymmetricNodeIdToLeafId()[indexesVec[docId]];
            }
            if constexpr (CalcLeafIndexesOnly) {
                const auto* __restrict firstLeafOffsetsPtr = applyData.TreeFirstLeafOffsets.data();
                const auto approxDimension = trees.GetDimensionsCount();
                for (size_t docId = 0; docId < docCountInBlock; ++docId) {
                    Y_ASSERT((indexesVec[docId] - firstLeafOffsetsPtr[treeId]) % approxDimension == 0);
                    indexesVec[docId] = ((indexesVec[docId] - firstLeafOffsetsPtr[treeId]) / approxDimension);
                }
                indexesVec += docCountInBlock;
            } else {
                if constexpr (IsSingleClassModel) {
                    for (size_t docId = 0; docId < docCountInBlock; ++docId) {
                        resultsPtr[docId] += trees.GetModelTreeData()->GetLeafValues()[indexesVec[docId]];
                    }
                } else {
                    auto* __restrict resultWritePtr = resultsPtr;
                    for (size_t docId = 0; docId < docCountInBlock; ++docId) {
                        const ui32 firstValueIdx = indexesVec[docId];
                        for (int classId = 0;
                             classId < (int)trees.GetDimensionsCount(); ++classId, ++resultWritePtr) {
                            *resultWritePtr += trees.GetModelTreeData()->GetLeafValues()[firstValueIdx + classId];
                        }
                    }
                }
            }
        }
    }
#endif


    template <bool IsSingleClassModel, bool NeedXorMask, bool CalcIndexesOnly>
    inline void CalcNonSymmetricTreesSingle(
        const TModelTrees& trees,
        const TModelTrees::TForApplyData& applyData,
        const TCPUEvaluatorQuantizedData* quantizedData,
        size_t,
        TCalcerIndexType* __restrict indexesVec,
        size_t treeStart,
        size_t treeEnd,
        double* __restrict resultsPtr
    ) {
        const ui8* __restrict binFeatures = quantizedData->QuantizedData.data();
        TCalcerIndexType index;
        const TRepackedBin* __restrict treeSplitsPtr = trees.GetRepackedBins().data();
        const TNonSymmetricTreeStepNode* __restrict treeStepNodes = trees.GetModelTreeData()->GetNonSymmetricStepNodes().data();
        const auto* __restrict firstLeafOffsetsPtr = applyData.TreeFirstLeafOffsets.data();
        // handle special empty-model case when there is no any splits at all
        const bool skipWork = quantizedData->QuantizedData.GetSize() == 0;
        for (size_t treeId = treeStart; treeId < treeEnd; ++treeId) {
            index = trees.GetModelTreeData()->GetTreeStartOffsets()[treeId];
            while (!skipWork) {
                const auto* __restrict stepNode = treeStepNodes + index;
                const TRepackedBin split = treeSplitsPtr[index];
                ui8 featureValue = binFeatures[split.FeatureIndex];
                if constexpr (NeedXorMask) {
                    featureValue ^= split.XorMask;
                }
                const auto diff = (featureValue >= split.SplitIdx) ? stepNode->RightSubtreeDiff
                                                                   : stepNode->LeftSubtreeDiff;
                index += diff;
                if (diff == 0) {
                    break;
                }
            }
            const ui32 firstValueIdx = trees.GetModelTreeData()->GetNonSymmetricNodeIdToLeafId()[index];
            if constexpr (CalcIndexesOnly) {
                Y_ASSERT((firstValueIdx - firstLeafOffsetsPtr[treeId]) % trees.GetDimensionsCount() == 0);
                *indexesVec++ = ((firstValueIdx - firstLeafOffsetsPtr[treeId]) / trees.GetDimensionsCount());
            } else {
                if constexpr (IsSingleClassModel) {
                    *resultsPtr += trees.GetModelTreeData()->GetLeafValues()[firstValueIdx];
                } else {
                    for (int classId = 0; classId < (int)trees.GetDimensionsCount(); ++classId) {
                        resultsPtr[classId] += trees.GetModelTreeData()->GetLeafValues()[firstValueIdx + classId];
                    }
                }
            }
        }
    }


    template <bool AreTreesOblivious, bool IsSingleDoc, bool IsSingleClassModel, bool NeedXorMask,
        bool CalcLeafIndexesOnly>
    struct CalcTreeFunctionInstantiationGetter {
        TTreeCalcFunction operator()() const {
            if constexpr (AreTreesOblivious) {
                if constexpr (IsSingleDoc) {
                    return CalcTreesSingleDocImpl<IsSingleClassModel, NeedXorMask, CalcLeafIndexesOnly>;
                } else {
                    return CalcTreesBlocked<IsSingleClassModel, NeedXorMask, CalcLeafIndexesOnly>;
                }
            } else {
                if constexpr (IsSingleDoc) {
                    return CalcNonSymmetricTreesSingle<IsSingleClassModel, NeedXorMask, CalcLeafIndexesOnly>;
                } else {
                    return CalcNonSymmetricTrees<IsSingleClassModel, NeedXorMask, CalcLeafIndexesOnly>;
                }
            }
        }
    };

    template <template <bool...> class TFunctor, bool... params>
    struct FunctorTemplateParamsSubstitutor {
        static auto Call() {
            return TFunctor<params...>()();
        }

        template <typename... Bools>
        static auto Call(bool nextParam, Bools... lastParams) {
            if (nextParam) {
                return FunctorTemplateParamsSubstitutor<TFunctor, params..., true>::Call(lastParams...);
            } else {
                return FunctorTemplateParamsSubstitutor<TFunctor, params..., false>::Call(lastParams...);
            }
        }
    };

    TTreeCalcFunction GetCalcTreesFunction(
        const TModelTrees& trees,
        size_t docCountInBlock,
        bool calcIndexesOnly
    ) {
        const bool areTreesOblivious = trees.IsOblivious();
        const bool isSingleDoc = (docCountInBlock == 1);
        const bool isSingleClassModel = (trees.GetDimensionsCount() == 1);
        const bool needXorMask = !trees.GetOneHotFeatures().empty();
        return FunctorTemplateParamsSubstitutor<CalcTreeFunctionInstantiationGetter>::Call(
            areTreesOblivious, isSingleDoc, isSingleClassModel, needXorMask, calcIndexesOnly);
    }
}
