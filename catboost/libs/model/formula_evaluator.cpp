#include "formula_evaluator.h"

#include <util/generic/algorithm.h>
#include <util/stream/format.h>
#include <util/system/compiler.h>

#include <cstring>

#include <library/sse/sse.h>

constexpr size_t SSE_BLOCK_SIZE = 16;
static_assert(SSE_BLOCK_SIZE * 8 == FORMULA_EVALUATION_BLOCK_SIZE);


void TFeatureCachedTreeEvaluator::Calc(size_t treeStart, size_t treeEnd, TArrayRef<double> results) const {
    CB_ENSURE(results.size() == DocCount * Model.ObliviousTrees.ApproxDimension);
    Fill(results.begin(), results.end(), 0.0);

    TVector<TCalcerIndexType> indexesVec(BlockSize);
    int id = 0;
    for (size_t blockStart = 0; blockStart < DocCount; blockStart += BlockSize) {
        const auto docCountInBlock = Min(BlockSize, DocCount - blockStart);
        CalcFunction(
                Model,
                BinFeatures[id].data(),
                docCountInBlock,
                indexesVec.data(),
                treeStart,
                treeEnd,
                results.data() + blockStart * Model.ObliviousTrees.ApproxDimension
        );
        ++id;
    }
}

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
        if (NeedXorMask) {
            Y_PREFETCH_READ(binFeaturePtr, 3);
            Y_PREFETCH_WRITE(indexesVec, 3);
            #pragma clang loop vectorize_width(16)
            for (size_t docId = START_BLOCK * SSE_BLOCK_SIZE; docId < docCountInBlock; ++docId) {
                indexesVec[docId] |= ((binFeaturePtr[docId] ^ xorMask) >= borderVal) << depth;
            }
        } else {
            Y_PREFETCH_READ(binFeaturePtr, 3);
            Y_PREFETCH_WRITE(indexesVec, 3);
            #pragma clang loop vectorize_width(16)
            for (size_t docId = START_BLOCK * SSE_BLOCK_SIZE; docId < docCountInBlock; ++docId) {
                indexesVec[docId] |= ((binFeaturePtr[docId]) >= borderVal) << depth;
            }
        }
    }
}

void CalcIndexes(
    bool needXorMask,
    const ui8* __restrict binFeatures,
    size_t docCountInBlock,
    ui32* __restrict indexesVec,
    const TRepackedBin* __restrict treeSplitsCurPtr,
    int curTreeSize) {
    // TODO(kirillovs): add sse dispatching here
    if (needXorMask) {
        CalcIndexesBasic<true, 0>(binFeatures, docCountInBlock, indexesVec, treeSplitsCurPtr, curTreeSize);
    } else {
        CalcIndexesBasic<false, 0>(binFeatures, docCountInBlock, indexesVec, treeSplitsCurPtr, curTreeSize);
    }
}
#ifdef ARCADIA_SSE

template <bool NeedXorMask, size_t SSEBlockCount, int curTreeSize>
Y_FORCE_INLINE void CalcIndexesSseDepthed(
        const ui8* __restrict binFeatures,
        size_t docCountInBlock,
        ui8* __restrict indexesVec,
        const TRepackedBin* __restrict treeSplitsCurPtr) {
    if (SSEBlockCount == 0) {
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
            if (!NeedXorMask) {
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
    if (SSEBlockCount != 8) {
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

#ifdef ARCADIA_SSE
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
    if (SSEBlockCount > 0) {
        _mm_prefetch((const char*)(writePtr), _MM_HINT_T2);
        GatherAddLeafSSE<SSEBlockCount>(treeLeafPtr0, indexesPtr0, (__m128d*)writePtr);
        GatherAddLeafSSE<SSEBlockCount>(treeLeafPtr1, indexesPtr1, (__m128d*)writePtr);
        GatherAddLeafSSE<SSEBlockCount>(treeLeafPtr2, indexesPtr2, (__m128d*)writePtr);
        GatherAddLeafSSE<SSEBlockCount>(treeLeafPtr3, indexesPtr3, (__m128d*)writePtr);
    }
    if (SSEBlockCount != 8) {
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
        auto leafValuePtr = leafPtr + indexesVec[docId] * approxDimension;
        for (int classId = 0; classId < approxDimension; ++classId) {
            writePtr[classId] += leafValuePtr[classId];
        }
        writePtr += approxDimension;
    }
}

template <bool IsSingleClassModel, bool NeedXorMask, int SSEBlockCount, bool CalcLeafIndexesOnly = false>
Y_FORCE_INLINE void CalcTreesBlockedImpl(
    const TFullModel& model,
    const ui8* __restrict binFeatures,
    const size_t docCountInBlock,
    TCalcerIndexType* __restrict indexesVecUI32,
    size_t treeStart,
    const size_t treeEnd,
    double* __restrict resultsPtr)
{
    const TRepackedBin* treeSplitsCurPtr =
        model.ObliviousTrees.GetRepackedBins().data() + model.ObliviousTrees.TreeStartOffsets[treeStart];

    ui8* __restrict indexesVec = (ui8*)indexesVecUI32;
    const auto treeLeafPtr = model.ObliviousTrees.LeafValues.data();
    auto firstLeafOffsetsPtr = model.ObliviousTrees.GetFirstLeafOffsets().data();
#ifdef ARCADIA_SSE
    bool allTreesAreShallow = AllOf(
            model.ObliviousTrees.TreeSizes.begin() + treeStart,
            model.ObliviousTrees.TreeSizes.begin() + treeEnd,
            [](int depth) { return depth <= 8; }
    );
    if (IsSingleClassModel && !CalcLeafIndexesOnly && allTreesAreShallow) {
        auto alignedResultsPtr = resultsPtr;
        TVector<double> resultsTmpArray;
        const size_t neededMemory = docCountInBlock * model.ObliviousTrees.ApproxDimension * sizeof(double);
        if ((uintptr_t)alignedResultsPtr % sizeof(__m128d) != 0) {
            if (neededMemory < 2048) {
                alignedResultsPtr = GetAligned((double *)alloca(neededMemory + 0x20));
            } else {
                resultsTmpArray.yresize(docCountInBlock * model.ObliviousTrees.ApproxDimension);
                alignedResultsPtr = resultsTmpArray.data();
            }
            memset(alignedResultsPtr, 0, neededMemory);
        }
        auto treeEnd4 = treeStart + (((treeEnd - treeStart) | 0x3) ^ 0x3);
        for (size_t treeId = treeStart; treeId < treeEnd4; treeId += 4) {
            memset(indexesVec, 0, sizeof(ui32) * docCountInBlock);
            CalcIndexesSse<NeedXorMask, SSEBlockCount>(binFeatures, docCountInBlock, indexesVec + docCountInBlock * 0, treeSplitsCurPtr, model.ObliviousTrees.TreeSizes[treeId]);
            treeSplitsCurPtr += model.ObliviousTrees.TreeSizes[treeId];
            CalcIndexesSse<NeedXorMask, SSEBlockCount>(binFeatures, docCountInBlock, indexesVec + docCountInBlock * 1, treeSplitsCurPtr, model.ObliviousTrees.TreeSizes[treeId + 1]);
            treeSplitsCurPtr += model.ObliviousTrees.TreeSizes[treeId + 1];
            CalcIndexesSse<NeedXorMask, SSEBlockCount>(binFeatures, docCountInBlock, indexesVec + docCountInBlock * 2, treeSplitsCurPtr, model.ObliviousTrees.TreeSizes[treeId + 2]);
            treeSplitsCurPtr += model.ObliviousTrees.TreeSizes[treeId + 2];
            CalcIndexesSse<NeedXorMask, SSEBlockCount>(binFeatures, docCountInBlock, indexesVec + docCountInBlock * 3, treeSplitsCurPtr, model.ObliviousTrees.TreeSizes[treeId + 3]);
            treeSplitsCurPtr += model.ObliviousTrees.TreeSizes[treeId + 3];

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
        auto curTreeSize = model.ObliviousTrees.TreeSizes[treeId];
        memset(indexesVec, 0, sizeof(ui32) * docCountInBlock);
#ifdef ARCADIA_SSE
        if (!CalcLeafIndexesOnly && curTreeSize <= 8) {
            CalcIndexesSse<NeedXorMask, SSEBlockCount>(binFeatures, docCountInBlock, indexesVec, treeSplitsCurPtr, curTreeSize);
            if (IsSingleClassModel) { // single class model
                CalculateLeafValues(docCountInBlock, treeLeafPtr + firstLeafOffsetsPtr[treeId], indexesVec, resultsPtr);
            } else { // multiclass model
                CalculateLeafValuesMulti(docCountInBlock, treeLeafPtr + firstLeafOffsetsPtr[treeId], indexesVec, model.ObliviousTrees.ApproxDimension, resultsPtr);
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
                if (IsSingleClassModel) { // single class model
                    CalculateLeafValues(docCountInBlock, treeLeafPtr + firstLeafOffsetsPtr[treeId],
                                        indexesVecUI32, resultsPtr);
                } else { // multiclass model
                    CalculateLeafValuesMulti(docCountInBlock, treeLeafPtr + firstLeafOffsetsPtr[treeId],
                                             indexesVecUI32, model.ObliviousTrees.ApproxDimension, resultsPtr);
                }
            }
        }
        treeSplitsCurPtr += curTreeSize;
    }
}

template <bool IsSingleClassModel, bool NeedXorMask, bool CalcLeafIndexesOnly = false>
Y_FORCE_INLINE void CalcTreesBlocked(
    const TFullModel& model,
    const ui8* __restrict binFeatures,
    size_t docCountInBlock,
    TCalcerIndexType* __restrict indexesVec,
    size_t treeStart,
    size_t treeEnd,
    double* __restrict resultsPtr)
{
    switch (docCountInBlock / SSE_BLOCK_SIZE) {
    case 0:
        CalcTreesBlockedImpl<IsSingleClassModel, NeedXorMask, 0, CalcLeafIndexesOnly>(
            model, binFeatures, docCountInBlock, indexesVec, treeStart, treeEnd, resultsPtr);
        break;
    case 1:
        CalcTreesBlockedImpl<IsSingleClassModel, NeedXorMask, 1, CalcLeafIndexesOnly>(
            model, binFeatures, docCountInBlock, indexesVec, treeStart, treeEnd, resultsPtr);
        break;
    case 2:
        CalcTreesBlockedImpl<IsSingleClassModel, NeedXorMask, 2, CalcLeafIndexesOnly>(
            model, binFeatures, docCountInBlock, indexesVec, treeStart, treeEnd, resultsPtr);
        break;
    case 3:
        CalcTreesBlockedImpl<IsSingleClassModel, NeedXorMask, 3, CalcLeafIndexesOnly>(
            model, binFeatures, docCountInBlock, indexesVec, treeStart, treeEnd, resultsPtr);
        break;
    case 4:
        CalcTreesBlockedImpl<IsSingleClassModel, NeedXorMask, 4, CalcLeafIndexesOnly>(
            model, binFeatures, docCountInBlock, indexesVec, treeStart, treeEnd, resultsPtr);
        break;
    case 5:
        CalcTreesBlockedImpl<IsSingleClassModel, NeedXorMask, 5, CalcLeafIndexesOnly>(
            model, binFeatures, docCountInBlock, indexesVec, treeStart, treeEnd, resultsPtr);
        break;
    case 6:
        CalcTreesBlockedImpl<IsSingleClassModel, NeedXorMask, 6, CalcLeafIndexesOnly>(
            model, binFeatures, docCountInBlock, indexesVec, treeStart, treeEnd, resultsPtr);
        break;
    case 7:
        CalcTreesBlockedImpl<IsSingleClassModel, NeedXorMask, 7, CalcLeafIndexesOnly>(
            model, binFeatures, docCountInBlock, indexesVec, treeStart, treeEnd, resultsPtr);
        break;
    case 8:
        CalcTreesBlockedImpl<IsSingleClassModel, NeedXorMask, 8, CalcLeafIndexesOnly>(
            model, binFeatures, docCountInBlock, indexesVec, treeStart, treeEnd, resultsPtr);
        break;
    default:
        Y_UNREACHABLE();
    }
}

template <bool IsSingleClassModel, bool NeedXorMask, bool calcIndexesOnly = false>
inline void CalcTreesSingleDocImpl(
    const TFullModel& model,
    const ui8* __restrict binFeatures,
    size_t,
    TCalcerIndexType* __restrict indexesVec,
    size_t treeStart,
    size_t treeEnd,
    double* __restrict results)
{
    Y_ASSERT(!calcIndexesOnly || (indexesVec && AllOf(indexesVec, indexesVec + (treeEnd - treeStart),
            [] (TCalcerIndexType index) { return index == 0; })));
    Y_ASSERT(calcIndexesOnly || (results && AllOf(results, results + model.ObliviousTrees.ApproxDimension,
            [] (double value) { return value == 0.0; })));
    const TRepackedBin* treeSplitsCurPtr =
        model.ObliviousTrees.GetRepackedBins().data() + model.ObliviousTrees.TreeStartOffsets[treeStart];
    const double* treeLeafPtr = model.ObliviousTrees.GetFirstLeafPtrForTree(treeStart);
    for (size_t treeId = treeStart; treeId < treeEnd; ++treeId) {
        const auto curTreeSize = model.ObliviousTrees.TreeSizes[treeId];
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
            Y_ASSERT(*indexesVec == 0);
            *indexesVec++ = index;
        } else {
            if constexpr (IsSingleClassModel) { // single class model
                results[0] += treeLeafPtr[index];
            } else { // multiclass model
                auto leafValuePtr = treeLeafPtr + index * model.ObliviousTrees.ApproxDimension;
                for (int classId = 0; classId < model.ObliviousTrees.ApproxDimension; ++classId) {
                    results[classId] += leafValuePtr[classId];
                }
            }
            treeLeafPtr += (1 << curTreeSize) * model.ObliviousTrees.ApproxDimension;
        }
        treeSplitsCurPtr += curTreeSize;
    }
}

template <bool NeedXorMask>
Y_FORCE_INLINE void CalcIndexesNonSymmetric(
    const TFullModel& model,
    const ui8* __restrict binFeatures,
    const size_t docCountInBlock,
    const size_t treeId,
    TCalcerIndexType* __restrict indexesVec
) {
    const TRepackedBin* treeSplitsPtr = model.ObliviousTrees.GetRepackedBins().data();
    const TNonSymmetricTreeStepNode* treeStepNodes = model.ObliviousTrees.NonSymmetricStepNodes.data();
    std::fill(indexesVec, indexesVec + docCountInBlock, model.ObliviousTrees.TreeStartOffsets[treeId]);
    size_t countStopped = 0;
    while (countStopped != docCountInBlock) {
        countStopped = 0;
        for (size_t docId = 0; docId < docCountInBlock; ++docId) {
            const auto* stepNode = treeStepNodes + indexesVec[docId];
            if (stepNode->IsTerminalNode()) {
                ++countStopped;
                continue;
            }
            const TRepackedBin split = treeSplitsPtr[indexesVec[docId]];
            ui8 featureValue = binFeatures[split.FeatureIndex * docCountInBlock + docId];
            if constexpr (NeedXorMask) {
                featureValue ^= split.XorMask;
            }
            const auto diff = (featureValue >= split.SplitIdx) ? stepNode->RightSubtreeDiff : stepNode->LeftSubtreeDiff;
            countStopped += (diff == 0);
            indexesVec[docId] += diff;
        }
    }
    for (size_t docId = 0; docId < docCountInBlock; ++docId) {
        indexesVec[docId] = model.ObliviousTrees.NonSymmetricNodeIdToLeafId[indexesVec[docId]];
    }
}

template <bool IsSingleClassModel, bool NeedXorMask, bool CalcLeafIndexesOnly = false>
inline void CalcNonSymmetricTreesSimple(
    const TFullModel& model,
    const ui8* __restrict binFeatures,
    size_t docCountInBlock,
    TCalcerIndexType* __restrict indexesVec,
    size_t treeStart,
    size_t treeEnd,
    double* __restrict resultsPtr
) {
    for (size_t treeId = treeStart; treeId < treeEnd; ++treeId) {
        CalcIndexesNonSymmetric<NeedXorMask>(model, binFeatures, docCountInBlock, treeId, indexesVec);
        if constexpr (CalcLeafIndexesOnly) {
            const auto firstLeafOffsets = model.ObliviousTrees.GetFirstLeafOffsets();
            const auto approxDimension = model.ObliviousTrees.ApproxDimension;
            for (size_t docId = 0; docId < docCountInBlock; ++docId) {
                Y_ASSERT((indexesVec[docId] - firstLeafOffsets[treeId]) % approxDimension == 0);
                indexesVec[docId] = ((indexesVec[docId] - firstLeafOffsets[treeId]) / approxDimension);
            }
            indexesVec += docCountInBlock;
        } else {
            if constexpr (IsSingleClassModel) {
                for (size_t docId = 0; docId < docCountInBlock; ++docId) {
                    resultsPtr[docId] += model.ObliviousTrees.LeafValues[indexesVec[docId]];
                }
            } else {
                auto resultWritePtr = resultsPtr;
                for (size_t docId = 0; docId < docCountInBlock; ++docId) {
                    const ui32 firstValueIdx = indexesVec[docId];
                    for (int classId = 0;
                         classId < model.ObliviousTrees.ApproxDimension; ++classId, ++resultWritePtr) {
                        *resultWritePtr += model.ObliviousTrees.LeafValues[firstValueIdx + classId];
                    }
                }
            }
        }
    }
}

template <bool IsSingleClassModel, bool NeedXorMask, bool CalcIndexesOnly>
inline void CalcNonSymmetricTreesSingle(
    const TFullModel& model,
    const ui8* __restrict binFeatures,
    size_t ,
    TCalcerIndexType* __restrict indexesVec,
    size_t treeStart,
    size_t treeEnd,
    double* __restrict resultsPtr
) {
    TCalcerIndexType index;
    const TRepackedBin* treeSplitsPtr = model.ObliviousTrees.GetRepackedBins().data();
    const TNonSymmetricTreeStepNode* treeStepNodes = model.ObliviousTrees.NonSymmetricStepNodes.data();
    const auto firstLeafOffsets = model.ObliviousTrees.GetFirstLeafOffsets();
    for (size_t treeId = treeStart; treeId < treeEnd; ++treeId) {
        index = model.ObliviousTrees.TreeStartOffsets[treeId];
        while (true) {
            const auto* stepNode = treeStepNodes + index;
            if (stepNode->IsTerminalNode()) {
                break;
            }
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
        const ui32 firstValueIdx = model.ObliviousTrees.NonSymmetricNodeIdToLeafId[index];
        if constexpr (CalcIndexesOnly) {
            Y_ASSERT((firstValueIdx - firstLeafOffsets[treeId]) % model.ObliviousTrees.ApproxDimension == 0);
            *indexesVec++ = ((firstValueIdx - firstLeafOffsets[treeId]) / model.ObliviousTrees.ApproxDimension);
        } else {
            if constexpr (IsSingleClassModel) {
                *resultsPtr += model.ObliviousTrees.LeafValues[firstValueIdx];
            } else {
                for (int classId = 0; classId < model.ObliviousTrees.ApproxDimension; ++classId) {
                    resultsPtr[classId] += model.ObliviousTrees.LeafValues[firstValueIdx + classId];
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
                return CalcNonSymmetricTreesSimple<IsSingleClassModel, NeedXorMask, CalcLeafIndexesOnly>;
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
    const TFullModel& model,
    size_t docCountInBlock,
    bool calcIndexesOnly
) {
    const bool areTreesOblivious = model.ObliviousTrees.IsOblivious();
    const bool isSingleDoc = (docCountInBlock == 1);
    const bool isSingleClassModel = (model.ObliviousTrees.ApproxDimension == 1);
    const bool needXorMask = !model.ObliviousTrees.OneHotFeatures.empty();
    return FunctorTemplateParamsSubstitutor<CalcTreeFunctionInstantiationGetter>::Call(
        areTreesOblivious, isSingleDoc, isSingleClassModel, needXorMask, calcIndexesOnly);
}
