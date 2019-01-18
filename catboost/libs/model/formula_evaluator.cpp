#include "formula_evaluator.h"

#include <util/generic/algorithm.h>
#include <util/stream/format.h>
#include <util/system/compiler.h>

#include <cstring>

#ifdef _sse2_
#include <emmintrin.h>
#include <pmmintrin.h>
#endif


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

constexpr size_t SSE_BLOCK_SIZE = 16;

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
#ifdef _sse2_

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

#ifdef _sse2_
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
#undef LOAD_LEAFS
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

template <bool IsSingleClassModel, bool NeedXorMask, int SSEBlockCount>
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
#ifdef _sse2_
    bool allTreesAreShallow = AllOf(
            model.ObliviousTrees.TreeSizes.begin() + treeStart,
            model.ObliviousTrees.TreeSizes.begin() + treeEnd,
            [](int depth) { return depth <= 8; }
    );
    if (IsSingleClassModel && allTreesAreShallow) {
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
#ifdef _sse2_
        if (curTreeSize <= 8) {
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
            CalcIndexesBasic<NeedXorMask, 0>(binFeatures, docCountInBlock, indexesVecUI32, treeSplitsCurPtr, curTreeSize);
            if (IsSingleClassModel) { // single class model
                CalculateLeafValues(docCountInBlock, treeLeafPtr + firstLeafOffsetsPtr[treeId], indexesVecUI32, resultsPtr);
            } else { // multiclass model
                CalculateLeafValuesMulti(docCountInBlock, treeLeafPtr + firstLeafOffsetsPtr[treeId], indexesVecUI32, model.ObliviousTrees.ApproxDimension, resultsPtr);
            }
        }
        treeSplitsCurPtr += curTreeSize;
    }
}

template <bool IsSingleClassModel, bool NeedXorMask>
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
        CalcTreesBlockedImpl<IsSingleClassModel, NeedXorMask, 0>(model, binFeatures, docCountInBlock, indexesVec, treeStart, treeEnd, resultsPtr);
        break;
    case 1:
        CalcTreesBlockedImpl<IsSingleClassModel, NeedXorMask, 1>(model, binFeatures, docCountInBlock, indexesVec, treeStart, treeEnd, resultsPtr);
        break;
    case 2:
        CalcTreesBlockedImpl<IsSingleClassModel, NeedXorMask, 2>(model, binFeatures, docCountInBlock, indexesVec, treeStart, treeEnd, resultsPtr);
        break;
    case 3:
        CalcTreesBlockedImpl<IsSingleClassModel, NeedXorMask, 3>(model, binFeatures, docCountInBlock, indexesVec, treeStart, treeEnd, resultsPtr);
        break;
    case 4:
        CalcTreesBlockedImpl<IsSingleClassModel, NeedXorMask, 4>(model, binFeatures, docCountInBlock, indexesVec, treeStart, treeEnd, resultsPtr);
        break;
    case 5:
        CalcTreesBlockedImpl<IsSingleClassModel, NeedXorMask, 5>(model, binFeatures, docCountInBlock, indexesVec, treeStart, treeEnd, resultsPtr);
        break;
    case 6:
        CalcTreesBlockedImpl<IsSingleClassModel, NeedXorMask, 6>(model, binFeatures, docCountInBlock, indexesVec, treeStart, treeEnd, resultsPtr);
        break;
    case 7:
        CalcTreesBlockedImpl<IsSingleClassModel, NeedXorMask, 7>(model, binFeatures, docCountInBlock, indexesVec, treeStart, treeEnd, resultsPtr);
        break;
    case 8:
        CalcTreesBlockedImpl<IsSingleClassModel, NeedXorMask, 8>(model, binFeatures, docCountInBlock, indexesVec, treeStart, treeEnd, resultsPtr);
        break;
    default:
        Y_UNREACHABLE();
    }
}

template <bool IsSingleClassModel, bool NeedXorMask>
inline void CalcTreesSingleDocImpl(
    const TFullModel& model,
    const ui8* __restrict binFeatures,
    size_t,
    TCalcerIndexType* __restrict,
    size_t treeStart,
    size_t treeEnd,
    double* __restrict results)
{
    const TRepackedBin* treeSplitsCurPtr =
        model.ObliviousTrees.GetRepackedBins().data() + model.ObliviousTrees.TreeStartOffsets[treeStart];
    double result = 0.0;
    const double* treeLeafPtr = model.ObliviousTrees.GetFirstLeafPtrForTree(treeStart);
    for (size_t treeId = treeStart; treeId < treeEnd; ++treeId) {
        const auto curTreeSize = model.ObliviousTrees.TreeSizes[treeId];
        TCalcerIndexType index = 0;
        for (int depth = 0; depth < curTreeSize; ++depth) {
            const ui8 borderVal = (ui8)(treeSplitsCurPtr[depth].SplitIdx);
            const ui32 featureIndex = (treeSplitsCurPtr[depth].FeatureIndex);
            if (NeedXorMask) {
                const ui8 xorMask = (ui8)(treeSplitsCurPtr[depth].XorMask);
                index |= ((binFeatures[featureIndex] ^ xorMask) >= borderVal) << depth;
            } else {
                index |= (binFeatures[featureIndex] >= borderVal) << depth;
            }
        }
        if (IsSingleClassModel) { // single class model
            result += treeLeafPtr[index];
        } else { // multiclass model
            auto leafValuePtr = treeLeafPtr + index * model.ObliviousTrees.ApproxDimension;
            for (int classId = 0; classId < model.ObliviousTrees.ApproxDimension; ++classId) {
                results[classId] += leafValuePtr[classId];
            }
        }
        treeLeafPtr += (1 << curTreeSize) * model.ObliviousTrees.ApproxDimension;
        treeSplitsCurPtr += curTreeSize;
    }
    if (IsSingleClassModel) {
        results[0] = result;
    }
}

TTreeCalcFunction GetCalcTreesFunction(const TFullModel& model, size_t docCountInBlock) {
    const bool hasOneHots = !model.ObliviousTrees.OneHotFeatures.empty();
    if (model.ObliviousTrees.ApproxDimension == 1) {
        if (docCountInBlock == 1) {
            if (hasOneHots) {
                return CalcTreesSingleDocImpl<true, true>;
            } else {
                return CalcTreesSingleDocImpl<true, false>;
            }
        } else {
            if (hasOneHots) {
                return CalcTreesBlocked<true, true>;
            } else {
                return CalcTreesBlocked<true, false>;
            }
        }
    } else {
        if (docCountInBlock == 1) {
            if (hasOneHots) {
                return CalcTreesSingleDocImpl<false, true>;
            } else {
                return CalcTreesSingleDocImpl<false, false>;
            }
        } else {
            if (hasOneHots) {
                return CalcTreesBlocked<false, true>;
            } else {
                return CalcTreesBlocked<false, false>;
            }
        }
    }
}
