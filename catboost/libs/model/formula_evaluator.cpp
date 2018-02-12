#include "formula_evaluator.h"

#include <util/stream/format.h>

#include <emmintrin.h>

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

template<bool NeedXorMask, size_t START_BLOCK, typename TIndexType>
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

template<bool NeedXorMask, size_t SSEBlockCount>
Y_FORCE_INLINE void CalcIndexesSse(
        const ui8* __restrict binFeatures,
        size_t docCountInBlock,
        ui8* __restrict indexesVec,
        const TRepackedBin* __restrict treeSplitsCurPtr,
        const int curTreeSize) {
    if (SSEBlockCount == 0) {
        CalcIndexesBasic<NeedXorMask, 0>(binFeatures, docCountInBlock, indexesVec, treeSplitsCurPtr, curTreeSize);
        return;
    }
    //const __m128i zeroes = _mm_setzero_si128();
    __m128i v0 = _mm_setzero_si128();
    __m128i v1 = _mm_setzero_si128();
    __m128i v2 = _mm_setzero_si128();
    __m128i v3 = _mm_setzero_si128();
    __m128i v4 = _mm_setzero_si128();
    __m128i v5 = _mm_setzero_si128();
    __m128i v6 = _mm_setzero_si128();
    __m128i v7 = _mm_setzero_si128();
    __m128i mask = _mm_set1_epi8(0x01);
    for (int depth = 0; depth < curTreeSize; ++depth) {
        const ui8* __restrict binFeaturePtr = binFeatures + treeSplitsCurPtr[depth].FeatureIndex * docCountInBlock;
#define _mm_cmpge_epu8(a, b) _mm_cmpeq_epi8(_mm_max_epu8((a), (b)), (a))

        const __m128i borderValVec = _mm_set1_epi8(treeSplitsCurPtr[depth].SplitIdx);
#define update_16_documents_bits(reg, binFeaturesPtr16) \
        {__m128i val = _mm_loadu_si128((const __m128i *)(binFeaturesPtr16));\
        reg = _mm_or_si128(reg, _mm_and_si128(_mm_cmpge_epu8(val, borderValVec), mask));}

#define update_16_documents_bits_xored(reg, binFeaturesPtr16) \
        {__m128i val = _mm_loadu_si128((const __m128i *)(binFeaturesPtr16));\
        reg = _mm_or_si128(reg, _mm_and_si128(_mm_cmpge_epu8(_mm_xor_si128(val, xorMaskVec), borderValVec), mask));}

        if (!NeedXorMask) {
            if (SSEBlockCount > 0) {
                update_16_documents_bits(v0, binFeaturePtr + 16 * 0);
            }
            if (SSEBlockCount > 1) {
                update_16_documents_bits(v1, binFeaturePtr + 16 * 1);
            }
            if (SSEBlockCount > 2) {
                update_16_documents_bits(v2, binFeaturePtr + 16 * 2);
            }
            if (SSEBlockCount > 3) {
                update_16_documents_bits(v3, binFeaturePtr + 16 * 3);
            }
            if (SSEBlockCount > 4) {
                update_16_documents_bits(v4, binFeaturePtr + 16 * 4);
            }
            if (SSEBlockCount > 5) {
                update_16_documents_bits(v5, binFeaturePtr + 16 * 5);
            }
            if (SSEBlockCount > 6) {
                update_16_documents_bits(v6, binFeaturePtr + 16 * 6);
            }
            if (SSEBlockCount > 7) {
                update_16_documents_bits(v7, binFeaturePtr + 16 * 7);
            }
        } else {
            const __m128i xorMaskVec = _mm_set1_epi8(treeSplitsCurPtr[depth].XorMask);
            if (SSEBlockCount > 0) {
                update_16_documents_bits_xored(v0, binFeaturePtr + 16 * 0);
            }
            if (SSEBlockCount > 1) {
                update_16_documents_bits_xored(v1, binFeaturePtr + 16 * 1);
            }
            if (SSEBlockCount > 2) {
                update_16_documents_bits_xored(v2, binFeaturePtr + 16 * 2);
            }
            if (SSEBlockCount > 3) {
                update_16_documents_bits_xored(v3, binFeaturePtr + 16 * 3);
            }
            if (SSEBlockCount > 4) {
                update_16_documents_bits_xored(v4, binFeaturePtr + 16 * 4);
            }
            if (SSEBlockCount > 5) {
                update_16_documents_bits_xored(v5, binFeaturePtr + 16 * 5);
            }
            if (SSEBlockCount > 6) {
                update_16_documents_bits_xored(v6, binFeaturePtr + 16 * 6);
            }
            if (SSEBlockCount > 7) {
                update_16_documents_bits_xored(v7, binFeaturePtr + 16 * 7);
            }
        }
        mask = _mm_slli_epi16(mask, 1);
    }
#define store_16_documents_results_ui32(reg, addr) \
        {__m128i unpacked = _mm_unpacklo_epi8(reg, zeroes);\
        _mm_store_si128((__m128i *)(addr + 4 * 0), _mm_unpacklo_epi16(unpacked, zeroes));\
        _mm_store_si128((__m128i *)(addr + 4 * 1), _mm_unpackhi_epi16(unpacked, zeroes));\
        unpacked = _mm_unpackhi_epi8(reg, zeroes);\
        _mm_store_si128((__m128i *)(addr + 4 * 2), _mm_unpacklo_epi16(unpacked, zeroes));\
        _mm_store_si128((__m128i *)(addr + 4 * 3), _mm_unpackhi_epi16(unpacked, zeroes));}

#define store_16_documents_results(reg, addr) _mm_storeu_si128((__m128i *)(addr), reg);

    if (SSEBlockCount > 0) {
        store_16_documents_results(v0, (indexesVec + 16 * 0));
    }
    if (SSEBlockCount > 1) {
        store_16_documents_results(v1, (indexesVec + 16 * 1));
    }
    if (SSEBlockCount > 2) {
        store_16_documents_results(v2, (indexesVec + 16 * 2));
    }
    if (SSEBlockCount > 3) {
        store_16_documents_results(v3, (indexesVec + 16 * 3));
    }
    if (SSEBlockCount > 4) {
        store_16_documents_results(v4, (indexesVec + 16 * 4));
    }
    if (SSEBlockCount > 5) {
        store_16_documents_results(v5, (indexesVec + 16 * 5));
    }
    if (SSEBlockCount > 6) {
        store_16_documents_results(v6, (indexesVec + 16 * 6));
    }
    if (SSEBlockCount > 7) {
        store_16_documents_results(v7, (indexesVec + 16 * 7));
    }
    if (SSEBlockCount != 8) {
        CalcIndexesBasic<NeedXorMask, SSEBlockCount>(binFeatures, docCountInBlock, indexesVec, treeSplitsCurPtr, curTreeSize);
    }
}

template<typename TIndexType>
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
    Y_PREFETCH_READ(treeLeafPtr0, 3);
    Y_PREFETCH_READ(treeLeafPtr1, 3);
    Y_PREFETCH_READ(treeLeafPtr2, 3);
    Y_PREFETCH_READ(treeLeafPtr3, 3);
    const auto docCountInBlock4 = (docCountInBlock | 0x3) ^ 0x3;
    for (size_t docId = 0; docId < docCountInBlock4; docId += 4) {
        writePtr[0] = writePtr[0] + treeLeafPtr0[indexesPtr0[0]] + treeLeafPtr1[indexesPtr1[0]] + treeLeafPtr2[indexesPtr2[0]] + treeLeafPtr3[indexesPtr3[0]];
        writePtr[1] = writePtr[1] + treeLeafPtr0[indexesPtr0[1]] + treeLeafPtr1[indexesPtr1[1]] + treeLeafPtr2[indexesPtr2[1]] + treeLeafPtr3[indexesPtr3[1]];
        writePtr[2] = writePtr[2] + treeLeafPtr0[indexesPtr0[2]] + treeLeafPtr1[indexesPtr1[2]] + treeLeafPtr2[indexesPtr2[2]] + treeLeafPtr3[indexesPtr3[2]];
        writePtr[3] = writePtr[3] + treeLeafPtr0[indexesPtr0[3]] + treeLeafPtr1[indexesPtr1[3]] + treeLeafPtr2[indexesPtr2[3]] + treeLeafPtr3[indexesPtr3[3]];
        writePtr += 4;
        indexesPtr0 += 4;
        indexesPtr1 += 4;
        indexesPtr2 += 4;
        indexesPtr3 += 4;
    }
    for (size_t docId = docCountInBlock4; docId < docCountInBlock; ++docId) {
        *writePtr = *writePtr + treeLeafPtr0[*indexesPtr0] + treeLeafPtr1[*indexesPtr1] + treeLeafPtr2[*indexesPtr2] + treeLeafPtr3[*indexesPtr3];
        ++writePtr;
        ++indexesPtr0;
        ++indexesPtr1;
        ++indexesPtr2;
        ++indexesPtr3;
    }
}

template<typename TIndexType>
Y_FORCE_INLINE void CalculateLeafValuesMulti(const size_t docCountInBlock, const double* __restrict leafPtr, const TIndexType* __restrict indexesVec, const int approxDimension, double* __restrict writePtr) {
    for (size_t docId = 0; docId < docCountInBlock; ++docId) {
        auto leafValuePtr = leafPtr + indexesVec[docId] * approxDimension;
        for (int classId = 0; classId < approxDimension; ++classId) {
            writePtr[classId] += leafValuePtr[classId];
        }
        writePtr += approxDimension;
    }
}

template<bool IsSingleClassModel, bool NeedXorMask, int SSEBlockCount>
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

    bool allTreesAreShallow = AllOf(
        model.ObliviousTrees.TreeSizes.begin() + treeStart,
        model.ObliviousTrees.TreeSizes.begin() + treeEnd,
        [](int depth) { return depth <= 8; }
    );
    ui8* __restrict indexesVec = (ui8*)indexesVecUI32;
    if (IsSingleClassModel && allTreesAreShallow) {
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

            CalculateLeafValues4(
                docCountInBlock,
                model.ObliviousTrees.LeafValues[treeId + 0].data(),
                model.ObliviousTrees.LeafValues[treeId + 1].data(),
                model.ObliviousTrees.LeafValues[treeId + 2].data(),
                model.ObliviousTrees.LeafValues[treeId + 3].data(),
                indexesVec + docCountInBlock * 0,
                indexesVec + docCountInBlock * 1,
                indexesVec + docCountInBlock * 2,
                indexesVec + docCountInBlock * 3,
                resultsPtr
            );
        }
        treeStart = treeEnd4;
    }
    for (size_t treeId = treeStart; treeId < treeEnd; ++treeId) {
        auto curTreeSize = model.ObliviousTrees.TreeSizes[treeId];
        memset(indexesVec, 0, sizeof(ui32) * docCountInBlock);
        if (curTreeSize <= 8) {
            CalcIndexesSse<NeedXorMask, SSEBlockCount>(binFeatures, docCountInBlock, indexesVec, treeSplitsCurPtr, curTreeSize);
            if (IsSingleClassModel) { // single class model
                CalculateLeafValues(docCountInBlock, model.ObliviousTrees.LeafValues[treeId].data(), indexesVec, resultsPtr);
            } else { // mutliclass model
                CalculateLeafValuesMulti(docCountInBlock, model.ObliviousTrees.LeafValues[treeId].data(), indexesVec, model.ObliviousTrees.ApproxDimension, resultsPtr);
            }
        } else {
            CalcIndexesBasic<NeedXorMask, 0>(binFeatures, docCountInBlock, indexesVecUI32, treeSplitsCurPtr, curTreeSize);
            if (IsSingleClassModel) { // single class model
                CalculateLeafValues(docCountInBlock, model.ObliviousTrees.LeafValues[treeId].data(), indexesVecUI32, resultsPtr);
            } else { // mutliclass model
                CalculateLeafValuesMulti(docCountInBlock, model.ObliviousTrees.LeafValues[treeId].data(), indexesVecUI32, model.ObliviousTrees.ApproxDimension, resultsPtr);
            }
        }
        treeSplitsCurPtr += curTreeSize;
    }
}

template<bool IsSingleClassModel, bool NeedXorMask>
inline void CalcTreesBlocked(
    const TFullModel& model,
    const ui8* __restrict binFeatures,
    size_t docCountInBlock,
    TCalcerIndexType* __restrict indexesVec,
    size_t treeStart,
    size_t treeEnd,
    double* __restrict resultsPtr) {
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

template<bool IsSingleClassModel, bool NeedXorMask>
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
    for (size_t treeId = treeStart; treeId < treeEnd; ++treeId) {
        auto curTreeSize = model.ObliviousTrees.TreeSizes[treeId];
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
        auto treeLeafPtr = model.ObliviousTrees.LeafValues[treeId].data();
        if (IsSingleClassModel) { // single class model
            result += treeLeafPtr[index];
        } else { // mutliclass model
            auto leafValuePtr = treeLeafPtr + index * model.ObliviousTrees.ApproxDimension;
            for (int classId = 0; classId < model.ObliviousTrees.ApproxDimension; ++classId) {
                results[classId] += leafValuePtr[classId];
            }
        }
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
