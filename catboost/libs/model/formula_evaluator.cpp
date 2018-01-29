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
                blockStart,
                BinFeatures[id].data(),
                docCountInBlock,
                indexesVec.data(),
                treeStart,
                treeEnd,
                results.data()
        );
        ++id;
    }
}

constexpr size_t SSE_BLOCK_SIZE = 16;

template<bool NeedXorMask, size_t START_BLOCK>
Y_FORCE_INLINE void CalcIndexesBasic(
        const ui8* __restrict binFeatures,
        size_t docCountInBlock,
        ui32* __restrict indexesVec,
        const ui32* __restrict treeSplitsCurPtr,
        int curTreeSize) {
    if (START_BLOCK * SSE_BLOCK_SIZE >= docCountInBlock) {
        return;
    }
    for (int depth = 0; depth < curTreeSize; ++depth) {
        const ui8 borderVal = (ui8)(treeSplitsCurPtr[depth] & 0xff);

        const auto featureId = treeSplitsCurPtr[depth] >> 16;
        const ui8* __restrict binFeaturePtr = &binFeatures[featureId * docCountInBlock];
        const ui8 xorMask = (ui8)((treeSplitsCurPtr[depth] & 0xff00) >> 8);
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
    const ui32* __restrict treeSplitsCurPtr,
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
        ui32* __restrict indexesVec,
        const ui32* __restrict treeSplitsCurPtr,
        int curTreeSize) {
    const __m128i zeroes = _mm_setzero_si128();
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
        const ui8 borderVal = (ui8)(treeSplitsCurPtr[depth] & 0xff);
        const auto featureId = (treeSplitsCurPtr[depth] >> 16);
        const ui8* __restrict binFeaturePtr = &binFeatures[featureId * docCountInBlock];
        #define _mm_cmpge_epu8(a, b) _mm_cmpeq_epi8(_mm_max_epu8(a, b), a)

        const __m128i borderValVec = _mm_set1_epi8(borderVal);
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
            const ui8 xorMask = (ui8)((treeSplitsCurPtr[depth] & 0xff00) >> 8);
            const __m128i xorMaskVec = _mm_set1_epi8(xorMask);
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
#define store_16_documents_results(reg, addr) \
        {__m128i unpacked = _mm_unpacklo_epi8(reg, zeroes);\
        _mm_store_si128((__m128i *)(addr + 4 * 0), _mm_unpacklo_epi16(unpacked, zeroes));\
        _mm_store_si128((__m128i *)(addr + 4 * 1), _mm_unpackhi_epi16(unpacked, zeroes));\
        unpacked = _mm_unpackhi_epi8(reg, zeroes);\
        _mm_store_si128((__m128i *)(addr + 4 * 2), _mm_unpacklo_epi16(unpacked, zeroes));\
        _mm_store_si128((__m128i *)(addr + 4 * 3), _mm_unpackhi_epi16(unpacked, zeroes));}

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

template<bool IsSingleClassModel, bool IsSingleDocCase, bool NeedXorMask>
inline void CalcTreesImpl(
    const TFullModel& model,
    size_t blockStart,
    const ui8* __restrict binFeatures,
    size_t docCountInBlock,
    TCalcerIndexType* __restrict indexesVec,
    size_t treeStart,
    size_t treeEnd,
    double* __restrict results)
{
    const auto docCountInBlock4 = (docCountInBlock | 0x3) ^0x3;
    const ui32* treeSplitsCurPtr =
        model.ObliviousTrees.GetRepackedBins().data() +
            model.ObliviousTrees.TreeStartOffsets[treeStart];
    if (!IsSingleDocCase) {
        for (size_t treeId = treeStart; treeId < treeEnd; ++treeId) {
            auto curTreeSize = model.ObliviousTrees.TreeSizes[treeId];

            memset(indexesVec, 0, sizeof(ui32) * docCountInBlock);
            if (curTreeSize <= 8) {
                switch (docCountInBlock / SSE_BLOCK_SIZE) {
                case 0:
                    CalcIndexesBasic<NeedXorMask, 0>(binFeatures, docCountInBlock, indexesVec, treeSplitsCurPtr, curTreeSize);
                    break;
                case 1:
                    CalcIndexesSse<NeedXorMask, 1>(binFeatures, docCountInBlock, indexesVec, treeSplitsCurPtr, curTreeSize);
                    break;
                case 2:
                    CalcIndexesSse<NeedXorMask, 2>(binFeatures, docCountInBlock, indexesVec, treeSplitsCurPtr, curTreeSize);
                    break;
                case 3:
                    CalcIndexesSse<NeedXorMask, 3>(binFeatures, docCountInBlock, indexesVec, treeSplitsCurPtr, curTreeSize);
                    break;
                case 4:
                    CalcIndexesSse<NeedXorMask, 4>(binFeatures, docCountInBlock, indexesVec, treeSplitsCurPtr, curTreeSize);
                    break;
                case 5:
                    CalcIndexesSse<NeedXorMask, 5>(binFeatures, docCountInBlock, indexesVec, treeSplitsCurPtr, curTreeSize);
                    break;
                case 6:
                    CalcIndexesSse<NeedXorMask, 6>(binFeatures, docCountInBlock, indexesVec, treeSplitsCurPtr, curTreeSize);
                    break;
                case 7:
                    CalcIndexesSse<NeedXorMask, 7>(binFeatures, docCountInBlock, indexesVec, treeSplitsCurPtr, curTreeSize);
                    break;
                case 8:
                    CalcIndexesSse<NeedXorMask, 8>(binFeatures, docCountInBlock, indexesVec, treeSplitsCurPtr, curTreeSize);
                    break;
                }
            } else {
                CalcIndexesBasic<NeedXorMask, 0>(binFeatures, docCountInBlock, indexesVec, treeSplitsCurPtr, curTreeSize);
            }
            auto treeLeafPtr = model.ObliviousTrees.LeafValues[treeId].data();
            if (IsSingleClassModel) { // single class model
                const ui32* __restrict indexesPtr = indexesVec;
                double* __restrict writePtr = &results[blockStart];
                Y_PREFETCH_READ(treeLeafPtr, 3);
                Y_PREFETCH_READ(treeLeafPtr + 128, 3);
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
            } else { // mutliclass model
                auto docResultPtr = &results[blockStart * model.ObliviousTrees.ApproxDimension];
                for (size_t docId = 0; docId < docCountInBlock; ++docId) {
                    auto leafValuePtr = treeLeafPtr + indexesVec[docId] * model.ObliviousTrees.ApproxDimension;
                    for (int classId = 0; classId < model.ObliviousTrees.ApproxDimension; ++classId) {
                        docResultPtr[classId] += leafValuePtr[classId];
                    }
                    docResultPtr += model.ObliviousTrees.ApproxDimension;
                }
            }
            treeSplitsCurPtr += curTreeSize;
        }
    } else {
        double result = 0.0;
        for (size_t treeId = treeStart; treeId < treeEnd; ++treeId) {
            auto curTreeSize = model.ObliviousTrees.TreeSizes[treeId];
            TCalcerIndexType index = 0;
            for (int depth = 0; depth < curTreeSize; ++depth) {
                const ui8 borderVal = (ui8)(treeSplitsCurPtr[depth] & 0xff);
                const ui32 featureIndex = (treeSplitsCurPtr[depth] >> 16);
                if (NeedXorMask) {
                    const ui8 xorMask = (ui8)((treeSplitsCurPtr[depth] & 0xff00) >> 8);
                    index |= ((binFeatures[featureIndex] ^ xorMask) >= borderVal) << depth;
                } else {
                    index |= (binFeatures[featureIndex] >= borderVal) << depth;
                }
            }
            auto treeLeafPtr = model.ObliviousTrees.LeafValues[treeId].data();
            if (IsSingleClassModel) { // single class model
                result += treeLeafPtr[index];
            } else { // mutliclass model
                auto docResultPtr = &results[model.ObliviousTrees.ApproxDimension];
                auto leafValuePtr = treeLeafPtr + index * model.ObliviousTrees.ApproxDimension;
                for (int classId = 0; classId < model.ObliviousTrees.ApproxDimension; ++classId) {
                    docResultPtr[classId] += leafValuePtr[classId];
                }
            }
            treeSplitsCurPtr += curTreeSize;
        }
        if (IsSingleClassModel) {
            results[0] = result;
        }
    }
}

TTreeCalcFunction GetCalcTreesFunction(int approxDimension, size_t docCountInBlock, bool hasOneHots) {
    if (approxDimension == 1) {
        if (docCountInBlock == 1) {
            if (hasOneHots) {
                return CalcTreesImpl<true, true, true>;
            } else {
                return CalcTreesImpl<true, true, false>;
            }
        } else {
            if (hasOneHots) {
                return CalcTreesImpl<true, false, true>;
            } else {
                return CalcTreesImpl<true, false, false>;
            }
        }
    } else {
        if (docCountInBlock == 1) {
            if (hasOneHots) {
                return CalcTreesImpl<false, true, true>;
            } else {
                return CalcTreesImpl<false, true, false>;
            }
        } else {
            if (hasOneHots) {
                return CalcTreesImpl<false, false, true>;
            } else {
                return CalcTreesImpl<false, false, false>;
            }
        }
    }
}
