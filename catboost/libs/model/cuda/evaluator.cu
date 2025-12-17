#include "evaluator.cuh"

#include <catboost/libs/helpers/exception.h>

#include <library/cpp/cuda/wrappers/kernel.cuh>
#include <library/cpp/cuda/wrappers/kernel_helpers.cuh>
#include <library/cpp/cuda/wrappers/arch.h>
#include <library/cpp/cuda/wrappers/kernel_helpers.cuh>

#include <util/string/cast.h>

#include <catboost/private/libs/ctr_description/ctr_type.h>

#include <cuda_runtime.h>
#include <assert.h>

template<typename TFeatureType, TGPUDataInput::EFeatureLayout Layout>
struct TFeatureAccessor {
    TFeatureAccessor() = default;
    using TFeature = TFeatureType;
    using TFeaturePtr = const TFeature*;

    i32 Stride = 0;

    i32 FeatureCount = 0;
    i32 ObjectCount = 0;
    TFeaturePtr FeaturesPtr = nullptr;

    __forceinline__ __device__ TFeature operator()(i32 featureId, i32 objectId) const {
        if (Layout == TGPUDataInput::EFeatureLayout::ColumnFirst) {
            return objectId < ObjectCount && featureId < FeatureCount ?
                __ldg(FeaturesPtr + featureId * Stride + objectId)
                : NegativeInfty();
        } else {
            return objectId < ObjectCount && featureId < FeatureCount ?
                __ldg(FeaturesPtr + featureId + objectId * Stride)
                : NegativeInfty();
        }
    }

    __forceinline__ __device__ int FeaturesCount() const {
        return FeatureCount;
    }

    __forceinline__ __device__ int SamplesCount() const {
        return ObjectCount;
    }
};

template <TGPUDataInput::EFeatureLayout Layout>
struct THashedCatFeatureAccessor {
    THashedCatFeatureAccessor() = default;

    using TFeature = ui32;
    using TFeaturePtr = const TFeature*;

    i32 Stride = 0;

    i32 FeatureCount = 0;
    i32 ObjectCount = 0;
    TFeaturePtr FeaturesPtr = nullptr;

    __forceinline__ __device__ TFeature operator()(i32 featureId, i32 objectId) const {
        if (Layout == TGPUDataInput::EFeatureLayout::ColumnFirst) {
            return objectId < ObjectCount && featureId < FeatureCount ?
                __ldg(FeaturesPtr + featureId * Stride + objectId)
                : 0u;
        } else {
            return objectId < ObjectCount && featureId < FeatureCount ?
                __ldg(FeaturesPtr + featureId + objectId * Stride)
                : 0u;
        }
    }

    __forceinline__ __device__ int FeaturesCount() const {
        return FeatureCount;
    }

    __forceinline__ __device__ int SamplesCount() const {
        return ObjectCount;
    }
};

constexpr ui32 ObjectsPerThread = 4;
constexpr ui32 TreeSubBlockWidth = 8;
constexpr ui32 ExtTreeBlockWidth = 128;
constexpr ui32 QuantizationDocBlockSize = 256;

constexpr ui32 BlockWidth = 256;
constexpr ui32 EvalDocBlockSize = BlockWidth / TreeSubBlockWidth;

static_assert(EvalDocBlockSize >= WarpSize, "EvalBlockSize should be greater than WarpSize");
using TTreeIndex = uint4;

void TCudaQuantizedData::SetDimensions(ui32 effectiveBucketCount, ui32 objectsCount) {
    ObjectsCount = objectsCount;
    EffectiveBucketCount = effectiveBucketCount;
    const auto one32blockSize = WarpSize * effectiveBucketCount;
    const auto desiredQuantBuff = one32blockSize * NKernel::CeilDivide<ui32>(objectsCount, 128) * 4;
    if (BinarizedFeaturesBuffer.Size() < desiredQuantBuff) {
        BinarizedFeaturesBuffer = TCudaVec<TCudaQuantizationBucket>(desiredQuantBuff, NCuda::EMemoryType::Device);
    }
}

void TEvaluationDataCache::PrepareCopyBufs(size_t bufSize, size_t resultsSize) {
    if (CopyDataBufDevice.Size() < bufSize) {
        CopyDataBufDevice = TCudaVec<float>(AlignBy<2048>(bufSize), NCuda::EMemoryType::Device);
    }
    if (CopyDataBufHost.Size() < bufSize) {
        CopyDataBufHost = TCudaVec<float>(AlignBy<2048>(bufSize), NCuda::EMemoryType::Host);
    }
    if (ResultsFloatBuf.Size() < resultsSize) {
        ResultsFloatBuf = TCudaVec<float>(AlignBy<2048>(resultsSize), NCuda::EMemoryType::Device);
    }
    if (ResultsDoubleBuf.Size() < resultsSize) {
        ResultsDoubleBuf = TCudaVec<double>(AlignBy<2048>(resultsSize), NCuda::EMemoryType::Device);
    }
}

void TEvaluationDataCache::PrepareCtrBuf(size_t objectsCount) {
    if (CtrTempBuf.Size() < objectsCount) {
        CtrTempBuf = TCudaVec<float>(AlignBy<2048>(objectsCount), NCuda::EMemoryType::Device);
    }
}

void TEvaluationDataCache::PrepareLeafIndexesBuf(size_t indexesSize) {
    if (LeafIndexesBuf.Size() < indexesSize) {
        LeafIndexesBuf = TCudaVec<ui32>(AlignBy<2048>(indexesSize), NCuda::EMemoryType::Device);
    }
}

template<typename TFloatFeatureAccessor>
__launch_bounds__(QuantizationDocBlockSize, 1)
__global__ void Binarize(
    TFloatFeatureAccessor floatAccessor,
    const float* __restrict__ borders,
    const ui32* __restrict__ featureBorderOffsets,
    const ui32* __restrict__ featureBordersCount,
    const ui32* __restrict__ floatFeatureForBucketIdx,
    const ui32 bucketsCount,
    TCudaQuantizationBucket* __restrict__ target
) {
    const int blockby32 = blockIdx.x * QuantizationDocBlockSize / WarpSize + threadIdx.x / WarpSize;
    const int firstDocForThread = blockby32 * WarpSize * ObjectsPerThread + threadIdx.x % WarpSize;

    const int targetBucketIdx = blockIdx.y;
    const float* featureBorders = borders + featureBorderOffsets[targetBucketIdx];
    const int featureBorderCount = __ldg(featureBordersCount + targetBucketIdx);
    const int featureIdx = floatFeatureForBucketIdx[targetBucketIdx];
    __shared__ float bordersLocal[QuantizationDocBlockSize];
    if (threadIdx.x < featureBorderCount) {
        bordersLocal[threadIdx.x] = __ldg(featureBorders + threadIdx.x);
    }
    __syncthreads();

    float4 features;
    features.x = floatAccessor(featureIdx, firstDocForThread + 0 * WarpSize);
    features.y = floatAccessor(featureIdx, firstDocForThread + 1 * WarpSize);
    features.z = floatAccessor(featureIdx, firstDocForThread + 2 * WarpSize);
    features.w = floatAccessor(featureIdx, firstDocForThread + 3 * WarpSize);

    TCudaQuantizationBucket bins = { 0 };
#pragma unroll 8
    for (int borderId = 0; borderId < featureBorderCount; ++borderId) {
        const float border = bordersLocal[borderId];

        bins.x += features.x > border;
        bins.y += features.y > border;
        bins.z += features.z > border;
        bins.w += features.w > border;
    }
    if (firstDocForThread < floatAccessor.SamplesCount()) {
        target[bucketsCount * WarpSize * blockby32 + targetBucketIdx * WarpSize + threadIdx.x % WarpSize] = bins;
    }
}

template <typename THashedCatAccessor>
__launch_bounds__(QuantizationDocBlockSize, 1)
__global__ void BinarizeOneHot(
    THashedCatAccessor catAccessor,
    const ui32* __restrict__ flatValues,
    const ui32* __restrict__ valueOffsets,
    const ui32* __restrict__ valueCounts,
    const ui32* __restrict__ catFeaturePackedIdxForBucket,
    const ui32 oneHotBucketOffset,
    const ui32 bucketsCount,
    TCudaQuantizationBucket* __restrict__ target
) {
    const int blockby32 = blockIdx.x * QuantizationDocBlockSize / WarpSize + threadIdx.x / WarpSize;
    const int firstDocForThread = blockby32 * WarpSize * ObjectsPerThread + threadIdx.x % WarpSize;

    const ui32 bucketIdx = blockIdx.y;
    const ui32 catFeatureIdx = __ldg(catFeaturePackedIdxForBucket + bucketIdx);
    const ui32 valsOffset = __ldg(valueOffsets + bucketIdx);
    const int valsCount = static_cast<int>(__ldg(valueCounts + bucketIdx));

    __shared__ ui32 valuesLocal[QuantizationDocBlockSize];
    if (threadIdx.x < valsCount) {
        valuesLocal[threadIdx.x] = __ldg(flatValues + valsOffset + threadIdx.x);
    }
    __syncthreads();

    const ui32 v0 = catAccessor(catFeatureIdx, firstDocForThread + 0 * WarpSize);
    const ui32 v1 = catAccessor(catFeatureIdx, firstDocForThread + 1 * WarpSize);
    const ui32 v2 = catAccessor(catFeatureIdx, firstDocForThread + 2 * WarpSize);
    const ui32 v3 = catAccessor(catFeatureIdx, firstDocForThread + 3 * WarpSize);

    ui8 b0 = 0;
    ui8 b1 = 0;
    ui8 b2 = 0;
    ui8 b3 = 0;
    for (int i = 0; i < valsCount; ++i) {
        const ui32 hv = valuesLocal[i];
        const ui8 code = static_cast<ui8>(i + 1);
        if (b0 == 0 && v0 == hv) {
            b0 = code;
        }
        if (b1 == 0 && v1 == hv) {
            b1 = code;
        }
        if (b2 == 0 && v2 == hv) {
            b2 = code;
        }
        if (b3 == 0 && v3 == hv) {
            b3 = code;
        }
        if (b0 && b1 && b2 && b3) {
            break;
        }
    }

    TCudaQuantizationBucket bins;
    bins.x = b0;
    bins.y = b1;
    bins.z = b2;
    bins.w = b3;

    if (firstDocForThread < catAccessor.SamplesCount()) {
        const ui32 globalBucketIdx = oneHotBucketOffset + bucketIdx;
        target[bucketsCount * WarpSize * blockby32 + globalBucketIdx * WarpSize + threadIdx.x % WarpSize] = bins;
    }
}

__device__ __forceinline__ ui64 CalcHashDevice(ui64 a, ui64 b) {
    constexpr ui64 MagicMult = 0x4906ba494954cb65ull;
    return MagicMult * (a + MagicMult * b);
}

__device__ __forceinline__ ui32 GetDenseHashIndex(
    const TGPUCtrBucket* __restrict buckets,
    ui32 bucketCount,
    ui64 hash
) {
    const ui64 mask = static_cast<ui64>(bucketCount - 1);
    ui64 z = hash & mask;
    while (true) {
        const ui64 curHash = __ldg(&buckets[z].Hash);
        if (curHash == TGPUCtrBucket::InvalidHashValue) {
            return TGPUCtrBucket::NotFoundIndex;
        }
        if (curHash == hash) {
            return __ldg(&buckets[z].IndexValue);
        }
        z = (z + 1) & mask;
    }
}

__device__ __forceinline__ float CalcCtrValue(
    float countInClass,
    float totalCount,
    float priorNum,
    float priorDenom,
    float shift,
    float scale
) {
    const float ctr = (countInClass + priorNum) / (totalCount + priorDenom);
    return (ctr + shift) * scale;
}

__global__ void ComputeCtrValuesImpl(
    const float* __restrict floatFeatures,
    ui32 floatFeatureCount,
    const ui32* __restrict hashedCatFeatures,
    ui32 catFeatureCount,
    ui32 stride,
    ui32 docCount,
    ui32 ctrFeatureId,
    // Feature descriptors
    const ui32* __restrict ctrFeatureTableIdx,
    const ui8* __restrict ctrFeatureType,
    const ui32* __restrict ctrFeatureTargetBorderIdx,
    const float* __restrict ctrPriorNum,
    const float* __restrict ctrPriorDenom,
    const float* __restrict ctrShift,
    const float* __restrict ctrScale,
    const ui32* __restrict ctrCatOffsets,
    const ui32* __restrict ctrCatCount,
    const ui32* __restrict ctrCatsFlat,
    const ui32* __restrict ctrFloatOffsets,
    const ui32* __restrict ctrFloatCount,
    const TGPUCtrFloatSplit* __restrict ctrFloatSplits,
    const ui32* __restrict ctrOneHotOffsets,
    const ui32* __restrict ctrOneHotCount,
    const TGPUCtrOneHotSplit* __restrict ctrOneHotSplits,
    // Tables
    const TGPUCtrBucket* __restrict tableBuckets,
    const ui32* __restrict tableBucketsOffsets,
    const ui32* __restrict tableBucketsCount,
    const ui8* __restrict tableDataKind,
    const ui32* __restrict tableMeanOffsets,
    const TGPUCtrMeanHistory* __restrict meanData,
    const ui32* __restrict tableIntOffsets,
    const int* __restrict intData,
    const int* __restrict tableCounterDenom,
    const int* __restrict tableTargetClassesCount,
    float* __restrict outValues
) {
    const ui32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= docCount) {
        return;
    }

    ui64 hash = 0;

    const ui32 catOff = __ldg(ctrCatOffsets + ctrFeatureId);
    const ui32 catCnt = __ldg(ctrCatCount + ctrFeatureId);
    for (ui32 j = 0; j < catCnt; ++j) {
        const ui32 packedCatIdx = __ldg(ctrCatsFlat + catOff + j);
        if (packedCatIdx >= catFeatureCount) {
            continue;
        }
        const ui32 hv = __ldg(hashedCatFeatures + packedCatIdx * stride + i);
        hash = CalcHashDevice(hash, static_cast<ui64>(static_cast<i32>(hv)));
    }

    const ui32 floatOff = __ldg(ctrFloatOffsets + ctrFeatureId);
    const ui32 floatCnt = __ldg(ctrFloatCount + ctrFeatureId);
    for (ui32 j = 0; j < floatCnt; ++j) {
        const TGPUCtrFloatSplit fs = Ldg(ctrFloatSplits + floatOff + j);
        const ui32 fIdx = fs.FloatFeatureIdx;
        float v = 0.0f;
        if (fIdx < floatFeatureCount) {
            v = __ldg(floatFeatures + fIdx * stride + i);
        }
        const bool isTrue = (v > fs.Border);
        hash = CalcHashDevice(hash, static_cast<ui64>(isTrue));
    }

    const ui32 ohOff = __ldg(ctrOneHotOffsets + ctrFeatureId);
    const ui32 ohCnt = __ldg(ctrOneHotCount + ctrFeatureId);
    for (ui32 j = 0; j < ohCnt; ++j) {
        const TGPUCtrOneHotSplit oh = Ldg(ctrOneHotSplits + ohOff + j);
        const ui32 packedCatIdx = oh.CatFeaturePackedIdx;
        ui32 hv = 0;
        if (packedCatIdx < catFeatureCount) {
            hv = __ldg(hashedCatFeatures + packedCatIdx * stride + i);
        }
        const bool isTrue = (hv == oh.Value);
        hash = CalcHashDevice(hash, static_cast<ui64>(isTrue));
    }

    const ui32 tableIdx = __ldg(ctrFeatureTableIdx + ctrFeatureId);
    const ui32 bucketsOff = __ldg(tableBucketsOffsets + tableIdx);
    const ui32 bucketsCnt = __ldg(tableBucketsCount + tableIdx);
    const TGPUCtrBucket* __restrict buckets = tableBuckets + bucketsOff;
    const ui32 idx = GetDenseHashIndex(buckets, bucketsCnt, hash);

    const ui8 ctrTypeRaw = __ldg(ctrFeatureType + ctrFeatureId);
    const ECtrType ctrType = static_cast<ECtrType>(ctrTypeRaw);
    const ui32 targetBorderIdx = __ldg(ctrFeatureTargetBorderIdx + ctrFeatureId);

    const float pNum = __ldg(ctrPriorNum + ctrFeatureId);
    const float pDen = __ldg(ctrPriorDenom + ctrFeatureId);
    const float sh = __ldg(ctrShift + ctrFeatureId);
    const float sc = __ldg(ctrScale + ctrFeatureId);

    float value = 0.0f;
    if (ctrType == ECtrType::BinarizedTargetMeanValue || ctrType == ECtrType::FloatTargetMeanValue) {
        const float emptyVal = CalcCtrValue(0.0f, 0.0f, pNum, pDen, sh, sc);
        if (idx != TGPUCtrBucket::NotFoundIndex) {
            const ui32 meanOff = __ldg(tableMeanOffsets + tableIdx);
            const TGPUCtrMeanHistory mh = Ldg(meanData + meanOff + idx);
            value = CalcCtrValue(mh.Sum, static_cast<float>(mh.Count), pNum, pDen, sh, sc);
        } else {
            value = emptyVal;
        }
    } else if (ctrType == ECtrType::Counter || ctrType == ECtrType::FeatureFreq) {
        const int denom = __ldg(tableCounterDenom + tableIdx);
        const float emptyVal = CalcCtrValue(0.0f, static_cast<float>(denom), pNum, pDen, sh, sc);
        if (idx != TGPUCtrBucket::NotFoundIndex) {
            const ui32 intOff = __ldg(tableIntOffsets + tableIdx);
            const int total = __ldg(intData + intOff + idx);
            value = CalcCtrValue(static_cast<float>(total), static_cast<float>(denom), pNum, pDen, sh, sc);
        } else {
            value = emptyVal;
        }
    } else if (ctrType == ECtrType::Buckets) {
        const float emptyVal = CalcCtrValue(0.0f, 0.0f, pNum, pDen, sh, sc);
        if (idx != TGPUCtrBucket::NotFoundIndex) {
            const int classCount = __ldg(tableTargetClassesCount + tableIdx);
            const ui32 intOff = __ldg(tableIntOffsets + tableIdx);
            const int* __restrict hist = intData + intOff + static_cast<ui64>(idx) * classCount;
            int good = 0;
            int total = 0;
            if (static_cast<int>(targetBorderIdx) < classCount) {
                good = __ldg(hist + targetBorderIdx);
            }
            for (int c = 0; c < classCount; ++c) {
                total += __ldg(hist + c);
            }
            value = CalcCtrValue(static_cast<float>(good), static_cast<float>(total), pNum, pDen, sh, sc);
        } else {
            value = emptyVal;
        }
    } else if (ctrType == ECtrType::Borders) {
        const float emptyVal = CalcCtrValue(0.0f, 0.0f, pNum, pDen, sh, sc);
        if (idx != TGPUCtrBucket::NotFoundIndex) {
            const int classCount = __ldg(tableTargetClassesCount + tableIdx);
            const ui32 intOff = __ldg(tableIntOffsets + tableIdx);
            const int* __restrict hist = intData + intOff + static_cast<ui64>(idx) * classCount;
            int good = 0;
            int total = 0;
            if (classCount > 2) {
                for (int c = 0; c < classCount; ++c) {
                    const int v = __ldg(hist + c);
                    if (c <= static_cast<int>(targetBorderIdx)) {
                        total += v;
                    } else {
                        good += v;
                    }
                }
                total += good;
            } else if (classCount == 2) {
                const int c0 = __ldg(hist + 0);
                const int c1 = __ldg(hist + 1);
                good = c1;
                total = c0 + c1;
            }
            value = CalcCtrValue(static_cast<float>(good), static_cast<float>(total), pNum, pDen, sh, sc);
        } else {
            value = emptyVal;
        }
    } else {
        value = CalcCtrValue(0.0f, 0.0f, pNum, pDen, sh, sc);
    }

    outValues[i] = value;
}

__launch_bounds__(QuantizationDocBlockSize, 1)
__global__ void BinarizeCtrValues(
    const float* __restrict ctrValues,
    ui32 docCount,
    const float* __restrict borders,
    const ui32* __restrict bucketBorderOffsets,
    const ui32* __restrict bucketBordersCount,
    const ui32 bucketBaseInModel,
    const ui32 bucketsCount,
    TCudaQuantizationBucket* __restrict target
) {
    const int blockby32 = blockIdx.x * QuantizationDocBlockSize / WarpSize + threadIdx.x / WarpSize;
    const int firstDocForThread = blockby32 * WarpSize * ObjectsPerThread + threadIdx.x % WarpSize;

    const ui32 bucketIdx = blockIdx.y;
    const float* bucketBorders = borders + __ldg(bucketBorderOffsets + bucketIdx);
    const int bucketBorderCount = static_cast<int>(__ldg(bucketBordersCount + bucketIdx));

    __shared__ float bordersLocal[QuantizationDocBlockSize];
    if (threadIdx.x < bucketBorderCount) {
        bordersLocal[threadIdx.x] = __ldg(bucketBorders + threadIdx.x);
    }
    __syncthreads();

    float4 features;
    features.x = (firstDocForThread + 0 * WarpSize) < docCount ? __ldg(ctrValues + firstDocForThread + 0 * WarpSize) : NegativeInfty();
    features.y = (firstDocForThread + 1 * WarpSize) < docCount ? __ldg(ctrValues + firstDocForThread + 1 * WarpSize) : NegativeInfty();
    features.z = (firstDocForThread + 2 * WarpSize) < docCount ? __ldg(ctrValues + firstDocForThread + 2 * WarpSize) : NegativeInfty();
    features.w = (firstDocForThread + 3 * WarpSize) < docCount ? __ldg(ctrValues + firstDocForThread + 3 * WarpSize) : NegativeInfty();

    TCudaQuantizationBucket bins = { 0 };
#pragma unroll 8
    for (int borderId = 0; borderId < bucketBorderCount; ++borderId) {
        const float border = bordersLocal[borderId];
        bins.x += features.x > border;
        bins.y += features.y > border;
        bins.z += features.z > border;
        bins.w += features.w > border;
    }
    if (firstDocForThread < docCount) {
        const ui32 globalBucketIdx = bucketBaseInModel + bucketIdx;
        target[bucketsCount * WarpSize * blockby32 + globalBucketIdx * WarpSize + threadIdx.x % WarpSize] = bins;
    }
}

template<int TreeDepth>
TTreeIndex __device__ __forceinline__ CalcIndexesUnwrapped(const TGPURepackedBin* const __restrict__ curRepackedBinPtr, const TCudaQuantizationBucket* const __restrict__ quantizedFeatures) {
    TTreeIndex result = { 0 };
#pragma unroll TreeDepth
    for (int depth = 0; depth < TreeDepth; ++depth) {
        const TGPURepackedBin bin = Ldg(curRepackedBinPtr + depth);
        TCudaQuantizationBucket buckets = __ldg(quantizedFeatures + bin.FeatureIdx);
        // |= operator fails (MLTOOLS-6839 on a100)
        result.x += ((static_cast<ui8>(buckets.x) ^ bin.FeatureXorMask) >= bin.FeatureVal) << depth;
        result.y += ((static_cast<ui8>(buckets.y) ^ bin.FeatureXorMask) >= bin.FeatureVal) << depth;
        result.z += ((static_cast<ui8>(buckets.z) ^ bin.FeatureXorMask) >= bin.FeatureVal) << depth;
        result.w += ((static_cast<ui8>(buckets.w) ^ bin.FeatureXorMask) >= bin.FeatureVal) << depth;
    }
    return result;
}

TTreeIndex __device__ CalcIndexesBase(int TreeDepth, const TGPURepackedBin* const __restrict__ curRepackedBinPtr, const TCudaQuantizationBucket* const __restrict__ quantizedFeatures) {
    TTreeIndex bins = { 0 };
    for (int depth = 0; depth < TreeDepth; ++depth) {
        const TGPURepackedBin bin = Ldg(curRepackedBinPtr + depth);
        TCudaQuantizationBucket vals = __ldg(quantizedFeatures + bin.FeatureIdx);
        // |= operator fails (MLTOOLS-6839 on a100)
        bins.x += ((static_cast<ui8>(vals.x) ^ bin.FeatureXorMask) >= bin.FeatureVal) << depth;
        bins.y += ((static_cast<ui8>(vals.y) ^ bin.FeatureXorMask) >= bin.FeatureVal) << depth;
        bins.z += ((static_cast<ui8>(vals.z) ^ bin.FeatureXorMask) >= bin.FeatureVal) << depth;
        bins.w += ((static_cast<ui8>(vals.w) ^ bin.FeatureXorMask) >= bin.FeatureVal) << depth;
    }
    return bins;
}

TTreeIndex __device__ __forceinline__ CalcTreeVals(int curTreeDepth, const TGPURepackedBin* const __restrict__ curRepackedBinPtr, const TCudaQuantizationBucket* const __restrict__ quantizedFeatures) {
    switch (curTreeDepth) {
    case 6:
        return CalcIndexesUnwrapped<6>(curRepackedBinPtr, quantizedFeatures);
    case 7:
        return CalcIndexesUnwrapped<7>(curRepackedBinPtr, quantizedFeatures);
    case 8:
        return CalcIndexesUnwrapped<8>(curRepackedBinPtr, quantizedFeatures);
    default:
        return CalcIndexesBase(curTreeDepth, curRepackedBinPtr, quantizedFeatures);
    }
}

__launch_bounds__(BlockWidth, 1)
__global__ void EvalObliviousTrees(
    const TCudaQuantizationBucket* __restrict__ quantizedFeatures,
    const ui32* __restrict__ treeSizes,
    const ui32 treeCount,
    const ui32* __restrict__ treeStartOffsets,
    const TGPURepackedBin* __restrict__ repackedBins,
    const ui32* __restrict__ firstLeafOfset,
    const ui32 bucketsCount,
    const TCudaEvaluatorLeafType* __restrict__ leafValues,
    const ui32 documentCount,
    const ui32 approxDimension,
    TCudaEvaluatorLeafType* __restrict__ results) {

    const int innerBlockBy32 = threadIdx.x / WarpSize;
    const int blockby32 = blockIdx.y * EvalDocBlockSize / WarpSize + innerBlockBy32;
    const int inBlockId = threadIdx.x % WarpSize;
    const int firstDocForThread = blockby32 * WarpSize * ObjectsPerThread + inBlockId;
    const ui32 dimId = static_cast<ui32>(blockIdx.z);

    quantizedFeatures += bucketsCount * WarpSize * blockby32 + threadIdx.x % WarpSize;

    const int firstTreeIdx = TreeSubBlockWidth * ExtTreeBlockWidth * (threadIdx.y + TreeSubBlockWidth * blockIdx.x);
    const int lastTreeIdx = min(firstTreeIdx + TreeSubBlockWidth * ExtTreeBlockWidth, treeCount);
    double4 localResult = { 0 };

    if (firstTreeIdx < lastTreeIdx && firstDocForThread < documentCount) {
        const TGPURepackedBin* __restrict__ curRepackedBinPtr = repackedBins + __ldg(treeStartOffsets + firstTreeIdx);
        const TCudaEvaluatorLeafType* __restrict__ curLeafValuesDim = leafValues + firstLeafOfset[firstTreeIdx] + dimId;
        int treeIdx = firstTreeIdx;
        const int lastTreeBy2 = lastTreeIdx - ((lastTreeIdx - firstTreeIdx) & 0x3);
        for (; treeIdx < lastTreeBy2; treeIdx += 2) {
            const int curTreeDepth1 = __ldg(treeSizes + treeIdx);
            const int curTreeDepth2 = __ldg(treeSizes + treeIdx + 1);

            const TTreeIndex bins1 = CalcTreeVals(curTreeDepth1, curRepackedBinPtr, quantizedFeatures);
            const TTreeIndex bins2 = CalcTreeVals(curTreeDepth2, curRepackedBinPtr + curTreeDepth1, quantizedFeatures);
            const auto leafValues2Dim = curLeafValuesDim + (static_cast<ui64>(1) << curTreeDepth1) * approxDimension;
            localResult.x += __ldg(curLeafValuesDim + static_cast<ui64>(bins1.x) * approxDimension) + __ldg(leafValues2Dim + static_cast<ui64>(bins2.x) * approxDimension);
            localResult.y += __ldg(curLeafValuesDim + static_cast<ui64>(bins1.y) * approxDimension) + __ldg(leafValues2Dim + static_cast<ui64>(bins2.y) * approxDimension);
            localResult.z += __ldg(curLeafValuesDim + static_cast<ui64>(bins1.z) * approxDimension) + __ldg(leafValues2Dim + static_cast<ui64>(bins2.z) * approxDimension);
            localResult.w += __ldg(curLeafValuesDim + static_cast<ui64>(bins1.w) * approxDimension) + __ldg(leafValues2Dim + static_cast<ui64>(bins2.w) * approxDimension);
            curRepackedBinPtr += curTreeDepth1 + curTreeDepth2;
            curLeafValuesDim = leafValues2Dim + (static_cast<ui64>(1) << curTreeDepth2) * approxDimension;
        }
        for (; treeIdx < lastTreeIdx; ++treeIdx) {
            const int curTreeDepth = __ldg(treeSizes + treeIdx);
            const TTreeIndex bins = CalcTreeVals(curTreeDepth, curRepackedBinPtr, quantizedFeatures);
            localResult.x += __ldg(curLeafValuesDim + static_cast<ui64>(bins.x) * approxDimension);
            localResult.y += __ldg(curLeafValuesDim + static_cast<ui64>(bins.y) * approxDimension);
            localResult.z += __ldg(curLeafValuesDim + static_cast<ui64>(bins.z) * approxDimension);
            localResult.w += __ldg(curLeafValuesDim + static_cast<ui64>(bins.w) * approxDimension);
            curRepackedBinPtr += curTreeDepth;
            curLeafValuesDim += (static_cast<ui64>(1) << curTreeDepth) * approxDimension;
        }
    }
    // TODO(kirillovs): reduce code is valid if those conditions met
    static_assert(EvalDocBlockSize * ObjectsPerThread == 128, "");
    static_assert(EvalDocBlockSize == 32, "");
    __shared__ TCudaEvaluatorLeafType reduceVals[EvalDocBlockSize * ObjectsPerThread * TreeSubBlockWidth];
    reduceVals[innerBlockBy32 * WarpSize * ObjectsPerThread + WarpSize * 0 + inBlockId + threadIdx.y * EvalDocBlockSize * ObjectsPerThread] = localResult.x;
    reduceVals[innerBlockBy32 * WarpSize * ObjectsPerThread + WarpSize * 1 + inBlockId + threadIdx.y * EvalDocBlockSize * ObjectsPerThread] = localResult.y;
    reduceVals[innerBlockBy32 * WarpSize * ObjectsPerThread + WarpSize * 2 + inBlockId + threadIdx.y * EvalDocBlockSize * ObjectsPerThread] = localResult.z;
    reduceVals[innerBlockBy32 * WarpSize * ObjectsPerThread + WarpSize * 3 + inBlockId + threadIdx.y * EvalDocBlockSize * ObjectsPerThread] = localResult.w;
    __syncthreads();
    TCudaEvaluatorLeafType lr = reduceVals[threadIdx.x + threadIdx.y * EvalDocBlockSize];
    for (int i = 256; i < 256 * 4; i += 256) {
        lr += reduceVals[i + threadIdx.x + threadIdx.y * EvalDocBlockSize];
    }
    reduceVals[threadIdx.x + threadIdx.y * EvalDocBlockSize] = lr;
    __syncthreads();
    if (threadIdx.y < ObjectsPerThread) {
        const ui32 docId = blockby32 * WarpSize * ObjectsPerThread + threadIdx.x + threadIdx.y * EvalDocBlockSize;
        const ui32 dstIdx = docId * approxDimension + dimId;
        TAtomicAdd<TCudaEvaluatorLeafType>::Add(
            results + dstIdx,
            reduceVals[threadIdx.x + threadIdx.y * EvalDocBlockSize] + reduceVals[threadIdx.x + threadIdx.y * EvalDocBlockSize + 128]
        );
    }
}

__launch_bounds__(BlockWidth, 1)
__global__ void CalcObliviousLeafIndexes(
    const TCudaQuantizationBucket* __restrict__ quantizedFeatures,
    const ui32* __restrict__ treeSizes,
    const ui32 treeCount,
    const ui32* __restrict__ treeStartOffsets,
    const TGPURepackedBin* __restrict__ repackedBins,
    const ui32 bucketsCount,
    const ui32 documentCount,
    ui32* __restrict__ leafIndexes
) {
    const int innerBlockBy32 = threadIdx.x / WarpSize;
    const int blockby32 = blockIdx.y * EvalDocBlockSize / WarpSize + innerBlockBy32;
    const int inBlockId = threadIdx.x % WarpSize;
    const int firstDocForThread = blockby32 * WarpSize * ObjectsPerThread + inBlockId;

    quantizedFeatures += bucketsCount * WarpSize * blockby32 + threadIdx.x % WarpSize;

    const int firstTreeIdx = TreeSubBlockWidth * ExtTreeBlockWidth * (threadIdx.y + TreeSubBlockWidth * blockIdx.x);
    const int lastTreeIdx = min(firstTreeIdx + TreeSubBlockWidth * ExtTreeBlockWidth, static_cast<int>(treeCount));

    if (firstTreeIdx < lastTreeIdx && firstDocForThread < static_cast<int>(documentCount)) {
        const TGPURepackedBin* __restrict__ curRepackedBinPtr = repackedBins + __ldg(treeStartOffsets + firstTreeIdx);
        for (int treeIdx = firstTreeIdx; treeIdx < lastTreeIdx; ++treeIdx) {
            const int curTreeDepth = __ldg(treeSizes + treeIdx);
            const TTreeIndex bins = CalcTreeVals(curTreeDepth, curRepackedBinPtr, quantizedFeatures);

            const ui32 outTreeIdx = static_cast<ui32>(treeIdx);
            const ui32 doc0 = static_cast<ui32>(firstDocForThread);
            const ui32 doc1 = doc0 + 1u * WarpSize;
            const ui32 doc2 = doc0 + 2u * WarpSize;
            const ui32 doc3 = doc0 + 3u * WarpSize;

            if (doc0 < documentCount) {
                leafIndexes[static_cast<ui64>(doc0) * treeCount + outTreeIdx] = bins.x;
            }
            if (doc1 < documentCount) {
                leafIndexes[static_cast<ui64>(doc1) * treeCount + outTreeIdx] = bins.y;
            }
            if (doc2 < documentCount) {
                leafIndexes[static_cast<ui64>(doc2) * treeCount + outTreeIdx] = bins.z;
            }
            if (doc3 < documentCount) {
                leafIndexes[static_cast<ui64>(doc3) * treeCount + outTreeIdx] = bins.w;
            }

            curRepackedBinPtr += curTreeDepth;
        }
    }
}

template<NCB::NModelEvaluation::EPredictionType PredictionType, bool OneDimension>
__global__ void ProcessResultsImpl(
    const float* __restrict__ rawResults,
    ui32 resultsSize,
    const double* __restrict__ bias,
    ui32 addBias,
    double scale,
    double* hostMemResults,
    ui32 approxDimension
) {
    for (ui32 resultId = threadIdx.x; resultId < resultsSize; resultId += blockDim.x) {
        if (OneDimension) {
            double res = scale * __ldg(rawResults + resultId) + (addBias ? __ldg(bias) : 0.0);
            if (PredictionType == NCB::NModelEvaluation::EPredictionType::RawFormulaVal) {
                hostMemResults[resultId] = res;
            } else if (PredictionType == NCB::NModelEvaluation::EPredictionType::Probability) {
                hostMemResults[resultId] = 1 / (1 + exp(-res));
            } else if (PredictionType == NCB::NModelEvaluation::EPredictionType::Class) {
                hostMemResults[resultId] = res > 0;
            } else {
                assert(0);
            }
        } else {
            const float* rawResultsSub = rawResults + resultId * approxDimension;
            if (PredictionType == NCB::NModelEvaluation::EPredictionType::Class) {
                double maxVal = scale * __ldg(rawResultsSub) + (addBias ? __ldg(bias) : 0.0);
                ui32 maxPos = 0;
                for (ui32 dim = 1; dim < approxDimension; ++dim) {
                    double val = scale * __ldg(rawResultsSub + dim) + (addBias ? __ldg(bias + dim) : 0.0);
                    if (val > maxVal) {
                        maxVal = val;
                        maxPos = dim;
                    }
                }
                hostMemResults[resultId] = maxPos;
            } else {
                double* hostMemResultsBase = hostMemResults + resultId * approxDimension;

                for (ui32 dim = 0; dim < approxDimension; ++dim) {
                    hostMemResultsBase[dim] = scale * __ldg(rawResultsSub + dim) + (addBias ? __ldg(bias + dim) : 0.0);
                }

                if (PredictionType != NCB::NModelEvaluation::EPredictionType::RawFormulaVal) {
                    // TODO(kirillovs): write softmax
                    assert(0);
                }
            }
        }
    }
}

template<bool OneDimension>
void ProcessResults(
    const TGPUCatboostEvaluationContext& ctx,
    NCB::NModelEvaluation::EPredictionType predictionType,
    size_t objectsCount,
    size_t treeStart) {
    const ui32 addBias = (treeStart == 0) ? 1u : 0u;

    switch (predictionType) {
        case NCB::NModelEvaluation::EPredictionType::RawFormulaVal:
            ProcessResultsImpl<NCB::NModelEvaluation::EPredictionType::RawFormulaVal, OneDimension><<<1, 256, 0, ctx.Stream>>> (
                ctx.EvalDataCache.ResultsFloatBuf.Get(),
                objectsCount,
                ctx.GPUModelData.Bias.Get(),
                addBias,
                ctx.GPUModelData.Scale,
                ctx.EvalDataCache.ResultsDoubleBuf.Get(),
                ctx.GPUModelData.ApproxDimension
            );
            break;
        case NCB::NModelEvaluation::EPredictionType::Exponent:
        case NCB::NModelEvaluation::EPredictionType::RMSEWithUncertainty:
        case NCB::NModelEvaluation::EPredictionType::MultiProbability:
            ythrow yexception() << "Unimplemented on GPU: prediction type " << ToString(predictionType);
            break;
        case NCB::NModelEvaluation::EPredictionType::Probability:
            ProcessResultsImpl<NCB::NModelEvaluation::EPredictionType::Probability, OneDimension><<<1, 256, 0, ctx.Stream>>> (
                ctx.EvalDataCache.ResultsFloatBuf.Get(),
                objectsCount,
                ctx.GPUModelData.Bias.Get(),
                addBias,
                ctx.GPUModelData.Scale,
                ctx.EvalDataCache.ResultsDoubleBuf.Get(),
                ctx.GPUModelData.ApproxDimension
            );
            break;
        case NCB::NModelEvaluation::EPredictionType::Class:
            ProcessResultsImpl<NCB::NModelEvaluation::EPredictionType::Class, OneDimension><<<1, 256, 0, ctx.Stream>>> (
                ctx.EvalDataCache.ResultsFloatBuf.Get(),
                objectsCount,
                ctx.GPUModelData.Bias.Get(),
                addBias,
                ctx.GPUModelData.Scale,
                ctx.EvalDataCache.ResultsDoubleBuf.Get(),
                ctx.GPUModelData.ApproxDimension
            );
            break;
    }
}


void TGPUCatboostEvaluationContext::EvalQuantizedData(
    const TCudaQuantizedData* data,
    size_t treeStart,
    size_t treeEnd,
    TArrayRef<double> result,
    NCB::NModelEvaluation::EPredictionType predictionType
    ) const {
    if (treeEnd == 0) {
        treeEnd = GPUModelData.TreeSizes.Size();
    }
    CB_ENSURE(treeStart <= treeEnd, "Invalid tree range: treeStart > treeEnd");
    CB_ENSURE(treeEnd <= GPUModelData.TreeSizes.Size(), "Invalid treeEnd");

    const size_t treeCountToEval = treeEnd - treeStart;
    CB_ENSURE(treeCountToEval > 0, "Empty tree range");
    const ui32 treeCountToEval32 = static_cast<ui32>(treeCountToEval);

    const size_t resultsSize = data->GetObjectsCount() * GPUModelData.ApproxDimension;
    EvalDataCache.PrepareCopyBufs(/*bufSize*/ 0, resultsSize);

    const dim3 treeCalcDimBlock(EvalDocBlockSize, TreeSubBlockWidth);
    const dim3 treeCalcDimGrid(
        NKernel::CeilDivide<unsigned int>(treeCountToEval32, TreeSubBlockWidth * ExtTreeBlockWidth),
        NKernel::CeilDivide<unsigned int>(data->GetObjectsCount(), EvalDocBlockSize * ObjectsPerThread),
        static_cast<unsigned int>(GPUModelData.ApproxDimension)
    );
    ClearMemoryAsync(EvalDataCache.ResultsFloatBuf.AsArrayRef(), Stream);
    EvalObliviousTrees<<<treeCalcDimGrid, treeCalcDimBlock, 0, Stream>>> (
        data->BinarizedFeaturesBuffer.Get(),
        GPUModelData.TreeSizes.Get() + treeStart,
        treeCountToEval32,
        GPUModelData.TreeStartOffsets.Get() + treeStart,
        GPUModelData.TreeSplits.Get(),
        GPUModelData.TreeFirstLeafOffsets.Get() + treeStart,
        GPUModelData.BucketsCount,
        GPUModelData.ModelLeafs.Get(),
        data->GetObjectsCount(),
        static_cast<ui32>(GPUModelData.ApproxDimension),
        EvalDataCache.ResultsFloatBuf.Get()
    );

    if (GPUModelData.ApproxDimension == 1) {
        ProcessResults<true>(*this, predictionType, data->GetObjectsCount(), treeStart);
    } else {
        ProcessResults<false>(*this, predictionType, data->GetObjectsCount(), treeStart);
    }

    const size_t predictionDim = (predictionType == NCB::NModelEvaluation::EPredictionType::Class)
        ? 1u
        : GPUModelData.ApproxDimension;
    NCuda::MemoryCopyAsync<double>(
        EvalDataCache.ResultsDoubleBuf.Slice(0, data->GetObjectsCount() * predictionDim),
        result,
        Stream
    );
}

void TGPUCatboostEvaluationContext::QuantizeData(const TGPUDataInput& dataInput, TCudaQuantizedData* quantizedData) const{
    const dim3 quantizationDimBlock(QuantizationDocBlockSize, 1);
    if (GPUModelData.FloatBucketCount > 0) {
        const dim3 quantizationDimGrid(
            NKernel::CeilDivide<unsigned int>(dataInput.ObjectCount, QuantizationDocBlockSize * ObjectsPerThread),
            GPUModelData.FloatBucketCount // float buckets from model
        );
        if (dataInput.FloatFeatureLayout == TGPUDataInput::EFeatureLayout::ColumnFirst) {
            TFeatureAccessor<float, TGPUDataInput::EFeatureLayout::ColumnFirst> floatFeatureAccessor;
            floatFeatureAccessor.FeatureCount = dataInput.FloatFeatureCount;
            floatFeatureAccessor.Stride = dataInput.Stride;
            floatFeatureAccessor.ObjectCount = dataInput.ObjectCount;
            floatFeatureAccessor.FeaturesPtr = dataInput.FlatFloatsVector.data();
            Binarize<<<quantizationDimGrid, quantizationDimBlock, 0, Stream>>> (
                floatFeatureAccessor,
                GPUModelData.FlatBordersVector.Get(),
                GPUModelData.BordersOffsets.Get(),
                GPUModelData.BordersCount.Get(),
                GPUModelData.FloatFeatureForBucketIdx.Get(),
                GPUModelData.BucketsCount,
                quantizedData->BinarizedFeaturesBuffer.Get()
            );
        } else {
            TFeatureAccessor<float, TGPUDataInput::EFeatureLayout::RowFirst> floatFeatureAccessor;
            floatFeatureAccessor.FeatureCount = dataInput.FloatFeatureCount;
            floatFeatureAccessor.ObjectCount = dataInput.ObjectCount;
            floatFeatureAccessor.Stride = dataInput.Stride;
            floatFeatureAccessor.FeaturesPtr = dataInput.FlatFloatsVector.data();
            Binarize<<<quantizationDimGrid, quantizationDimBlock, 0, Stream>>> (
                floatFeatureAccessor,
                GPUModelData.FlatBordersVector.Get(),
                GPUModelData.BordersOffsets.Get(),
                GPUModelData.BordersCount.Get(),
                GPUModelData.FloatFeatureForBucketIdx.Get(),
                GPUModelData.BucketsCount,
                quantizedData->BinarizedFeaturesBuffer.Get()
            );
        }
    }

    if (GPUModelData.OneHotBucketCount > 0) {
        CB_ENSURE(dataInput.CatFeatureCount > 0, "Model requires categorical features, but no categorical data provided");
        CB_ENSURE(
            dataInput.CatFeatureLayout == TGPUDataInput::EFeatureLayout::ColumnFirst,
            "Only column-first categorical feature layout is supported"
        );

        const dim3 oneHotGrid(
            NKernel::CeilDivide<unsigned int>(dataInput.ObjectCount, QuantizationDocBlockSize * ObjectsPerThread),
            GPUModelData.OneHotBucketCount
        );

        THashedCatFeatureAccessor<TGPUDataInput::EFeatureLayout::ColumnFirst> catAccessor;
        catAccessor.FeatureCount = dataInput.CatFeatureCount;
        catAccessor.Stride = dataInput.Stride;
        catAccessor.ObjectCount = dataInput.ObjectCount;
        catAccessor.FeaturesPtr = dataInput.HashedFlatCatFeatures.data();

        BinarizeOneHot<<<oneHotGrid, quantizationDimBlock, 0, Stream>>>(
            catAccessor,
            GPUModelData.FlatOneHotValues.Get(),
            GPUModelData.OneHotValuesOffsets.Get(),
            GPUModelData.OneHotValuesCount.Get(),
            GPUModelData.OneHotCatFeaturePackedIdxForBucket.Get(),
            /*oneHotBucketOffset*/ GPUModelData.FloatBucketCount,
            GPUModelData.BucketsCount,
            quantizedData->BinarizedFeaturesBuffer.Get()
        );
    }

    if (GPUModelData.CtrFeatureCount > 0) {
        CB_ENSURE(dataInput.CatFeatureLayout == TGPUDataInput::EFeatureLayout::ColumnFirst, "Only column-first categorical feature layout is supported");
        CB_ENSURE(dataInput.FloatFeatureLayout == TGPUDataInput::EFeatureLayout::ColumnFirst, "Only column-first float feature layout is supported");
        CB_ENSURE(dataInput.CatFeatureCount > 0, "Model requires categorical features for CTRs, but no categorical data provided");

        EvalDataCache.PrepareCtrBuf(dataInput.ObjectCount);

        const ui32 ctrBucketsOffset = GPUModelData.FloatBucketCount + GPUModelData.OneHotBucketCount;
        const ui32 blockSize = 256;
        const ui32 numBlocks = (static_cast<ui32>(dataInput.ObjectCount) + blockSize - 1) / blockSize;

        for (ui32 ctrFeatureId = 0; ctrFeatureId < GPUModelData.CtrFeatureCount; ++ctrFeatureId) {
            ComputeCtrValuesImpl<<<numBlocks, blockSize, 0, Stream>>>(
                dataInput.FlatFloatsVector.data(),
                dataInput.FloatFeatureCount,
                dataInput.HashedFlatCatFeatures.data(),
                dataInput.CatFeatureCount,
                dataInput.Stride,
                static_cast<ui32>(dataInput.ObjectCount),
                ctrFeatureId,
                GPUModelData.CtrFeatureTableIdx.Get(),
                GPUModelData.CtrFeatureType.Get(),
                GPUModelData.CtrFeatureTargetBorderIdx.Get(),
                GPUModelData.CtrFeaturePriorNum.Get(),
                GPUModelData.CtrFeaturePriorDenom.Get(),
                GPUModelData.CtrFeatureShift.Get(),
                GPUModelData.CtrFeatureScale.Get(),
                GPUModelData.CtrFeatureCatOffsets.Get(),
                GPUModelData.CtrFeatureCatCount.Get(),
                GPUModelData.CtrProjCatFeaturePackedIdx.Get(),
                GPUModelData.CtrFeatureFloatOffsets.Get(),
                GPUModelData.CtrFeatureFloatCount.Get(),
                GPUModelData.CtrProjFloatSplits.Get(),
                GPUModelData.CtrFeatureOneHotOffsets.Get(),
                GPUModelData.CtrFeatureOneHotCount.Get(),
                GPUModelData.CtrProjOneHotSplits.Get(),
                GPUModelData.CtrIndexBuckets.Get(),
                GPUModelData.CtrTableBucketsOffsets.Get(),
                GPUModelData.CtrTableBucketsCount.Get(),
                GPUModelData.CtrTableDataKind.Get(),
                GPUModelData.CtrTableMeanOffsets.Get(),
                GPUModelData.CtrMeanData.Get(),
                GPUModelData.CtrTableIntOffsets.Get(),
                GPUModelData.CtrIntData.Get(),
                GPUModelData.CtrTableCounterDenom.Get(),
                GPUModelData.CtrTableTargetClassesCount.Get(),
                EvalDataCache.CtrTempBuf.Get()
            );

            const ui32 ctrBucketStart = GPUModelData.CtrFeatureBucketOffsets[ctrFeatureId];
            const ui32 ctrBucketEnd = GPUModelData.CtrFeatureBucketOffsets[ctrFeatureId + 1];
            const ui32 bucketCount = ctrBucketEnd - ctrBucketStart;
            if (bucketCount == 0) {
                continue;
            }

            const dim3 ctrBinarizeGrid(
                NKernel::CeilDivide<unsigned int>(dataInput.ObjectCount, QuantizationDocBlockSize * ObjectsPerThread),
                bucketCount
            );
            BinarizeCtrValues<<<ctrBinarizeGrid, quantizationDimBlock, 0, Stream>>>(
                EvalDataCache.CtrTempBuf.Get(),
                static_cast<ui32>(dataInput.ObjectCount),
                GPUModelData.FlatCtrBordersVector.Get(),
                GPUModelData.CtrBordersOffsets.Get() + ctrBucketStart,
                GPUModelData.CtrBordersCount.Get() + ctrBucketStart,
                ctrBucketsOffset + ctrBucketStart,
                GPUModelData.BucketsCount,
                quantizedData->BinarizedFeaturesBuffer.Get()
            );
        }
    }
}

void TGPUCatboostEvaluationContext::EvalData(
    const TGPUDataInput& dataInput,
    size_t treeStart,
    size_t treeEnd,
    TArrayRef<double> result,
    NCB::NModelEvaluation::EPredictionType predictionType) const {
    TCudaQuantizedData quantizedData;
    quantizedData.SetDimensions(GPUModelData.BucketsCount, dataInput.ObjectCount);
    QuantizeData(dataInput, &quantizedData);
    EvalQuantizedData(&quantizedData, treeStart, treeEnd, result, predictionType);
}

void TGPUCatboostEvaluationContext::CalcLeafIndexesOnDevice(
    const TGPUDataInput& dataInput,
    size_t treeStart,
    size_t treeEnd,
    TArrayRef<NCB::NModelEvaluation::TCalcerIndexType> indexes
) const {
    CB_ENSURE(treeStart < treeEnd, "Invalid tree range");
    CB_ENSURE(treeEnd <= GPUModelData.TreeSizes.Size(), "Invalid treeEnd");

    const size_t treeCountToEval = treeEnd - treeStart;
    const ui32 treeCountToEval32 = static_cast<ui32>(treeCountToEval);

    const size_t expectedSize = static_cast<size_t>(dataInput.ObjectCount) * treeCountToEval;
    CB_ENSURE(indexes.size() == expectedSize, "Leaf indexes buffer size mismatch");
    if (expectedSize == 0) {
        return;
    }

    TCudaQuantizedData quantizedData;
    quantizedData.SetDimensions(GPUModelData.BucketsCount, dataInput.ObjectCount);
    QuantizeData(dataInput, &quantizedData);

    EvalDataCache.PrepareLeafIndexesBuf(expectedSize);

    const dim3 treeCalcDimBlock(EvalDocBlockSize, TreeSubBlockWidth);
    const dim3 treeCalcDimGrid(
        NKernel::CeilDivide<unsigned int>(treeCountToEval32, TreeSubBlockWidth * ExtTreeBlockWidth),
        NKernel::CeilDivide<unsigned int>(dataInput.ObjectCount, EvalDocBlockSize * ObjectsPerThread)
    );
    CalcObliviousLeafIndexes<<<treeCalcDimGrid, treeCalcDimBlock, 0, Stream>>>(
        quantizedData.BinarizedFeaturesBuffer.Get(),
        GPUModelData.TreeSizes.Get() + treeStart,
        treeCountToEval32,
        GPUModelData.TreeStartOffsets.Get() + treeStart,
        GPUModelData.TreeSplits.Get(),
        GPUModelData.BucketsCount,
        static_cast<ui32>(dataInput.ObjectCount),
        EvalDataCache.LeafIndexesBuf.Get()
    );
    CUDA_SAFE_CALL(cudaGetLastError());

    NCuda::MemoryCopyAsync<ui32>(
        EvalDataCache.LeafIndexesBuf.Slice(0, expectedSize),
        TArrayRef<ui32>(reinterpret_cast<ui32*>(indexes.data()), expectedSize),
        Stream
    );
}
