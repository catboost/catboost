#include "evaluator.cuh"


#include <catboost/libs/cuda_wrappers/kernel.cuh>
#include <catboost/libs/cuda_wrappers/kernel_helpers.cuh>
#include <catboost/libs/cuda_wrappers/arch.cuh>
#include <catboost/libs/cuda_wrappers/kernel_helpers.cuh>

#include <cuda_runtime.h>
#include <assert.h>

template<typename TFeatureType, TGPUDataInput::EFeatureLayout Layout>
struct TFeatureAccessor {
    TFeatureAccessor() = default;
    using TFeature = TFeatureType;
    using TFeaturePtr = const TFeature*;

    i32 FeatureStride = 0;
    i32 ObjectStride = 0;

    i32 FeatureCount = 0;
    i32 ObjectCount = 0;
    TFeaturePtr FeaturesPtr = nullptr;

    __forceinline__ __device__ TFeature operator()(i32 featureId, i32 objectId) const {
        if (Layout == TGPUDataInput::EFeatureLayout::ColumnFirst) {
            return objectId < ObjectCount && featureId < FeatureCount ?
                __ldg(FeaturesPtr + featureId * FeatureStride + objectId)
                : NegativeInfty();
        } else {
            return objectId < ObjectCount && featureId < FeatureCount ?
                __ldg(FeaturesPtr + featureId + objectId * ObjectStride)
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

constexpr ui32 ObjectsPerThread = 4;
constexpr ui32 TreeSubBlockWidth = 8;
constexpr ui32 ExtTreeBlockWidth = 128;
constexpr ui32 QuantizationDocBlockSize = 256;

constexpr ui32 BlockWidth = 256;
constexpr ui32 EvalDocBlockSize = BlockWidth / TreeSubBlockWidth;

static_assert(EvalDocBlockSize >= WarpSize, "EvalBlockSize should be greater than WarpSize");
using TTreeIndex = uint4;

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

template<int TreeDepth>
TTreeIndex __device__ __forceinline__ CalcIndexesUnwrapped(const TGPURepackedBin* const __restrict__ curRepackedBinPtr, const TCudaQuantizationBucket* const __restrict__ quantizedFeatures) {
    TTreeIndex result = { 0 };
#pragma unroll TreeDepth
    for (int depth = 0; depth < TreeDepth; ++depth) {
        const TGPURepackedBin bin = Ldg(curRepackedBinPtr + depth);
        TCudaQuantizationBucket buckets = __ldg(quantizedFeatures + bin.FeatureIdx);
        result.x |= ((buckets.x) >= bin.FeatureVal) << depth;
        result.y |= ((buckets.y) >= bin.FeatureVal) << depth;
        result.z |= ((buckets.z) >= bin.FeatureVal) << depth;
        result.w |= ((buckets.w) >= bin.FeatureVal) << depth;
    }
    return result;
}

TTreeIndex __device__ CalcIndexesBase(int TreeDepth, const TGPURepackedBin* const __restrict__ curRepackedBinPtr, const TCudaQuantizationBucket* const __restrict__ quantizedFeatures) {
    TTreeIndex bins = { 0 };
    for (int depth = 0; depth < TreeDepth; ++depth) {
        const TGPURepackedBin bin = Ldg(curRepackedBinPtr + depth);
        TCudaQuantizationBucket vals = __ldg(quantizedFeatures + bin.FeatureIdx);
        bins.x |= ((vals.x) >= bin.FeatureVal) << depth;
        bins.y |= ((vals.y) >= bin.FeatureVal) << depth;
        bins.z |= ((vals.z) >= bin.FeatureVal) << depth;
        bins.w |= ((vals.w) >= bin.FeatureVal) << depth;
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
    TCudaEvaluatorLeafType* __restrict__ results) {

    const int innerBlockBy32 = threadIdx.x / WarpSize;
    const int blockby32 = blockIdx.y * EvalDocBlockSize / WarpSize + innerBlockBy32;
    const int inBlockId = threadIdx.x % WarpSize;
    const int firstDocForThread = blockby32 * WarpSize * ObjectsPerThread + inBlockId;

    quantizedFeatures += bucketsCount * WarpSize * blockby32 + threadIdx.x % WarpSize;

    const int firstTreeIdx = TreeSubBlockWidth * ExtTreeBlockWidth * (threadIdx.y + TreeSubBlockWidth * blockIdx.x);
    const int lastTreeIdx = min(firstTreeIdx + TreeSubBlockWidth * ExtTreeBlockWidth, treeCount);
    double4 localResult = { 0 };

    if (firstTreeIdx < lastTreeIdx && firstDocForThread < documentCount) {
        const TGPURepackedBin* __restrict__ curRepackedBinPtr = repackedBins + __ldg(treeStartOffsets + firstTreeIdx);
        leafValues += firstLeafOfset[firstTreeIdx];
        int treeIdx = firstTreeIdx;
        const int lastTreeBy2 = lastTreeIdx - ((lastTreeIdx - firstTreeIdx) & 0x3);
        for (; treeIdx < lastTreeBy2; treeIdx += 2) {
            const int curTreeDepth1 = __ldg(treeSizes + treeIdx);
            const int curTreeDepth2 = __ldg(treeSizes + treeIdx + 1);

            const TTreeIndex bins1 = CalcTreeVals(curTreeDepth1, curRepackedBinPtr, quantizedFeatures);
            const TTreeIndex bins2 = CalcTreeVals(curTreeDepth2, curRepackedBinPtr + curTreeDepth1, quantizedFeatures);
            const auto leafValues2 = leafValues + (1 << curTreeDepth1);
            localResult.x += __ldg(leafValues + bins1.x) + __ldg(leafValues2 + bins2.x);
            localResult.y += __ldg(leafValues + bins1.y) + __ldg(leafValues2 + bins2.y);
            localResult.z += __ldg(leafValues + bins1.z) + __ldg(leafValues2 + bins2.z);
            localResult.w += __ldg(leafValues + bins1.w) + __ldg(leafValues2 + bins2.w);
            curRepackedBinPtr += curTreeDepth1 + curTreeDepth2;
            leafValues = leafValues2 + (1 << curTreeDepth2);
        }
        for (; treeIdx < lastTreeIdx; ++treeIdx) {
            const int curTreeDepth = __ldg(treeSizes + treeIdx);
            const TTreeIndex bins = CalcTreeVals(curTreeDepth, curRepackedBinPtr, quantizedFeatures);
            localResult.x += __ldg(leafValues + bins.x);
            localResult.y += __ldg(leafValues + bins.y);
            localResult.z += __ldg(leafValues + bins.z);
            localResult.w += __ldg(leafValues + bins.w);
            curRepackedBinPtr += curTreeDepth;
            leafValues += (1 << curTreeDepth);
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
        TAtomicAdd<TCudaEvaluatorLeafType>::Add(
            results + blockby32 * WarpSize * ObjectsPerThread + threadIdx.x + threadIdx.y * EvalDocBlockSize,
            reduceVals[threadIdx.x + threadIdx.y * EvalDocBlockSize] + reduceVals[threadIdx.x + threadIdx.y * EvalDocBlockSize + 128]
        );
    }
}

void TEvaluationDataCache::PrepareCache(ui32 effectiveBucketCount, ui32 objectsCount) {
    const auto one32blockSize = WarpSize * effectiveBucketCount;
    const auto desiredQuantBuff = one32blockSize * NKernel::CeilDivide<ui32>(objectsCount, 128) * 4;
    if (BinarizedFeaturesBuffer.Size() < desiredQuantBuff) {
        BinarizedFeaturesBuffer = TCudaVec<TCudaQuantizationBucket>(desiredQuantBuff, EMemoryType::Device);
    }
    if (EvalResults.Size() < objectsCount) {
        EvalResults = TCudaVec<TCudaEvaluatorLeafType>(AlignBy<2048>(objectsCount), EMemoryType::Device);
    }
}

void TGPUCatboostEvaluationContext::EvalData(const TGPUDataInput& dataInput, TArrayRef<TCudaEvaluatorLeafType> result, size_t treeStart, size_t treeEnd) const {
    TFeatureAccessor<float, TGPUDataInput::EFeatureLayout::ColumnFirst> floatFeatureAccessor;
    floatFeatureAccessor.FeatureCount = dataInput.FloatFeatureCount;
    floatFeatureAccessor.FeatureStride = dataInput.Stride;
    floatFeatureAccessor.ObjectCount = dataInput.ObjectCount;
    floatFeatureAccessor.ObjectStride = 1;
    floatFeatureAccessor.FeaturesPtr = dataInput.FlatFloatsVector.Get();

    EvalDataCache.PrepareCache(GPUModelData.FloatFeatureForBucketIdx.Size(), dataInput.ObjectCount);
    const dim3 quantizationDimBlock(QuantizationDocBlockSize, 1);
    const dim3 quantizationDimGrid(
        NKernel::CeilDivide<unsigned int>(dataInput.ObjectCount, QuantizationDocBlockSize * ObjectsPerThread),
        GPUModelData.BordersCount.Size() // float features from models
    );
    Binarize<<<quantizationDimGrid, quantizationDimBlock, 0, Stream>>> (
        floatFeatureAccessor,
        GPUModelData.FlatBordersVector.Get(),
        GPUModelData.BordersOffsets.Get(),
        GPUModelData.BordersCount.Get(),
        GPUModelData.FloatFeatureForBucketIdx.Get(),
        GPUModelData.FloatFeatureForBucketIdx.Size(),
        EvalDataCache.BinarizedFeaturesBuffer.Get()
    );

    const dim3 treeCalcDimBlock(EvalDocBlockSize, TreeSubBlockWidth);
    const dim3 treeCalcDimGrid(
        NKernel::CeilDivide<unsigned int>(GPUModelData.TreeSizes.Size(), TreeSubBlockWidth * ExtTreeBlockWidth),
        NKernel::CeilDivide<unsigned int>(dataInput.ObjectCount, EvalDocBlockSize * ObjectsPerThread)
    );
    ClearMemoryAsync(EvalDataCache.EvalResults.AsArrayRef(), Stream);
    EvalObliviousTrees<<<treeCalcDimGrid, treeCalcDimBlock, 0, Stream>>> (
        EvalDataCache.BinarizedFeaturesBuffer.Get(),
        GPUModelData.TreeSizes.Get(),
        GPUModelData.TreeSizes.Size(),
        GPUModelData.TreeStartOffsets.Get(),
        GPUModelData.TreeSplits.Get(),
        GPUModelData.TreeFirstLeafOffsets.Get(),
        GPUModelData.FloatFeatureForBucketIdx.Size(),
        GPUModelData.ModelLeafs.Get(),
        dataInput.ObjectCount,
        EvalDataCache.EvalResults.Get()
    );
    MemoryCopyAsync<TCudaEvaluatorLeafType>(
        EvalDataCache.EvalResults.AsArrayRef().Slice(0, dataInput.ObjectCount),
        result,
        Stream
    );
    Stream.Synchronize();
}
