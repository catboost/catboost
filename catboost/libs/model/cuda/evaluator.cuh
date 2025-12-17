#include <library/cpp/cuda/wrappers/cuda_vec.h>

#include <catboost/libs/model/fwd.h>

#include <util/generic/ptr.h>
#include <util/generic/vector.h>


constexpr ui32 WarpSize = 32;

struct TGPURepackedBin {
    ui32 FeatureIdx = 0;
    ui8 FeatureVal = 0;
    ui8 FeatureXorMask = 0;
};

using TCudaEvaluatorLeafType = float;
using TCudaQuantizationBucket = uchar4;

struct alignas(8) TGPUCtrBucket {
    static constexpr ui64 InvalidHashValue = 0xffffffffffffffffull;
    static constexpr ui32 NotFoundIndex = 0xffffffffu;

    ui64 Hash = InvalidHashValue;
    ui32 IndexValue = 0;
};
static_assert(sizeof(TGPUCtrBucket) == 16, "Expected sizeof(TGPUCtrBucket) == 16 bytes");

struct TGPUCtrMeanHistory {
    float Sum = 0.0f;
    i32 Count = 0;
};

struct TGPUCtrFloatSplit {
    ui32 FloatFeatureIdx = 0;
    float Border = 0.0f;
};

struct TGPUCtrOneHotSplit {
    ui32 CatFeaturePackedIdx = 0;
    ui32 Value = 0;
};

struct TGPUModelData : public TThrRefBase {
    TCudaVec<TGPURepackedBin> TreeSplits;
    TCudaVec<ui32> BordersOffsets;
    TCudaVec<ui32> BordersCount;
    TCudaVec<float> FlatBordersVector;

    TCudaVec<ui32> TreeSizes;
    TCudaVec<ui32> TreeStartOffsets;
    TCudaVec<ui32> TreeFirstLeafOffsets;

    TCudaVec<TCudaEvaluatorLeafType> ModelLeafs;
    TCudaVec<ui32> FloatFeatureForBucketIdx;
    TVector<bool> UsedInModel;

    // Bucket counts follow the same order as TModelTrees::CalcBinFeatures:
    // float/estimated buckets first, then one-hot buckets, then CTR buckets.
    ui32 FloatBucketCount = 0;
    ui32 OneHotBucketCount = 0;
    ui32 CtrBucketCount = 0;
    ui32 BucketsCount = 0;

    // One-hot metadata (bucket-local).
    // For bucket b in [0, OneHotBucketCount):
    //  - OneHotCatFeaturePackedIdxForBucket[b] is packed cat feature index (see CPU quantization).
    //  - FlatOneHotValues[OneHotValuesOffsets[b] .. + OneHotValuesCount[b]) are hashed values for this bucket group.
    TCudaVec<ui32> OneHotCatFeaturePackedIdxForBucket;
    TCudaVec<ui32> OneHotValuesOffsets;
    TCudaVec<ui32> OneHotValuesCount;
    TCudaVec<ui32> FlatOneHotValues;

    // CTR feature metadata.
    ui32 CtrFeatureCount = 0;
    // Host-side offsets for per-feature CTR buckets within the CTR bucket range [0, CtrBucketCount).
    TVector<ui32> CtrFeatureBucketOffsets;

    // CTR bucket borders (bucket-local, in the same order as TModelTrees::GetCtrFeatures()).
    TCudaVec<ui32> CtrBordersOffsets;
    TCudaVec<ui32> CtrBordersCount;
    TCudaVec<float> FlatCtrBordersVector;

    // Uploaded CTR tables (static CTR provider).
    TCudaVec<TGPUCtrBucket> CtrIndexBuckets;
    TCudaVec<ui32> CtrTableBucketsOffsets;
    TCudaVec<ui32> CtrTableBucketsCount;
    TCudaVec<ui8> CtrTableDataKind; // 0: mean history, 1: int array
    TCudaVec<ui32> CtrTableMeanOffsets;
    TCudaVec<ui32> CtrTableMeanCount;
    TCudaVec<TGPUCtrMeanHistory> CtrMeanData;
    TCudaVec<ui32> CtrTableIntOffsets;
    TCudaVec<ui32> CtrTableIntCount;
    TCudaVec<int> CtrIntData;
    TCudaVec<int> CtrTableCounterDenom;
    TCudaVec<int> CtrTableTargetClassesCount;

    // Per-CTR-feature evaluation descriptors (same order as TModelTrees::GetCtrFeatures()).
    TCudaVec<ui32> CtrFeatureTableIdx;
    TCudaVec<ui8> CtrFeatureType; // ECtrType as ui8
    TCudaVec<ui32> CtrFeatureTargetBorderIdx;
    TCudaVec<float> CtrFeaturePriorNum;
    TCudaVec<float> CtrFeaturePriorDenom;
    TCudaVec<float> CtrFeatureShift;
    TCudaVec<float> CtrFeatureScale;

    TCudaVec<ui32> CtrFeatureCatOffsets;
    TCudaVec<ui32> CtrFeatureCatCount;
    TCudaVec<ui32> CtrProjCatFeaturePackedIdx;

    TCudaVec<ui32> CtrFeatureFloatOffsets;
    TCudaVec<ui32> CtrFeatureFloatCount;
    TCudaVec<TGPUCtrFloatSplit> CtrProjFloatSplits;

    TCudaVec<ui32> CtrFeatureOneHotOffsets;
    TCudaVec<ui32> CtrFeatureOneHotCount;
    TCudaVec<TGPUCtrOneHotSplit> CtrProjOneHotSplits;

    size_t ApproxDimension = 0;

    TCudaVec<double> Bias;
    double Scale = 0.0;
};

struct TGPUDataInput {
    enum class EFeatureLayout {
        ColumnFirst,
        RowFirst
    };
    i32 ObjectCount = 0;
    ui32 FloatFeatureCount = 0;
    ui32 CatFeatureCount = 0;
    ui32 Stride = 0;
    EFeatureLayout FloatFeatureLayout = EFeatureLayout::ColumnFirst;
    EFeatureLayout CatFeatureLayout = EFeatureLayout::ColumnFirst;
    TConstArrayRef<float> FlatFloatsVector;
    TConstArrayRef<ui32> HashedFlatCatFeatures;
};

class TCudaQuantizedData : public NCB::NModelEvaluation::IQuantizedData {
public:
    size_t GetObjectsCount() const override {
        return ObjectsCount;
    }
    void SetDimensions(ui32 effectiveBucketCount, ui32 objectsCount);
public:
    TCudaVec<TCudaQuantizationBucket> BinarizedFeaturesBuffer;
private:
    size_t ObjectsCount = 0;
    size_t EffectiveBucketCount = 0;
};

class TEvaluationDataCache {
public:
    TCudaVec<float> CopyDataBufHost;
    TCudaVec<float> CopyDataBufDevice;
    TCudaVec<float> ResultsFloatBuf;
    TCudaVec<double> ResultsDoubleBuf;
    TCudaVec<float> CtrTempBuf;
    TCudaVec<ui32> LeafIndexesBuf;
public:
    void PrepareCopyBufs(size_t bufSize, size_t resultsSize);
    void PrepareCtrBuf(size_t objectsCount);
    void PrepareLeafIndexesBuf(size_t indexesSize);
};

class TGPUCatboostEvaluationContext {
public:
    TGPUModelData GPUModelData;
    TCudaStream Stream = TCudaStream::ZeroStream();
    mutable TEvaluationDataCache EvalDataCache;
public:
    void QuantizeData(const TGPUDataInput& dataInput, TCudaQuantizedData* quantizedData) const;
    void CalcLeafIndexesOnDevice(
        const TGPUDataInput& dataInput,
        size_t treeStart,
        size_t treeEnd,
        TArrayRef<NCB::NModelEvaluation::TCalcerIndexType> indexes) const;
    void EvalQuantizedData(
        const TCudaQuantizedData* data,
        size_t treeStart,
        size_t treeEnd,
        TArrayRef<double> result,
        NCB::NModelEvaluation::EPredictionType predictionType) const;
    void EvalData(
        const TGPUDataInput& dataInput,
        size_t treeStart,
        size_t treeEnd,
        TArrayRef<double> result,
        NCB::NModelEvaluation::EPredictionType predictionType) const;
};
