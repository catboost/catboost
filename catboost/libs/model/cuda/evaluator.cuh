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
public:
    void PrepareCopyBufs(size_t bufSize, size_t objectsCount);
};

class TGPUCatboostEvaluationContext {
public:
    TGPUModelData GPUModelData;
    TCudaStream Stream = TCudaStream::ZeroStream();
    mutable TEvaluationDataCache EvalDataCache;
public:
    void QuantizeData(const TGPUDataInput& dataInput, TCudaQuantizedData* quantizedData) const;
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
