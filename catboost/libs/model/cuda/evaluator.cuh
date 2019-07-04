#include <catboost/libs/cuda_wrappers/cuda_vec.h>

#include <catboost/libs/model/evaluation_interface.h>

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
    TCudaVec<float> FlatFloatsVector;
    TCudaVec<ui32> HashedFlatCatFeatures;
};

class TEvaluationDataCache {
public:
    TCudaVec<TCudaQuantizationBucket> BinarizedFeaturesBuffer;
    TCudaVec<TCudaEvaluatorLeafType> EvalResults;
public:
    void PrepareCache(ui32 effectiveBucketCount, ui32 objectsCount);
};

class TGPUCatboostEvaluationContext {
public:
    TGPUModelData GPUModelData;
    TCudaStream Stream = TCudaStream::ZeroStream();
    NCB::NModelEvaluation::EPredictionType PredictionType = NCB::NModelEvaluation::EPredictionType::RawFormulaVal;
    mutable TEvaluationDataCache EvalDataCache;
public:
    void EvalData(const TGPUDataInput& dataInput, TArrayRef<TCudaEvaluatorLeafType> result, size_t treeStart, size_t treeEnd) const;
};
