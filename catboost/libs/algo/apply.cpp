#include "apply.h"
#include "index_calcer.h"
#include "features_data_helpers.h"

#include <catboost/libs/cat_feature/cat_feature.h>
#include <catboost/libs/data_new/data_provider.h>
#include <catboost/libs/data_new/model_dataset_compatibility.h>
#include <catboost/libs/eval_result/eval_helpers.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/logging/logging.h>

#include <util/generic/array_ref.h>
#include <util/generic/cast.h>
#include <util/generic/utility.h>

#include <cmath>

using namespace NCB;
using NPar::TLocalExecutor;

TLocalExecutor::TExecRangeParams GetBlockParams(int executorThreadCount, int docCount, int begin, int end) {
    const int threadCount = executorThreadCount + 1; // one for current thread
    const int minBlockSize = ceil(10000.0 / sqrt(end - begin + 1)); // for 1 iteration it will be 7k docs, for 10k iterations it will be 100 docs.
    const int effectiveBlockCount = Min(threadCount, (docCount + minBlockSize - 1) / minBlockSize);

    TLocalExecutor::TExecRangeParams blockParams(0, docCount);
    blockParams.SetBlockCount(effectiveBlockCount);
    return blockParams;
};

static void ApplyOnRawFeatures(
    const TFullModel& model,
    const TRawObjectsDataProvider* rawObjectsData,
    int begin,
    int end,
    const THashMap<ui32, ui32>& columnReorderMap,
    const TLocalExecutor::TExecRangeParams& blockParams,
    TVector<double>* approxesFlat,
    TLocalExecutor* executor)
{
    const int approxesDimension = model.ObliviousTrees.ApproxDimension;
    int consecutiveSubsetBegin = GetConsecutiveSubsetBegin(*rawObjectsData);

    const auto getFeatureDataBeginPtr = [&](ui32 flatFeatureIdx, TVector<TMaybe<TPackedBinaryIndex>>*) -> const float* {
        return GetRawFeatureDataBeginPtr(
            *rawObjectsData,
            consecutiveSubsetBegin,
            flatFeatureIdx);
    };
    const auto applyOnBlock = [&](int blockId) {
        const int blockFirstIdx = blockParams.FirstId + blockId * blockParams.GetBlockSize();
        const int blockLastIdx = Min(blockParams.LastId, blockFirstIdx + blockParams.GetBlockSize());
        const int blockSize = blockLastIdx - blockFirstIdx;

        TVector<TConstArrayRef<float>> repackedFeatures;
        GetRepackedFeatures(
            blockFirstIdx,
            blockLastIdx,
            model.ObliviousTrees.GetFlatFeatureVectorExpectedSize(),
            columnReorderMap,
            getFeatureDataBeginPtr,
            *rawObjectsData->GetFeaturesLayout(),
            &repackedFeatures);

        model.CalcFlatTransposed(
            repackedFeatures,
            begin,
            end,
            MakeArrayRef(
                approxesFlat->data() + blockFirstIdx * approxesDimension,
                blockSize * approxesDimension));
    };
    if (executor) {
        executor->ExecRange(applyOnBlock, 0, blockParams.GetBlockCount(), TLocalExecutor::WAIT_COMPLETE);
    } else {
        applyOnBlock(0);
    }
}

static void ApplyOnQuantizedFeatures(
    const TFullModel& model,
    const TQuantizedForCPUObjectsDataProvider& quantizedObjectsData,
    int begin,
    int end,
    const THashMap<ui32, ui32>& columnReorderMap,
    const TLocalExecutor::TExecRangeParams& blockParams,
    TVector<double>* approxesFlat,
    TLocalExecutor* executor)
{
    const int approxesDimension = model.ObliviousTrees.ApproxDimension;
    int consecutiveSubsetBegin = GetConsecutiveSubsetBegin(quantizedObjectsData);

    auto floatBinsRemap = GetFloatFeaturesBordersRemap(model, *quantizedObjectsData.GetQuantizedFeaturesInfo().Get());

    auto getFeatureDataBeginPtr = [&](ui32 featureIdx, TVector<TMaybe<TPackedBinaryIndex>>* packedIdx) -> const ui8* {
        (*packedIdx)[featureIdx] = quantizedObjectsData.GetFloatFeatureToPackedBinaryIndex(TFeatureIdx<EFeatureType::Float>(featureIdx));
        if (!(*packedIdx)[featureIdx].Defined()) {
            return GetQuantizedForCpuFloatFeatureDataBeginPtr(
                quantizedObjectsData,
                consecutiveSubsetBegin,
                featureIdx);
        } else {
            return (**quantizedObjectsData.GetBinaryFeaturesPack((*packedIdx)[featureIdx]->PackIdx).GetSrc()).Data();
        }
    };

    const auto applyOnBlock = [&](int blockId) {
        const int blockFirstIdx = blockParams.FirstId + blockId * blockParams.GetBlockSize();
        const int blockLastIdx = Min(blockParams.LastId, blockFirstIdx + blockParams.GetBlockSize());
        const int blockSize = blockLastIdx - blockFirstIdx;

        TVector<TConstArrayRef<ui8>> repackedFeatures;
        TVector<TMaybe<TPackedBinaryIndex>> packedIndexes;
        GetRepackedFeatures(
            blockFirstIdx,
            blockLastIdx,
            model.ObliviousTrees.GetFlatFeatureVectorExpectedSize(),
            columnReorderMap,
            getFeatureDataBeginPtr,
            *quantizedObjectsData.GetFeaturesLayout(),
            &repackedFeatures,
            &packedIndexes);

        constexpr bool isQuantized = true;
        CalcGeneric<isQuantized>(
            model,
            [&floatBinsRemap, &repackedFeatures, &packedIndexes](const TFloatFeature& floatFeature, size_t index) -> ui8 {
                return QuantizedFeaturesFloatAccessor(floatBinsRemap, repackedFeatures, packedIndexes, floatFeature, index);
            },
            [](const TCatFeature&, size_t) -> ui32 {
                Y_ASSERT("Quantized datasets with categorical features are not currently supported");
                return 0;
            },
            blockSize,
            begin,
            end,
            MakeArrayRef(
                approxesFlat->data() + blockFirstIdx * approxesDimension,
                blockSize * approxesDimension));
    };
    if (executor) {
        executor->ExecRange(applyOnBlock, 0, blockParams.GetBlockCount(), TLocalExecutor::WAIT_COMPLETE);
    } else {
        applyOnBlock(0);
    }
}


TVector<TVector<double>> ApplyModelMulti(
    const TFullModel& model,
    const TObjectsDataProvider& objectsData,
    const EPredictionType predictionType,
    int begin, /*= 0*/
    int end,   /*= 0*/
    TLocalExecutor* executor)
{
    const int docCount = SafeIntegerCast<int>(objectsData.GetObjectCount());
    const int approxesDimension = model.ObliviousTrees.ApproxDimension;
    TVector<double> approxesFlat(docCount * approxesDimension);
    if (docCount > 0) {
        end = end == 0 ? model.GetTreeCount() : Min<int>(end, model.GetTreeCount());
        const int executorThreadCount = executor ? executor->GetThreadCount() : 0;
        auto blockParams = GetBlockParams(executorThreadCount, docCount, begin, end);

        THashMap<ui32, ui32> columnReorderMap;
        CheckModelAndDatasetCompatibility(model, objectsData, &columnReorderMap);

        if (const auto *const rawObjectsData = dynamic_cast<const TRawObjectsDataProvider*>(&objectsData)) {
            ApplyOnRawFeatures(
                model,
                rawObjectsData,
                begin,
                end,
                columnReorderMap,
                blockParams,
                &approxesFlat,
                executor);
        } else if (const auto *const quantizedObjectsData = dynamic_cast<const TQuantizedForCPUObjectsDataProvider*>(&objectsData)) {
            ApplyOnQuantizedFeatures(
                model,
                *quantizedObjectsData,
                begin,
                end,
                columnReorderMap,
                blockParams,
                &approxesFlat,
                executor);
        } else {
            ythrow TCatBoostException() << "Unsupported objects data - neither raw nor quantized for CPU";
        }
    }

    TVector<TVector<double>> approxes(approxesDimension);
    if (approxesDimension == 1) { //shortcut
        approxes[0].swap(approxesFlat);
    } else {
        for (int dim = 0; dim < approxesDimension; ++dim) {
            approxes[dim].yresize(docCount);
            for (int doc = 0; doc < docCount; ++doc) {
                approxes[dim][doc] = approxesFlat[approxesDimension * doc + dim];
            };
        }
    }

    if (predictionType == EPredictionType::InternalRawFormulaVal) {
        //shortcut
        return approxes;
    } else {
        return PrepareEvalForInternalApprox(predictionType, model, approxes, executor);
    }
}

TVector<TVector<double>> ApplyModelMulti(const TFullModel& model,
                                         const TObjectsDataProvider& objectsData,
                                         bool verbose,
                                         const EPredictionType predictionType,
                                         int begin,
                                         int end,
                                         int threadCount) {
    TSetLoggingVerboseOrSilent inThisScope(verbose);

    NPar::TLocalExecutor executor;
    executor.RunAdditionalThreads(threadCount - 1);
    const auto& result = ApplyModelMulti(model, objectsData, predictionType, begin, end, &executor);
    return result;
}

TVector<TVector<double>> ApplyModelMulti(const TFullModel& model,
                                         const TDataProvider& data,
                                         bool verbose,
                                         const EPredictionType predictionType,
                                         int begin,
                                         int end,
                                         int threadCount) {

    return ApplyModelMulti(model, *data.ObjectsData, verbose, predictionType, begin, end, threadCount);
}

TVector<double> ApplyModel(const TFullModel& model,
                           const TObjectsDataProvider& objectsData,
                           bool verbose,
                           const EPredictionType predictionType,
                           int begin, /*= 0*/
                           int end,   /*= 0*/
                           int threadCount /*= 1*/) {
    return ApplyModelMulti(model, objectsData, verbose, predictionType, begin, end, threadCount)[0];
}


void TModelCalcerOnPool::ApplyModelMulti(
    const EPredictionType predictionType,
    int begin,
    int end,
    TVector<double>* flatApproxBuffer,
    TVector<TVector<double>>* approx)
{
    const ui32 docCount = ObjectsData->GetObjectCount();
    auto approxDimension = SafeIntegerCast<ui32>(Model->ObliviousTrees.ApproxDimension);
    TVector<double>& approxFlat = *flatApproxBuffer;
    approxFlat.resize(static_cast<unsigned long>(docCount * approxDimension)); // TODO(annaveronika): yresize?

    if (end == 0) {
        end = Model->GetTreeCount();
    } else {
        end = Min<int>(end, Model->GetTreeCount());
    }

    Executor->ExecRange([&](int blockId) {
        auto& calcer = *ThreadCalcers[blockId];
        const int blockFirstId = BlockParams.FirstId + blockId * BlockParams.GetBlockSize();
        const int blockLastId = Min(BlockParams.LastId, blockFirstId + BlockParams.GetBlockSize());
        TArrayRef<double> resultRef(approxFlat.data() + blockFirstId * approxDimension, (blockLastId - blockFirstId) * approxDimension);
        calcer.Calc(begin, end, resultRef);
    }, 0, BlockParams.GetBlockCount(), NPar::TLocalExecutor::WAIT_COMPLETE);

    approx->resize(approxDimension);

    if (approxDimension == 1) { //shortcut
        (*approx)[0].swap(approxFlat);
    } else {
        for (auto& approxProjection : *approx) {
            approxProjection.clear();
            approxProjection.resize(docCount);
        }
        for (ui32 dim = 0; dim < approxDimension; ++dim) {
            for (ui32 doc = 0; doc < docCount; ++doc) {
                (*approx)[dim][doc] = approxFlat[approxDimension * doc + dim];
            };
        }
    }

    if (predictionType == EPredictionType::InternalRawFormulaVal) {
        //shortcut
        return;
    } else {
        (*approx) = PrepareEvalForInternalApprox(predictionType, *Model, *approx, Executor);
    }
    flatApproxBuffer->clear();
}

void TModelCalcerOnPool::InitForRawFeatures(
    const TFullModel& model,
    const TRawObjectsDataProvider& rawObjectsData,
    const THashMap<ui32, ui32>& columnReorderMap,
    const TLocalExecutor::TExecRangeParams& blockParams,
    TLocalExecutor* executor)
{
    const ui32 consecutiveSubsetBegin = GetConsecutiveSubsetBegin(rawObjectsData);
    const auto& featuresLayout = *rawObjectsData.GetFeaturesLayout();

    auto getFeatureDataBeginPtr = [&](ui32 flatFeatureIdx, TVector<TMaybe<TPackedBinaryIndex>>*) -> const float* {
        return GetRawFeatureDataBeginPtr(
            rawObjectsData,
            consecutiveSubsetBegin,
            flatFeatureIdx);
    };

    executor->ExecRange([&](int blockId) {
        const int blockFirstIdx = blockParams.FirstId + blockId * blockParams.GetBlockSize();
        const int blockLastIdx = Min(blockParams.LastId, blockFirstIdx + blockParams.GetBlockSize());

        TVector<TConstArrayRef<float>> repackedFeatures;
        GetRepackedFeatures(
            blockFirstIdx,
            blockLastIdx,
            model.ObliviousTrees.GetFlatFeatureVectorExpectedSize(),
            columnReorderMap,
            getFeatureDataBeginPtr,
            featuresLayout,
            &repackedFeatures);

        auto floatAccessor = [&repackedFeatures](const TFloatFeature& floatFeature, size_t index) -> float {
            return repackedFeatures[floatFeature.FlatFeatureIndex][index];
        };

        auto catAccessor = [&repackedFeatures](const TCatFeature& catFeature, size_t index) -> ui32 {
            return ConvertFloatCatFeatureToIntHash(repackedFeatures[catFeature.FlatFeatureIndex][index]);
        };
        ui64 docCount = ui64(blockLastIdx - blockFirstIdx);
        ThreadCalcers[blockId] = MakeHolder<TFeatureCachedTreeEvaluator>(model, floatAccessor, catAccessor, docCount);
    }, 0, blockParams.GetBlockCount(), NPar::TLocalExecutor::WAIT_COMPLETE);
}

void TModelCalcerOnPool::InitForQuantizedFeatures(
    const TFullModel& model,
    const TQuantizedForCPUObjectsDataProvider& quantizedObjectsData,
    const THashMap<ui32, ui32>& columnReorderMap,
    const NPar::TLocalExecutor::TExecRangeParams& blockParams,
    NPar::TLocalExecutor* executor)
{
    const ui32 consecutiveSubsetBegin = GetConsecutiveSubsetBegin(quantizedObjectsData);
    const auto& featuresLayout = *quantizedObjectsData.GetFeaturesLayout();

    executor->ExecRange([&](int blockId) {
        const int blockFirstIdx = blockParams.FirstId + blockId * blockParams.GetBlockSize();
        const int blockLastIdx = Min(blockParams.LastId, blockFirstIdx + blockParams.GetBlockSize());

        auto floatBinsRemap = GetFloatFeaturesBordersRemap(model, *quantizedObjectsData.GetQuantizedFeaturesInfo().Get());

        TVector<TConstArrayRef<ui8>> repackedFeatures;
        TVector<TMaybe<TPackedBinaryIndex>> packedIndexes;
        GetRepackedFeatures(
            blockFirstIdx,
            blockLastIdx,
            model.ObliviousTrees.GetFlatFeatureVectorExpectedSize(),
            columnReorderMap,
            [&quantizedObjectsData, &consecutiveSubsetBegin](ui32 featureIdx, TVector<TMaybe<TPackedBinaryIndex>>* packedIdx) -> const ui8* {
                return  GetFeatureDataBeginPtr(quantizedObjectsData, featureIdx, consecutiveSubsetBegin, packedIdx);
            },
            featuresLayout,
            &repackedFeatures,
            &packedIndexes);

        ui64 docCount = ui64(blockLastIdx - blockFirstIdx);
        ThreadCalcers[blockId] = MakeHolder<TFeatureCachedTreeEvaluator>(
            model,
            [&floatBinsRemap, &repackedFeatures, &packedIndexes](const TFloatFeature& floatFeature, size_t index) -> ui8 {
                return QuantizedFeaturesFloatAccessor(floatBinsRemap, repackedFeatures, packedIndexes, floatFeature, index);
            },
            docCount);
    }, 0, blockParams.GetBlockCount(), NPar::TLocalExecutor::WAIT_COMPLETE);
}

TModelCalcerOnPool::TModelCalcerOnPool(
    const TFullModel& model,
    TObjectsDataProviderPtr objectsData,
    NPar::TLocalExecutor* executor)
    : Model(&model)
    , ObjectsData(objectsData)
    , Executor(executor)
    , BlockParams(0, SafeIntegerCast<int>(objectsData->GetObjectCount()))
{
    if (BlockParams.FirstId == BlockParams.LastId) {
        return;
    }
    THashMap<ui32, ui32> columnReorderMap;
    CheckModelAndDatasetCompatibility(model, *ObjectsData, &columnReorderMap);
    const int threadCount = executor->GetThreadCount() + 1; // one for current thread
    BlockParams.SetBlockCount(threadCount);
    ThreadCalcers.resize(BlockParams.GetBlockCount());
    if (const auto *const rawObjectsData = dynamic_cast<const TRawObjectsDataProvider*>(ObjectsData.Get())) {
        TModelCalcerOnPool::InitForRawFeatures(
            model,
            *rawObjectsData,
            columnReorderMap,
            BlockParams,
            executor);
    } else if (
        const auto *const quantizedObjectsData =
            dynamic_cast<const TQuantizedForCPUObjectsDataProvider*>(ObjectsData.Get()))
    {
        TModelCalcerOnPool::InitForQuantizedFeatures(
            model,
            *quantizedObjectsData,
            columnReorderMap,
            BlockParams,
            executor);
    } else {
        ythrow TCatBoostException() << "Unsupported objects data - neither raw nor quantized for CPU";
    }
}
