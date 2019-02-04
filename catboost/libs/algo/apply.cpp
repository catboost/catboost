#include "apply.h"
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

TVector<TVector<double>> ApplyModelMulti(
    const TFullModel& model,
    const TObjectsDataProvider& objectsData,
    const EPredictionType predictionType,
    int begin, /*= 0*/
    int end,   /*= 0*/
    TLocalExecutor* executor)
{
    const auto* const rawObjectsData = dynamic_cast<const TRawObjectsDataProvider*>(&objectsData);
    CB_ENSURE(rawObjectsData, "Not supported for quantized pools");

    end = end == 0 ? model.GetTreeCount() : Min<int>(end, model.GetTreeCount());
    const int executorThreadCount = executor ? executor->GetThreadCount() : 0;
    const int docCount = SafeIntegerCast<int>(objectsData.GetObjectCount());
    const int approxesDimension = model.ObliviousTrees.ApproxDimension;

    THashMap<ui32, ui32> columnReorderMap;
    CheckModelAndDatasetCompatibility(model, objectsData, &columnReorderMap);

    TVector<double> approxesFlat;
    approxesFlat.yresize(docCount * approxesDimension);
    if (docCount > 0) {
        const ui32 consecutiveSubsetBegin = GetConsecutiveSubsetBegin(*rawObjectsData);
        const int threadCount = executorThreadCount + 1; // one for current thread
        const int minBlockSize = ceil(10000.0 / sqrt(end - begin + 1)); // for 1 iteration it will be 7k docs, for 10k iterations it will be 100 docs.
        const int effectiveBlockCount = Min(threadCount, (docCount + minBlockSize - 1) / minBlockSize);

        TLocalExecutor::TExecRangeParams blockParams(0, docCount);
        blockParams.SetBlockCount(effectiveBlockCount);

        const auto featuresLayout = rawObjectsData->GetFeaturesLayout();
        const auto getFeatureDataPtr = [&](ui32 flatFeatureIdx) -> const float* {
            return GetRawFeatureDataBeginPtr(
                *rawObjectsData,
                *featuresLayout,
                consecutiveSubsetBegin,
                flatFeatureIdx);
        };
        const auto applyOnBlock = [&](int blockId) {
            TVector<TConstArrayRef<float>> repackedFeatures(model.ObliviousTrees.GetFlatFeatureVectorExpectedSize());
            const int blockFirstIdx = blockParams.FirstId + blockId * blockParams.GetBlockSize();
            const int blockLastIdx = Min(blockParams.LastId, blockFirstIdx + blockParams.GetBlockSize());
            const int blockSize = blockLastIdx - blockFirstIdx;
            if (columnReorderMap.empty()) {
                for (size_t i = 0; i < model.ObliviousTrees.GetFlatFeatureVectorExpectedSize(); ++i) {
                    repackedFeatures[i] = MakeArrayRef(getFeatureDataPtr(i) + blockFirstIdx, blockSize);
                }
            } else {
                for (const auto& [origIdx, sourceIdx] : columnReorderMap) {
                    repackedFeatures[origIdx] = MakeArrayRef(getFeatureDataPtr(sourceIdx) + blockFirstIdx, blockSize);
                }
            }
            model.CalcFlatTransposed(
                repackedFeatures,
                begin,
                end,
                MakeArrayRef(
                    approxesFlat.data() + blockFirstIdx * approxesDimension,
                    blockSize * approxesDimension));
        };
        if (executor) {
            executor->ExecRange(applyOnBlock, 0, blockParams.GetBlockCount(), TLocalExecutor::WAIT_COMPLETE);
        } else {
            applyOnBlock(0);
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
    const ui32 docCount = RawObjectsData->GetObjectCount();
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

TModelCalcerOnPool::TModelCalcerOnPool(
    const TFullModel& model,
    TObjectsDataProviderPtr objectsData,
    NPar::TLocalExecutor* executor)
    : Model(&model)
    , RawObjectsData(dynamic_cast<TRawObjectsDataProvider*>(objectsData.Get()))
    , Executor(executor)
    , BlockParams(0, SafeIntegerCast<int>(objectsData->GetObjectCount()))
{
    if (BlockParams.FirstId == BlockParams.LastId) {
        return;
    }
    CB_ENSURE(RawObjectsData, "Not supported for quantized pools");
    THashMap<ui32, ui32> columnReorderMap;
    CheckModelAndDatasetCompatibility(model, *RawObjectsData, &columnReorderMap);

    const int threadCount = executor->GetThreadCount() + 1; // one for current thread
    BlockParams.SetBlockCount(threadCount);
    ThreadCalcers.resize(BlockParams.GetBlockCount());

    const ui32 consecutiveSubsetBegin = GetConsecutiveSubsetBegin(*RawObjectsData);
    const auto& featuresLayout = *RawObjectsData->GetFeaturesLayout();

    auto getFeatureDataBeginPtr = [&](ui32 flatFeatureIdx) -> const float* {
        return GetRawFeatureDataBeginPtr(
            *RawObjectsData,
            featuresLayout,
            consecutiveSubsetBegin,
            flatFeatureIdx);
    };

    executor->ExecRange([&](int blockId) {
        TVector<TConstArrayRef<float>> repackedFeatures(Model->ObliviousTrees.GetFlatFeatureVectorExpectedSize());
        const int blockFirstId = BlockParams.FirstId + blockId * BlockParams.GetBlockSize();
        const int blockLastId = Min(BlockParams.LastId, blockFirstId + BlockParams.GetBlockSize());
        if (columnReorderMap.empty()) {
            for (ui32 i = 0; i < Model->ObliviousTrees.GetFlatFeatureVectorExpectedSize(); ++i) {
                repackedFeatures[i] = MakeArrayRef(getFeatureDataBeginPtr(i) + blockFirstId, blockLastId - blockFirstId);
            }
        } else {
            for (const auto& [origIdx, sourceIdx] : columnReorderMap) {
                repackedFeatures[origIdx] = MakeArrayRef(getFeatureDataBeginPtr(sourceIdx) + blockFirstId, blockLastId - blockFirstId);
            }
        }
        auto floatAccessor = [&repackedFeatures](const TFloatFeature& floatFeature, size_t index) -> float {
            return repackedFeatures[floatFeature.FlatFeatureIndex][index];
        };

        auto catAccessor = [&repackedFeatures](const TCatFeature& catFeature, size_t index) -> ui32 {
            return ConvertFloatCatFeatureToIntHash(repackedFeatures[catFeature.FlatFeatureIndex][index]);
        };
        ui64 docCount = ui64(blockLastId - blockFirstId);
        ThreadCalcers[blockId] = MakeHolder<TFeatureCachedTreeEvaluator>(*Model, floatAccessor, catAccessor, docCount);
    }, 0, BlockParams.GetBlockCount(), NPar::TLocalExecutor::WAIT_COMPLETE);
}
