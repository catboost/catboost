#include "apply.h"
#include "index_calcer.h"
#include "features_data_helpers.h"

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

    // for 1 iteration it will be 7k docs, for 10k iterations it will be 100 docs.
    const int minBlockSize = ceil(10000.0 / sqrt(end - begin + 1));
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
    TLocalExecutor* executor
) {
    const int approxDimension = model.ObliviousTrees.ApproxDimension;

    const auto applyOnBlock = [&](int blockId) {
        const int blockFirstIdx = blockParams.FirstId + blockId * blockParams.GetBlockSize();
        const int blockLastIdx = Min(blockParams.LastId, blockFirstIdx + blockParams.GetBlockSize());
        const int blockSize = blockLastIdx - blockFirstIdx;

        TRawFeatureAccessor featureAccessor(
            model, *rawObjectsData, columnReorderMap, blockFirstIdx, blockLastIdx);

        CalcGeneric(
            model,
            featureAccessor,
            featureAccessor,
            blockSize,
            begin,
            end,
            MakeArrayRef(approxesFlat->data() + blockFirstIdx * approxDimension,blockSize * approxDimension)
        );
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
    TLocalExecutor* executor
) {
    const int approxDimension = model.ObliviousTrees.ApproxDimension;
    const auto applyOnBlock = [&](int blockId) {
        const int blockFirstIdx = blockParams.FirstId + blockId * blockParams.GetBlockSize();
        const int blockLastIdx = Min(blockParams.LastId, blockFirstIdx + blockParams.GetBlockSize());
        const int blockSize = blockLastIdx - blockFirstIdx;

        TQuantizedFeatureAccessor quantizedFeatureAccessor(
            model, quantizedObjectsData, columnReorderMap, blockFirstIdx, blockLastIdx);

        constexpr bool isQuantized = true;
        CalcGeneric<isQuantized>(
            model,
            quantizedFeatureAccessor,
            quantizedFeatureAccessor,
            blockSize,
            begin,
            end,
            MakeArrayRef(approxesFlat->data() + blockFirstIdx * approxDimension, blockSize * approxDimension)
        );
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
        } else if (const auto *const quantizedObjectsData
            = dynamic_cast<const TQuantizedForCPUObjectsDataProvider*>(&objectsData)
        ) {
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

TVector<TVector<double>> ApplyModelMulti(
    const TFullModel& model,
    const TObjectsDataProvider& objectsData,
    bool verbose,
    const EPredictionType predictionType,
    int begin,
    int end,
    int threadCount)
{
    TSetLoggingVerboseOrSilent inThisScope(verbose);

    NPar::TLocalExecutor executor;
    executor.RunAdditionalThreads(threadCount - 1);
    const auto& result = ApplyModelMulti(model, objectsData, predictionType, begin, end, &executor);
    return result;
}

TVector<TVector<double>> ApplyModelMulti(
    const TFullModel& model,
    const TDataProvider& data,
    bool verbose,
    const EPredictionType predictionType,
    int begin,
    int end,
    int threadCount)
{
    auto approxes = ApplyModelMulti(model, *data.ObjectsData, verbose, predictionType, begin, end, threadCount);
    if (const auto& baseline = data.RawTargetData.GetBaseline()) {
        for (size_t i = 0; i < approxes.size(); ++i) {
            for (size_t j = 0; j < approxes[0].size(); ++j) {
                approxes[i][j] += (*baseline)[i][j];
            }
        }
    }
    return approxes;
}

TVector<double> ApplyModel(
    const TFullModel& model,
    const TObjectsDataProvider& objectsData,
    bool verbose,
    const EPredictionType predictionType,
    int begin, /*= 0*/
    int end, /*= 0*/
    int threadCount /*= 1*/)
{
    return ApplyModelMulti(model, objectsData, verbose, predictionType, begin, end, threadCount)[0];
}

TVector<double> ApplyModel(
    const TFullModel& model,
    const TDataProvider& data,
    bool verbose,
    const EPredictionType predictionType,
    int begin, /*= 0*/
    int end, /*= 0*/
    int threadCount /*= 1*/)
{
    return ApplyModelMulti(model, data, verbose, predictionType, begin, end, threadCount)[0];
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

    Executor->ExecRange(
        [&](int blockId) {
            auto& calcer = *ThreadCalcers[blockId];
            const int blockFirstId = BlockParams.FirstId + blockId * BlockParams.GetBlockSize();
            const int blockLastId = Min(BlockParams.LastId, blockFirstId + BlockParams.GetBlockSize());
            TArrayRef<double> resultRef(
                approxFlat.data() + blockFirstId * approxDimension,
                (blockLastId - blockFirstId) * approxDimension);
            calcer.Calc(begin, end, resultRef);
        },
        0,
        BlockParams.GetBlockCount(),
        NPar::TLocalExecutor::WAIT_COMPLETE);

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
    executor->ExecRange(
        [&](int blockId) {
            const int blockFirstIdx = blockParams.FirstId + blockId * blockParams.GetBlockSize();
            const int blockLastIdx = Min(blockParams.LastId, blockFirstIdx + blockParams.GetBlockSize());
            TRawFeatureAccessor featureAccessor(
                model, rawObjectsData, columnReorderMap, blockFirstIdx, blockLastIdx);
            ui64 docCount = ui64(blockLastIdx - blockFirstIdx);
            ThreadCalcers[blockId] = MakeHolder<TFeatureCachedTreeEvaluator>(
                model,
                featureAccessor,
                featureAccessor,
                docCount);
        },
        0,
        blockParams.GetBlockCount(),
        NPar::TLocalExecutor::WAIT_COMPLETE
    );
}

void TModelCalcerOnPool::InitForQuantizedFeatures(
    const TFullModel& model,
    const TQuantizedForCPUObjectsDataProvider& quantizedObjectsData,
    const THashMap<ui32, ui32>& columnReorderMap,
    const NPar::TLocalExecutor::TExecRangeParams& blockParams,
    NPar::TLocalExecutor* executor)
{
    executor->ExecRange(
        [&](int blockId) {
            const int blockFirstIdx = blockParams.FirstId + blockId * blockParams.GetBlockSize();
            const int blockLastIdx = Min(blockParams.LastId, blockFirstIdx + blockParams.GetBlockSize());

            ui64 docCount = ui64(blockLastIdx - blockFirstIdx);
            ThreadCalcers[blockId] = MakeHolder<TFeatureCachedTreeEvaluator>(
                model,
                TQuantizedFeatureAccessor(
                    model, quantizedObjectsData, columnReorderMap, blockFirstIdx, blockLastIdx),
                docCount
            );
        },
        0,
        blockParams.GetBlockCount(),
        NPar::TLocalExecutor::WAIT_COMPLETE);
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

TLeafIndexCalcerOnPool::TLeafIndexCalcerOnPool(
    const TFullModel& model,
    NCB::TObjectsDataProviderPtr objectsData,
    int treeStart,
    int treeEnd)
{
    CB_ENSURE(treeStart >= 0);
    CB_ENSURE(treeEnd >= 0);
    THashMap<ui32, ui32> columnReorderMap;
    CheckModelAndDatasetCompatibility(model, *objectsData, &columnReorderMap);
    const size_t docCount = objectsData->GetObjectCount();
    if (const auto *const rawObjectsData = dynamic_cast<const TRawObjectsDataProvider*>(objectsData.Get())) {
        TRawFeatureAccessor featureAccessor(
            model, *rawObjectsData, columnReorderMap, 0, SafeIntegerCast<int>(docCount));
        InnerLeafIndexCalcer = MakeLeafIndexCalcer(
            model, featureAccessor, featureAccessor, docCount, treeStart, treeEnd);
    } else if (
        const auto *const quantizedObjectsData =
            dynamic_cast<const TQuantizedForCPUObjectsDataProvider*>(objectsData.Get()))
    {
        TQuantizedFeatureAccessor quantizedFeatureAccessor(
            model, *quantizedObjectsData, columnReorderMap, 0, SafeIntegerCast<int>(docCount));
        InnerLeafIndexCalcer = MakeLeafIndexCalcer(
                model, quantizedFeatureAccessor, docCount, treeStart, treeEnd);
    } else {
        ythrow TCatBoostException() << "Unsupported objects data - neither raw nor quantized for CPU";
    }
}

bool TLeafIndexCalcerOnPool::CanGet() const {
    return InnerLeafIndexCalcer->CanGet();
}

TVector<TCalcerIndexType> TLeafIndexCalcerOnPool::Get() const {
    return InnerLeafIndexCalcer->Get();
}

bool TLeafIndexCalcerOnPool::Next() {
    return InnerLeafIndexCalcer->Next();
}

template <bool IsQuantizedData, class TDataProvider, class TFeatureAccessorType =
    typename std::conditional<IsQuantizedData, TQuantizedFeatureAccessor, TRawFeatureAccessor>::type>
static void CalcLeafIndexesMultiImpll(
    const TFullModel& model,
    const TDataProvider& objectsData,
    int treeStart,
    int treeEnd,
    NPar::TLocalExecutor* executor, /* = nullptr */
    TArrayRef<ui32> leafIndexes)
{
    THashMap<ui32, ui32> columnReorderMap;
    CheckModelAndDatasetCompatibility(model, objectsData, &columnReorderMap);

    const int treeCount = treeEnd - treeStart;
    const int docCount = SafeIntegerCast<int>(objectsData.GetObjectCount());
    const int executorThreadCount = executor ? executor->GetThreadCount() : 0;
    auto blockParams = GetBlockParams(executorThreadCount, docCount, treeStart, treeEnd);

    const auto applyOnBlock = [&](int blockId) {
        const int blockFirstIdx = blockParams.FirstId + blockId * blockParams.GetBlockSize();
        const int blockLastIdx = Min(blockParams.LastId, blockFirstIdx + blockParams.GetBlockSize());
        const int blockSize = blockLastIdx - blockFirstIdx;
        TFeatureAccessorType featureAccessor(
            model, objectsData, columnReorderMap, blockFirstIdx, blockLastIdx);
        CalcLeafIndexesGeneric<IsQuantizedData>(
            model,
            featureAccessor,
            featureAccessor,
            blockSize,
            treeStart,
            treeEnd,
            MakeArrayRef(leafIndexes.data() + blockFirstIdx * treeCount, blockSize * treeCount)
        );
    };

    if (executor) {
        executor->ExecRange(applyOnBlock, 0, blockParams.GetBlockCount(), TLocalExecutor::WAIT_COMPLETE);
    } else {
        applyOnBlock(0);
    }
}

TVector<ui32> CalcLeafIndexesMulti(
    const TFullModel& model,
    NCB::TObjectsDataProviderPtr objectsData,
    int treeStart,
    int treeEnd,
    NPar::TLocalExecutor* executor /* = nullptr */)
{
    CB_ENSURE(treeStart >= 0);
    CB_ENSURE(treeEnd >= 0);
    CB_ENSURE(treeEnd >= treeStart);
    const int totalTreeCount = SafeIntegerCast<int>(model.GetTreeCount());
    treeEnd = treeEnd == 0 ? totalTreeCount : Min<int>(treeEnd, totalTreeCount);
    const size_t objCount = objectsData->GetObjectCount();
    TVector<ui32> result(objCount * (treeEnd - treeStart), 0);

    if (objCount > 0) {
        if (const auto* const rawObjectsData =
            dynamic_cast<const TRawObjectsDataProvider*>(objectsData.Get()))
        {
            CalcLeafIndexesMultiImpll<false>(model, *rawObjectsData, treeStart, treeEnd, executor, result);
        } else if (const auto* const quantizedObjectsData =
            dynamic_cast<const TQuantizedForCPUObjectsDataProvider*>(objectsData.Get()))
        {
            CalcLeafIndexesMultiImpll<true>(
                model, *quantizedObjectsData, treeStart, treeEnd, executor, result);
        } else {
            ythrow TCatBoostException() << "Unsupported objects data - neither raw nor quantized for CPU";
        }
    }
    return result;
}

TVector<ui32> CalcLeafIndexesMulti(
    const TFullModel& model,
    NCB::TObjectsDataProviderPtr objectsData,
    bool verbose,
    int treeStart,
    int treeEnd,
    int threadCount)
{
    TSetLoggingVerboseOrSilent inThisScope(verbose);

    CB_ENSURE(threadCount > 0);
    NPar::TLocalExecutor executor;
    executor.RunAdditionalThreads(threadCount - 1);
    return CalcLeafIndexesMulti(model, objectsData, treeStart, treeEnd, &executor);
}
