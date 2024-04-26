#include "apply.h"
#include "index_calcer.h"
#include "features_data_helpers.h"

#include <catboost/libs/data/data_provider.h>
#include <catboost/libs/data/model_dataset_compatibility.h>
#include <catboost/libs/eval_result/eval_helpers.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/helpers/vector_helpers.h>
#include <catboost/libs/logging/logging.h>
#include <catboost/libs/model/cpu/evaluator.h>
#include <catboost/libs/model/scale_and_bias.h>
#include <catboost/private/libs/options/enum_helpers.h>

#include <util/generic/array_ref.h>
#include <util/generic/cast.h>
#include <util/generic/utility.h>
#include <util/system/rwlock.h>

#include <cmath>


using namespace NCB;
using NPar::ILocalExecutor;


static ILocalExecutor::TExecRangeParams GetBlockParams(int executorThreadCount, int docCount, int treeCount) {
    const int threadCount = executorThreadCount + 1; // one for current thread

    // for 1 iteration it will be 7k docs, for 10k iterations it will be 100 docs.
    const int minBlockSize = ceil(10000.0 / sqrt(treeCount + 1));
    const int effectiveBlockCount = Min(threadCount, (docCount + minBlockSize - 1) / minBlockSize);

    ILocalExecutor::TExecRangeParams blockParams(0, docCount);
    blockParams.SetBlockCount(effectiveBlockCount);
    return blockParams;
};

static inline void FixupTreeEnd(size_t treeCount_, int treeBegin, int* treeEnd) {
    int treeCount = SafeIntegerCast<int>(treeCount_);
    if (treeBegin == 0 && *treeEnd == 0) *treeEnd = treeCount;
    CB_ENSURE(0 <= treeBegin && treeBegin <= treeCount, "Out of range treeBegin=" << treeBegin);
    CB_ENSURE(0 <= *treeEnd && *treeEnd <= treeCount, "Out of range treeEnd=" << *treeEnd);
    CB_ENSURE(treeBegin < *treeEnd, "Empty tree range [" << treeBegin << ", " << *treeEnd << ")");
}

namespace {
    class IQuantizedBlockVisitor {
    public:
        virtual ~IQuantizedBlockVisitor() = default;

        virtual void Do(
            const NModelEvaluation::IQuantizedData& quantizedBlock,
            ui32 objectBlockStart,
            ui32 objectBlockEnd) = 0;
    };
}


static void BlockedEvaluation(
    const TFullModel& model,
    const TObjectsDataProvider& objectsData,
    ui32 objectBlockStart,
    ui32 objectBlockEnd,
    ui32 subBlockSize,
    IQuantizedBlockVisitor* visitor
) {
    THolder<IFeaturesBlockIterator> featuresBlockIterator
        = CreateFeaturesBlockIterator(model, objectsData, objectBlockStart, objectBlockEnd);

    for (; objectBlockStart < objectBlockEnd; objectBlockStart += subBlockSize) {
        const ui32 currentBlockSize = Min(objectBlockEnd - objectBlockStart, subBlockSize);

        featuresBlockIterator->NextBlock((size_t)currentBlockSize);

        auto quantizedBlock = MakeQuantizedFeaturesForEvaluator(
            model,
            *featuresBlockIterator,
            objectBlockStart,
            objectBlockStart + currentBlockSize);

        visitor->Do(*quantizedBlock, objectBlockStart, objectBlockStart + currentBlockSize);
    }
}


namespace {
    class TApplyVisitor final : public IQuantizedBlockVisitor {
    public:
        TApplyVisitor(
            const TFullModel& model,
            int treeBegin,
            int treeEnd,
            ui32 approxesFlatBeginAt,
            TArrayRef<double> approxesFlat)
            : ModelEvaluator(model.GetCurrentEvaluator())
            , ApproxDimension(model.GetDimensionsCount())
            , TreeBegin(treeBegin)
            , TreeEnd(treeEnd)
            , ApproxesFlatBeginAt(approxesFlatBeginAt)
            , ApproxesFlat(approxesFlat)
        {}

        void Do(
            const NModelEvaluation::IQuantizedData& quantizedBlock,
            ui32 objectBlockStart,
            ui32 objectBlockEnd) override
        {
            ModelEvaluator->Calc(
                &quantizedBlock,
                TreeBegin,
                TreeEnd,
                MakeArrayRef(
                    ApproxesFlat.data() + (objectBlockStart - ApproxesFlatBeginAt) * ApproxDimension,
                    (objectBlockEnd - objectBlockStart) * ApproxDimension));
        }

    private:
        NModelEvaluation::TConstModelEvaluatorPtr ModelEvaluator;
        size_t ApproxDimension;
        int TreeBegin;
        int TreeEnd;
        ui32 ApproxesFlatBeginAt;
        TArrayRef<double> ApproxesFlat;
    };
}

static void PrepareObjectsDataProviderForEvaluation(const TObjectsDataProvider& objectsData) {
    if (auto* quantizedObjectsData = dynamic_cast<const TQuantizedObjectsDataProvider*>(&objectsData)) {
        auto quantizedFeaturesInfo = quantizedObjectsData->GetQuantizedFeaturesInfo();
        TWriteGuard guard(quantizedFeaturesInfo->GetRWMutex());
        quantizedFeaturesInfo->LoadCatFeaturePerfectHashToRam();
    }
}


TVector<TVector<double>> ApplyModelMulti(
    const TFullModel& model,
    const TObjectsDataProvider& objectsData,
    const EPredictionType predictionType,
    int begin, /*= 0*/
    int end,   /*= 0*/
    ILocalExecutor* executor,
    const NCB::TMaybeData<TConstArrayRef<TConstArrayRef<float>>>& baseline)
{
    if (baseline) {
        // TODO: only one dimension of baseline is checked, what about others?
        CB_ENSURE(
            baseline->size() == model.GetDimensionsCount(),
            "Baseline should have the same dimension count as model: expected " << model.GetDimensionsCount()
                                                                                << " got " << baseline->size()
        );
    }

    const int docCount = SafeIntegerCast<int>(objectsData.GetObjectCount());
    const int approxesDimension = model.GetDimensionsCount();
    TVector<double> approxesFlat(docCount * approxesDimension);
    if (docCount > 0) {
        FixupTreeEnd(model.GetTreeCount(), begin, &end);
        PrepareObjectsDataProviderForEvaluation(objectsData);

        const int executorThreadCount = executor ? executor->GetThreadCount() : 0;
        auto blockParams = GetBlockParams(executorThreadCount, docCount, end - begin);

        TApplyVisitor visitor(model, begin, end, 0, approxesFlat);

        const ui32 subBlockSize = ui32(NModelEvaluation::FORMULA_EVALUATION_BLOCK_SIZE * 64);

        const auto applyOnBlock = [&](int blockId) {
            const int blockFirstIdx = blockParams.FirstId + blockId * blockParams.GetBlockSize();
            const int blockLastIdx = Min(blockParams.LastId, blockFirstIdx + blockParams.GetBlockSize());

            BlockedEvaluation(model, objectsData, (ui32)blockFirstIdx, (ui32)blockLastIdx, subBlockSize, &visitor);
        };
        if (executor) {
            executor->ExecRangeWithThrow(applyOnBlock, 0, blockParams.GetBlockCount(), ILocalExecutor::WAIT_COMPLETE);
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
    if (baseline) {
        for (int i = 0; i < approxesDimension; ++i) {
            for (int j = 0; j < docCount; ++j) {
                approxes[i][j] += (*baseline)[i][j];
            }
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
    int threadCount,
    const NCB::TMaybeData<TConstArrayRef<TConstArrayRef<float>>>& baseline)
{
    TSetLoggingVerboseOrSilent inThisScope(verbose);

    int lastTreeIdx = end;
    FixupTreeEnd(model.GetTreeCount(), begin, &lastTreeIdx);
    const auto blockParams = GetBlockParams(threadCount, objectsData.GetObjectCount(), lastTreeIdx - begin);

    NPar::TLocalExecutor executor;
    // ToDo: fix prediction on GPU for several threads
    if (model.GetEvaluatorType() == EFormulaEvaluatorType::CPU) {
        executor.RunAdditionalThreads(Min<int>(threadCount, blockParams.GetBlockCount()) - 1);
    }
    const auto& result = ApplyModelMulti(model, objectsData, predictionType, begin, end, &executor, baseline);
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
    auto approxes = ApplyModelMulti(model, *data.ObjectsData, verbose, predictionType, begin, end, threadCount, data.RawTargetData.GetBaseline());
    return approxes;
}

TMinMax<double> ApplyModelForMinMax(
    const TFullModel& model,
    const NCB::TObjectsDataProvider& objectsData,
    int treeBegin,
    int treeEnd,
    NPar::ILocalExecutor* executor)
{
    CB_ENSURE(model.GetTreeCount(), "Bad usage: empty model");
    CB_ENSURE(model.GetDimensionsCount() == 1, "Bad usage: multiclass/multitarget model, dim=" << model.GetDimensionsCount());
    FixupTreeEnd(model.GetTreeCount(), treeBegin, &treeEnd);
    CB_ENSURE(objectsData.GetObjectCount(), "Bad usage: empty dataset");

    const ui32 docCount = objectsData.GetObjectCount();
    TMinMax<double> result{+DBL_MAX, -DBL_MAX};
    TMutex result_guard;

    if (docCount > 0) {
        PrepareObjectsDataProviderForEvaluation(objectsData);

        const int executorThreadCount = executor ? executor->GetThreadCount() : 0;
        auto blockParams = GetBlockParams(executorThreadCount, docCount, treeEnd - treeBegin);

        const ui32 subBlockSize = ui32(NModelEvaluation::FORMULA_EVALUATION_BLOCK_SIZE * 64);

        const auto applyOnBlock = [&](int blockId) {
            const int blockFirstIdx = blockParams.FirstId + blockId * blockParams.GetBlockSize();
            const int blockLastIdx = Min(blockParams.LastId, blockFirstIdx + blockParams.GetBlockSize());

            TVector<double> blockResult;
            blockResult.yresize(blockLastIdx - blockFirstIdx);
            TApplyVisitor visitor(model, treeBegin, treeEnd, blockFirstIdx, blockResult);

            BlockedEvaluation(model, objectsData, (ui32)blockFirstIdx, (ui32)blockLastIdx, subBlockSize, &visitor);

            auto minMax = CalcMinMax(blockResult.begin(), blockResult.end());
            GuardedUpdateMinMax(minMax, &result, result_guard);
        };
        if (executor) {
            executor->ExecRangeWithThrow(applyOnBlock, 0, blockParams.GetBlockCount(), NPar::TLocalExecutor::WAIT_COMPLETE);
        } else {
            applyOnBlock(0);
        }
    }
    return result;
}

void TModelCalcerOnPool::ApplyModelMulti(
    const EPredictionType predictionType,
    int begin,
    int end,
    TVector<double>* flatApproxBuffer,
    TVector<TVector<double>>* approx)
{
    const ui32 docCount = ObjectsData->GetObjectCount();
    auto approxDimension = Model->GetDimensionsCount();
    TVector<double>& approxFlat = *flatApproxBuffer;
    approxFlat.yresize(static_cast<unsigned long>(docCount * approxDimension));

    FixupTreeEnd(Model->GetTreeCount(), begin, &end);

    Executor->ExecRangeWithThrow(
        [&, this](int blockId) {
            const int blockFirstId = BlockParams.FirstId + blockId * BlockParams.GetBlockSize();
            const int blockLastId = Min(BlockParams.LastId, blockFirstId + BlockParams.GetBlockSize());
            TArrayRef<double> resultRef(
                approxFlat.data() + blockFirstId * approxDimension,
                (blockLastId - blockFirstId) * approxDimension);
            ModelEvaluator->Calc(QuantizedDataForThreads[blockId].Get(), begin, end, resultRef);
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

TModelCalcerOnPool::TModelCalcerOnPool(
    const TFullModel& model,
    TObjectsDataProviderPtr objectsData,
    NPar::ILocalExecutor* executor)
    : Model(&model)
    , ModelEvaluator(model.GetCurrentEvaluator())
    , ObjectsData(objectsData)
    , Executor(executor)
    , BlockParams(0, SafeIntegerCast<int>(objectsData->GetObjectCount()))
{
    if (BlockParams.FirstId == BlockParams.LastId) {
        return;
    }
    PrepareObjectsDataProviderForEvaluation(*objectsData);

    const int threadCount = executor->GetThreadCount() + 1; // one for current thread
    BlockParams.SetBlockCount(threadCount);
    QuantizedDataForThreads.resize(BlockParams.GetBlockCount());

    executor->ExecRangeWithThrow(
        [this, objectsData](int blockId) {
            const int blockFirstIdx = BlockParams.FirstId + blockId * BlockParams.GetBlockSize();
            const int blockLastIdx = Min(BlockParams.LastId, blockFirstIdx + BlockParams.GetBlockSize());
            QuantizedDataForThreads[blockId] = MakeQuantizedFeaturesForEvaluator(*Model, *objectsData, blockFirstIdx, blockLastIdx);
        },
        0,
        BlockParams.GetBlockCount(),
        NPar::TLocalExecutor::WAIT_COMPLETE
    );
}

TLeafIndexCalcerOnPool::TLeafIndexCalcerOnPool(
    const TFullModel& model,
    NCB::TObjectsDataProviderPtr objectsData,
    int treeStart,
    int treeEnd)
    : Model(model)
    , ModelEvaluator(model.GetCurrentEvaluator())
    , FeaturesBlockIterator(CreateFeaturesBlockIterator(model, *objectsData, 0, objectsData->GetObjectCount()))
    , DocCount(objectsData->GetObjectCount())
    , TreeStart(treeStart)
    , TreeEnd(treeEnd)
    , CurrBatchStart(0)
    , CurrBatchSize(Min(DocCount, NModelEvaluation::FORMULA_EVALUATION_BLOCK_SIZE))
    , CurrDocIndex(0)
{
    PrepareObjectsDataProviderForEvaluation(*objectsData);
    FixupTreeEnd(model.GetTreeCount(), treeStart, &treeEnd);
    CB_ENSURE(TreeEnd == size_t(treeEnd));

    CurrentBatchLeafIndexes.yresize(CurrBatchSize * (TreeEnd - TreeStart));

    CalcNextBatch();
}

bool TLeafIndexCalcerOnPool::Next() {
    ++CurrDocIndex;
    if (CurrDocIndex < DocCount) {
        if (CurrDocIndex == CurrBatchStart + CurrBatchSize) {
            CurrBatchStart += CurrBatchSize;
            CurrBatchSize = Min(DocCount - CurrDocIndex, NModelEvaluation::FORMULA_EVALUATION_BLOCK_SIZE);
            CalcNextBatch();
        }
        return true;
    } else {
        return false;
    }
}

bool TLeafIndexCalcerOnPool::CanGet() const {
    return CurrDocIndex < DocCount;
}

TVector<NModelEvaluation::TCalcerIndexType> TLeafIndexCalcerOnPool::Get() const {
    const auto treeCount = TreeEnd - TreeStart;
    const auto docIndexInBatch = CurrDocIndex - CurrBatchStart;
    return TVector<NModelEvaluation::TCalcerIndexType>(
        CurrentBatchLeafIndexes.begin() + treeCount * docIndexInBatch,
        CurrentBatchLeafIndexes.begin() + treeCount * (docIndexInBatch + 1)
    );
}

void TLeafIndexCalcerOnPool::CalcNextBatch() {
    FeaturesBlockIterator->NextBlock(CurrBatchSize);

    auto quantizedBlock = MakeQuantizedFeaturesForEvaluator(
        Model,
        *FeaturesBlockIterator,
        CurrBatchStart,
        CurrBatchStart + CurrBatchSize);

    ModelEvaluator->CalcLeafIndexes(quantizedBlock.Get(), TreeStart, TreeEnd, CurrentBatchLeafIndexes);
}


namespace {
    class TLeafCalcerVisitor final : public IQuantizedBlockVisitor {
    public:
        TLeafCalcerVisitor(
            const TFullModel& model,
            int treeBegin,
            int treeEnd,
            TArrayRef<NModelEvaluation::TCalcerIndexType> leafIndices)
            : ModelEvaluator(model.GetCurrentEvaluator())
            , TreeBegin(treeBegin)
            , TreeEnd(treeEnd)
            , LeafIndices(leafIndices)
        {}

        void Do(
            const NModelEvaluation::IQuantizedData& quantizedBlock,
            ui32 objectBlockStart,
            ui32 objectBlockEnd) override
        {
            const int treeCount = TreeEnd - TreeBegin;
            const ui32 objectBlockSize = objectBlockEnd - objectBlockStart;
            const ui32 indexBlockSize = objectBlockSize * (ui32)treeCount;

            ModelEvaluator->CalcLeafIndexes(&quantizedBlock, TreeBegin, TreeEnd, MakeArrayRef(LeafIndices.data() + objectBlockStart * treeCount, indexBlockSize));
        }

    private:
        NModelEvaluation::TConstModelEvaluatorPtr ModelEvaluator;
        int TreeBegin;
        int TreeEnd;
        TArrayRef<NModelEvaluation::TCalcerIndexType> LeafIndices;
    };
}


TVector<ui32> CalcLeafIndexesMulti(
    const TFullModel& model,
    NCB::TObjectsDataProviderPtr objectsData,
    int treeStart,
    int treeEnd,
    NPar::ILocalExecutor* executor /* = nullptr */)
{
    FixupTreeEnd(model.GetTreeCount(), treeStart, &treeEnd);
    PrepareObjectsDataProviderForEvaluation(*objectsData);

    const size_t objCount = objectsData->GetObjectCount();
    TVector<ui32> result(objCount * (treeEnd - treeStart), 0);

    if (objCount > 0) {
        const int executorThreadCount = executor ? executor->GetThreadCount() : 0;
        auto blockParams = GetBlockParams(executorThreadCount, objCount, treeEnd - treeStart);

        const ui32 subBlockSize = ui32(NModelEvaluation::FORMULA_EVALUATION_BLOCK_SIZE * 64);

        const auto applyOnBlock = [&](int blockId) {
            const int blockFirstIdx = blockParams.FirstId + blockId * blockParams.GetBlockSize();
            const int blockLastIdx = Min(blockParams.LastId, blockFirstIdx + blockParams.GetBlockSize());

            TLeafCalcerVisitor visitor(model, treeStart, treeEnd, result);

            BlockedEvaluation(model, *objectsData, (ui32)blockFirstIdx, (ui32)blockLastIdx, subBlockSize, &visitor);
        };
        if (executor) {
            executor->ExecRangeWithThrow(applyOnBlock, 0, blockParams.GetBlockCount(), ILocalExecutor::WAIT_COMPLETE);
        } else {
            applyOnBlock(0);
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

void ApplyVirtualEnsembles(
    const TFullModel& model,
    const NCB::TDataProvider& dataset,
    size_t end,
    size_t virtualEnsemblesCount,
    TVector<TVector<double>>* rawValuesPtr,
    NPar::ILocalExecutor* executor
) {
    auto& rawValues = *rawValuesPtr;
    TModelCalcerOnPool modelCalcerOnPool(model, dataset.ObjectsData, executor);
    TVector<double> flatApprox;
    TVector<TVector<double>> approx;
    TVector<TVector<double>> baseApprox;
    const auto approxDimension = model.GetDimensionsCount();
    const size_t evalPeriod = end / (2 * virtualEnsemblesCount);
    CB_ENSURE(evalPeriod > 0 && evalPeriod * virtualEnsemblesCount < end,
              "Not enough trees in model for " << virtualEnsemblesCount << " virtual Ensembles");
    size_t begin = end - evalPeriod * virtualEnsemblesCount;
    modelCalcerOnPool.ApplyModelMulti(
        EPredictionType::InternalRawFormulaVal,
        0,
        begin,
        &flatApprox,
        &baseApprox);
    const auto objectCount = baseApprox[0].size();
    Y_ASSERT(rawValues.empty());
    // init first virtual ensemble predictions by prediction on trees [0; end / 2)
    rawValues.insert(rawValues.end(), baseApprox.begin(), baseApprox.end());
    rawValues.resize(virtualEnsemblesCount * approxDimension);
    for (auto i : xrange(approxDimension, approxDimension * virtualEnsemblesCount)) {
        rawValues[i].resize(objectCount);
    }

    const float actualShrinkCoef = model.GetActualShrinkCoef();
    CB_ENSURE(actualShrinkCoef >= 0.0f && actualShrinkCoef < 1.0f,
              "For Constant shrink mode: (model_shrink_rate * learning_rate) should be in [0, 1).");

    auto copyerLambda = [approxDimension, objectCount] (
            const auto copyToNextEnsemble,
            const TVector<TVector<double>>& approx,
            const float unshrinkCoef,
            TVector<TVector<double>>& rawValues,
            size_t vEnsembleIdx
        ) {
        const size_t shift = vEnsembleIdx * approxDimension;
        for (size_t i = 0; i < approxDimension; ++i) {
            const auto srcPtr = approx[i].data();
            auto dstPtr = rawValues[shift + i].data();
            auto nextDstPtr = rawValues[Min(shift + approxDimension + i, rawValues.size() - 1)].data(); // failsafe ptr !
            for (size_t j = 0; j < objectCount; ++j) {
                dstPtr[j] += srcPtr[j];
                if constexpr (copyToNextEnsemble) {
                    nextDstPtr[j] = dstPtr[j];
                }
                dstPtr[j] *= unshrinkCoef;
            }
        }
    };

    for (size_t vEnsembleIdx = 0; begin < end; begin += evalPeriod, vEnsembleIdx++) {
        const auto lastTreeIdx = Min(begin + evalPeriod, end);
        modelCalcerOnPool.ApplyModelMulti(
            EPredictionType::InternalRawFormulaVal,
            begin,
            lastTreeIdx,
            &flatApprox,
            &approx);
        const float unshrinkCoef = pow(1. - actualShrinkCoef, float(lastTreeIdx) - float(end));
        if (vEnsembleIdx != virtualEnsemblesCount - 1) {
            copyerLambda(std::true_type(), approx, unshrinkCoef, rawValues, vEnsembleIdx);
        } else {
            copyerLambda(std::false_type(), approx, 1.0f, rawValues, vEnsembleIdx);
        }
    }
}

TVector<TVector<double>> ApplyUncertaintyPredictions(
    const TFullModel& model,
    const NCB::TDataProvider& data,
    bool verbose,
    const EPredictionType predictionType,
    int end,
    int virtualEnsemblesCount,
    int threadCount)
{
    TSetLoggingVerboseOrSilent inThisScope(verbose);

    CB_ENSURE_INTERNAL(IsUncertaintyPredictionType(predictionType), "Unsupported prediction type " << predictionType);

    int lastTreeIdx = end;
    FixupTreeEnd(model.GetTreeCount(), 0, &lastTreeIdx);
    TVector<TVector<double>> approxes;

    NPar::TLocalExecutor executor;
    executor.RunAdditionalThreads(threadCount - 1);
    ApplyVirtualEnsembles(
        model,
        data,
        lastTreeIdx,
        virtualEnsemblesCount,
        &approxes,
        &executor);
    return PrepareEval(predictionType, virtualEnsemblesCount, model.GetLossFunctionName(), approxes,  &executor);
}
