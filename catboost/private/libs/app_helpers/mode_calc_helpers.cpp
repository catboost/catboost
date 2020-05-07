#include "mode_calc_helpers.h"

#include <catboost/private/libs/algo/apply.h>
#include <catboost/libs/data/proceed_pool_in_blocks.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/helpers/vector_helpers.h>
#include <catboost/libs/eval_result/eval_result.h>
#include <catboost/private/libs/labels/external_label_helper.h>
#include <catboost/libs/logging/logging.h>

#include <library/getopt/small/last_getopt.h>

#include <util/string/cast.h>
#include <util/string/split.h>

#include <util/generic/utility.h>
#include <util/generic/xrange.h>


void NCB::PrepareCalcModeParamsParser(
    NCB::TAnalyticalModeCommonParams* paramsPtr,
    size_t* iterationsLimitPtr,
    size_t* evalPeriodPtr,
    NLastGetopt::TOpts* parserPtr ) {

    auto& params = *paramsPtr;
    auto& iterationsLimit = *iterationsLimitPtr;
    auto& evalPeriod = *evalPeriodPtr;
    auto& parser = *parserPtr;

    parser.AddHelpOption();
    params.BindParserOpts(parser);
    parser.AddLongOption("tree-count-limit", "limit count of used trees")
        .StoreResult(&iterationsLimit);
    parser.AddLongOption("output-columns")
        .RequiredArgument("Comma separated list of column indexes")
        .Handler1T<TString>([&](const TString& outputColumns) {
            params.OutputColumnsIds.clear();
            for (const auto& typeName : StringSplitter(outputColumns).Split(',').SkipEmpty()) {
                params.OutputColumnsIds.push_back(FromString<TString>(typeName.Token()));
            }
        });
    parser.AddLongOption("prediction-type")
        .RequiredArgument(
            "Comma separated list of prediction types. Every prediction type should be one of: Probability, Class, RawFormulaVal")
        .Handler1T<TString>([&](const TString& predictionTypes) {
            params.PredictionTypes.clear();
            params.OutputColumnsIds = {"SampleId"};
            for (const auto& typeName : StringSplitter(predictionTypes).Split(',').SkipEmpty()) {
                params.PredictionTypes.push_back(FromString<EPredictionType>(typeName.Token()));
                params.OutputColumnsIds.push_back(FromString<TString>(typeName.Token()));
            }
        });
    parser.AddLongOption("eval-period", "predictions are evaluated every <eval-period> trees")
        .StoreResult(&evalPeriod);
    parser.SetFreeArgsNum(0);
}

void NCB::ReadModelAndUpdateParams(
    NCB::TAnalyticalModeCommonParams* paramsPtr,
    size_t* iterationsLimitPtr,
    size_t* evalPeriodPtr,
    TFullModel* modelPtr) {

    auto& params = *paramsPtr;
    auto& iterationsLimit = *iterationsLimitPtr;
    auto& evalPeriod = *evalPeriodPtr;
    auto& model = *modelPtr;

    model = ReadModel(params.ModelFileName, params.ModelFormat);
    if (model.HasCategoricalFeatures()) {
        CB_ENSURE(params.DatasetReadingParams.ColumnarPoolFormatParams.CdFilePath.Inited(),
                  "Model has categorical features. Specify column_description file with correct categorical features.");
        CB_ENSURE(model.HasValidCtrProvider(),
                  "Model has invalid ctr provider, possibly you are using core model without or with incomplete ctr data");
    }
    if (model.HasTextFeatures()) {
        CB_ENSURE(params.DatasetReadingParams.ColumnarPoolFormatParams.CdFilePath.Inited(),
                  "Model has text features. Specify column_description file with correct text features.");
        CB_ENSURE(model.HasValidTextProcessingCollection(),
                  "Model has invalid text processing collection, possibly you are using"
                  " core model without or with incomplete estimatedFeatures data");
    }

    params.DatasetReadingParams.ClassLabels = model.GetModelClassLabels();

    if (iterationsLimit == 0) {
        iterationsLimit = model.GetTreeCount();
    }
    iterationsLimit = Min(iterationsLimit, model.GetTreeCount());

    if (evalPeriod == 0) {
        evalPeriod = iterationsLimit;
    } else {
        evalPeriod = Min(evalPeriod, iterationsLimit);
    }
}

static NCB::TEvalResult Apply(
    const TFullModel& model,
    const NCB::TDataProvider& dataset,
    size_t begin, size_t end,
    size_t evalPeriod,
    NPar::TLocalExecutor* executor) {

    NCB::TEvalResult resultApprox;
    TVector<TVector<TVector<double>>>& rawValues = resultApprox.GetRawValuesRef();
    rawValues.resize(1);

    auto maybeBaseline = dataset.RawTargetData.GetBaseline();
    if (maybeBaseline) {
        AssignRank2(*maybeBaseline, &rawValues[0]);
    } else {
        rawValues[0].resize(model.GetDimensionsCount(),
                            TVector<double>(dataset.ObjectsGrouping->GetObjectCount(), 0.0));
    }
    TModelCalcerOnPool modelCalcerOnPool(model, dataset.ObjectsData, executor);
    TVector<double> flatApprox;
    TVector<TVector<double>> approx;
    for (; begin < end; begin += evalPeriod) {
        modelCalcerOnPool.ApplyModelMulti(EPredictionType::InternalRawFormulaVal,
                                          begin,
                                          Min(begin + evalPeriod, end),
                                          &flatApprox,
                                          &approx);

        for (size_t i = 0; i < approx.size(); ++i) {
            for (size_t j = 0; j < approx[0].size(); ++j) {
                rawValues.back()[i][j] += approx[i][j];
            }
        }
        if (begin + evalPeriod < end) {
            rawValues.push_back(rawValues.back());
        }
    }
    return resultApprox;
}

void NCB::CalcModelSingleHost(
    const NCB::TAnalyticalModeCommonParams& params,
    size_t iterationsLimit,
    size_t evalPeriod,
    TFullModel&& model) {

    CB_ENSURE(params.OutputPath.Scheme == "dsv" || params.OutputPath.Scheme == "stream", "Local model evaluation supports only \"dsv\"  and \"stream\" output file schemas.");
    params.DatasetReadingParams.ValidatePoolParams();

    TSetLogging logging(params.OutputPath.Scheme == "dsv" ? ELoggingLevel::Info : ELoggingLevel::Silent);
    THolder<IOutputStream> outputStream;
    if (params.OutputPath.Scheme == "dsv") {
         outputStream = MakeHolder<TOFStream>(params.OutputPath.Path);
    } else {
        CB_ENSURE(params.OutputPath.Path == "stdout" || params.OutputPath.Path == "stderr", "Local model evaluation supports only stderr and stdout paths.");

        if (params.OutputPath.Path == "stdout") {
            outputStream = MakeHolder<TFileOutput>(Duplicate(1));
        } else {
            CB_ENSURE(params.OutputPath.Path == "stderr", "Unknown output stream: " << params.OutputPath.Path);
            outputStream = MakeHolder<TFileOutput>(Duplicate(2));
        }
    }
    NPar::TLocalExecutor executor;
    executor.RunAdditionalThreads(params.ThreadCount - 1);

    bool IsFirstBlock = true;
    ui64 docIdOffset = 0;
    auto poolColumnsPrinter = CreatePoolColumnPrinter(
        params.DatasetReadingParams.PoolPath,
        params.DatasetReadingParams.ColumnarPoolFormatParams.DsvFormat);
    const int blockSize = Max<int>(
        32,
        static_cast<int>(10000. / (static_cast<double>(iterationsLimit) / evalPeriod) / model.GetDimensionsCount())
    );
    ReadAndProceedPoolInBlocks(
        params.DatasetReadingParams,
        blockSize,
        [&](const NCB::TDataProviderPtr datasetPart) {
            if (IsFirstBlock) {
                ValidateColumnOutput(params.OutputColumnsIds, *datasetPart);
            }
            auto approx = Apply(model, *datasetPart, 0, iterationsLimit, evalPeriod, &executor);
            const TExternalLabelsHelper visibleLabelsHelper(model);

            poolColumnsPrinter->UpdateColumnTypeInfo(datasetPart->MetaInfo.ColumnsInfo);

            TSetLoggingSilent inThisScope;
            OutputEvalResultToFile(
                approx,
                &executor,
                params.OutputColumnsIds,
                model.GetLossFunctionName(),
                visibleLabelsHelper,
                *datasetPart,
                outputStream.Get(),
                // TODO: src file columns output is incompatible with block processing
                poolColumnsPrinter,
                /*testFileWhichOf*/ {0, 0},
                IsFirstBlock,
                docIdOffset,
                std::make_pair(evalPeriod, iterationsLimit));
            docIdOffset += datasetPart->ObjectsGrouping->GetObjectCount();
            IsFirstBlock = false;
        },
        &executor);
}

