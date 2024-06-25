#include "mode_calc_helpers.h"

#include <catboost/private/libs/algo/apply.h>
#include <catboost/libs/data/proceed_pool_in_blocks.h>
#include <catboost/libs/eval_result/eval_result.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/helpers/vector_helpers.h>
#include <catboost/private/libs/labels/external_label_helper.h>
#include <catboost/libs/logging/logging.h>

#include <library/cpp/getopt/small/last_getopt.h>
#include <library/cpp/expression/expression.h>

#include <util/string/cast.h>
#include <util/string/split.h>

#include <util/generic/utility.h>
#include <util/generic/xrange.h>

constexpr int MaxApproxCount = 1000;

static TString GetApproxKey(int i) {
    static const TString prefix("a");
    CB_ENSURE(i < MaxApproxCount, "Raw formula values index should be less than " << MaxApproxCount);
    return prefix + ToString(i);
}

void NCB::PrepareCalcModeParamsParser(
    NCB::TAnalyticalModeCommonParams* paramsPtr,
    size_t* iterationsLimitPtr,
    size_t* evalPeriodPtr,
    size_t* virtualEnsemblesCountPtr,
    NLastGetopt::TOpts* parserPtr) {

    auto& params = *paramsPtr;
    auto& iterationsLimit = *iterationsLimitPtr;
    auto& evalPeriod = *evalPeriodPtr;
    auto& virtualEnsemblesCount = *virtualEnsemblesCountPtr;
    auto& parser = *parserPtr;

    parser.AddHelpOption();
    params.BindParserOpts(parser);
    parser.AddLongOption("tree-count-limit", "limit count of used trees")
        .StoreResult(&iterationsLimit);
    parser.AddLongOption("output-columns")
        .RequiredArgument("Comma separated list of column indexes")
        .Handler1T<TString>([&](const TString& outputColumns) {
            if (params.ForceSingleModel) {
                params.OutputColumnsIds.back().clear();
            } else {
                params.OutputColumnsIds.push_back({});
            }
            for (const auto& typeName : StringSplitter(outputColumns).Split(',').SkipEmpty()) {
                params.OutputColumnsIds.back().push_back(TString(typeName.Token()));
            }
        });
    parser.AddLongOption("prediction-type")
        .RequiredArgument(
            "Comma separated list of prediction types. Every prediction type should be one of: "
            + GetEnumAllNames<EPredictionType>())
        .Handler1T<TString>([&](const TString& predictionTypes) {
            params.ForceSingleModel = true;
            CB_ENSURE(params.OutputColumnsIds.size() <= 1, "prediction-type can be used with at most one output-columns list");
            if (params.OutputColumnsIds.empty()) {
                params.OutputColumnsIds.push_back({});
            }
            params.OutputColumnsIds.back().clear();
            params.OutputColumnsIds.back().push_back(TString("SampleId"));
            for (const auto& typeName : StringSplitter(predictionTypes).Split(',').SkipEmpty()) {
                params.OutputColumnsIds.back().push_back(TString(typeName.Token()));
            }
        });
    parser.AddLongOption("virtual-ensembles-count", "count of virtual ensembles for VirtEnsembles and TotalUncertainty predictions")
        .DefaultValue(10)
        .StoreResult(&virtualEnsemblesCount);
    parser.AddLongOption("eval-period", "predictions are evaluated every <eval-period> trees")
        .StoreResult(&evalPeriod);
    parser.AddLongOption("binary-classification-threshold", "probability threshold for binary classification")
        .Handler1T<TString>([&](const TString& threshold) {
            double probabilityThreshold = FromString<double>(threshold);
            CB_ENSURE(probabilityThreshold > 0. && probabilityThreshold < 1., "Probability threshold should be between 0 and 1");
            params.BinClassLogitThreshold = NCB::Logit(probabilityThreshold);
        });

    const TString blendingHelp = TString("expression to blend raw formula values, e.g. ")
        + "1.5 / " + GetApproxKey(0) + " + 0.5 / " + GetApproxKey(1) + ", etc.";
    parser.AddLongOption("blending-expression", blendingHelp)
        .Handler1T<TString>([&](const TString& expression) {
            params.BlendingExpression = expression;
            THashMap<TString, double> approx;
            for (int i = 0; i < MaxApproxCount; ++i) {
                approx[GetApproxKey(i)] = 0.0;
            }
            try {
                CalcExpression(expression, approx);
            } catch (...) {
                CB_ENSURE(false, "Cannot parse blending expression, or prediction column index exceeds " << MaxApproxCount);
            }
        });
    parser.SetFreeArgsNum(0);
}

void NCB::ReadModelAndUpdateParams(
    NCB::TAnalyticalModeCommonParams* paramsPtr,
    size_t* iterationsLimitPtr,
    size_t* evalPeriodPtr,
    TVector<TFullModel>* allModelsPtr) {

    auto& params = *paramsPtr;
    auto& iterationsLimit = *iterationsLimitPtr;
    auto& evalPeriod = *evalPeriodPtr;
    auto& allModels = *allModelsPtr;

    CB_ENSURE(
        params.OutputColumnsIds.size() <= params.ModelFileName.size(),
        "Number of output columns specifications must not exceed number of models"
    );
    const auto defaultOutputCount = params.ModelFileName.size() - params.OutputColumnsIds.size();
    params.OutputColumnsIds.insert(
        params.OutputColumnsIds.end(),
        defaultOutputCount,
        {TString("SampleId"), TString("RawFormulaVal")}
    );

    for (const auto& columnsIds : params.OutputColumnsIds) {
        for (const auto& id : columnsIds) {
            EPredictionType type;
            if (TryFromString<EPredictionType>(id, type) && IsUncertaintyPredictionType(type)) {
                params.IsUncertaintyPrediction = true;
            }
        }
    }
    if (params.IsUncertaintyPrediction) {
        for (const auto& columnsIds : params.OutputColumnsIds) {
            for (const auto& id : columnsIds) {
                EPredictionType type;
                CB_ENSURE(
                    !TryFromString<EPredictionType>(id, type) || IsUncertaintyPredictionType(type),
                    "Predicton type " << type << " is incompatible with Uncertainty Prediction Type");
            }
        }
    }
    CB_ENSURE(!params.IsUncertaintyPrediction || evalPeriod == 0, "Uncertainty prediction requires Eval period 0.");
    for (const auto& modelFile : params.ModelFileName) {
        allModels.push_back(ReadModel(modelFile, params.ModelFormat));
        const auto& model = allModels.back();
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

        if (params.IsUncertaintyPrediction) {
            if (!model.IsPosteriorSamplingModel()) {
                CATBOOST_WARNING_LOG <<  "Uncertainty Prediction asked for model fitted without Posterior Sampling option" << Endl;
            }
        }
    }

    params.DatasetReadingParams.ClassLabels = allModels[0].GetModelClassLabels();
    // requirements to class labels and logit thresholds can be relaxed
    for (const auto& model : allModels) {
        const auto& labels = model.GetModelClassLabels();
        CB_ENSURE(
            labels.empty() || params.DatasetReadingParams.ClassLabels == labels,
            "All models should have same class labels"
        );
    }
    if (!params.BinClassLogitThreshold.Defined()) {
        params.BinClassLogitThreshold = allModels[0].GetBinClassLogitThreshold();
        for (const auto& model : allModels) {
            const auto& threshold = model.GetBinClassLogitThreshold();
            CB_ENSURE(params.BinClassLogitThreshold == threshold, "All models should have same logit thresholds");
        }
    }

    auto minTreeCount = MinElementBy(allModels, [] (const auto& model) -> size_t { return model.GetTreeCount(); })->GetTreeCount();
    if (iterationsLimit == 0) {
        iterationsLimit = minTreeCount;
    }
    iterationsLimit = Min(iterationsLimit, minTreeCount);

    if (evalPeriod == 0) {
        evalPeriod = iterationsLimit;
    } else {
        evalPeriod = Min(evalPeriod, iterationsLimit);
    }
}

NCB::TEvalResult NCB::Apply(
    const TFullModel& model,
    const NCB::TDataProvider& dataset,
    size_t begin,
    size_t end,
    size_t evalPeriod,
    size_t virtualEnsemblesCount,
    bool isUncertaintyPrediction,
    NPar::ILocalExecutor* executor) {

    NCB::TEvalResult resultApprox(virtualEnsemblesCount);
    TVector<TVector<TVector<double>>>& rawValues = resultApprox.GetRawValuesRef();

    auto maybeBaseline = dataset.RawTargetData.GetBaseline();
    rawValues.resize(1);
    if (isUncertaintyPrediction) {
        CB_ENSURE(!maybeBaseline, "Baseline unsupported with uncertainty prediction");
        CB_ENSURE_INTERNAL(begin == 0, "For Uncertainty Prediction application only from first tree is supported");
        ApplyVirtualEnsembles(
            model,
            dataset,
            end,
            virtualEnsemblesCount,
            &(rawValues[0]),
            executor
        );
    } else {
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
            modelCalcerOnPool.ApplyModelMulti(
                EPredictionType::InternalRawFormulaVal,
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
    }
    return resultApprox;
}

TVector<NCB::TEvalResult> NCB::ApplyAllModels(
    const TVector<TFullModel>& allModels,
    const NCB::TDataProvider& dataset,
    size_t begin,
    size_t end,
    size_t evalPeriod,
    size_t virtualEnsemblesCount,
    bool isUncertaintyPrediction,
    NPar::ILocalExecutor* executor
) {
    const auto modelCount = allModels.size();
    TVector<NCB::TEvalResult> allApproxes;
    allApproxes.reserve(modelCount);
    for (const auto& model : allModels) {
        auto approx = Apply(
            model,
            dataset,
            begin,
            end,
            evalPeriod,
            virtualEnsemblesCount,
            isUncertaintyPrediction,
            executor);
        allApproxes.push_back(approx);
    }
    return allApproxes;
}

void NCB::CalcModelSingleHost(
    const NCB::TAnalyticalModeCommonParams& params,
    size_t iterationsLimit,
    size_t evalPeriod,
    size_t virtualEnsemblesCount,
    TVector<TFullModel>&& allModels) {

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
    CB_ENSURE_INTERNAL(params.BinClassLogitThreshold.Defined(), "Logit threshold should be defined");
    NPar::TLocalExecutor executor;
    executor.RunAdditionalThreads(params.ThreadCount - 1);

    bool IsFirstBlock = true;
    ui64 docIdOffset = 0;
    auto poolColumnsPrinter = TIntrusivePtr<NCB::IPoolColumnsPrinter>(
        GetProcessor<NCB::IPoolColumnsPrinter>(
            params.DatasetReadingParams.PoolPath,
            NCB::TPoolColumnsPrinterPullArgs{
                params.DatasetReadingParams.PoolPath,
                params.DatasetReadingParams.ColumnarPoolFormatParams.DsvFormat,
                /*columnsMetaInfo*/ Nothing()
            }
        ).Release()
    );
    size_t dimensionCount = 0;
    for (const auto& model : allModels) {
        dimensionCount += model.GetDimensionsCount();
    }
    const int blockSize = Max<int>(
        32,
        static_cast<int>(10000. / (static_cast<double>(iterationsLimit) / evalPeriod) / dimensionCount)
    );
    ReadAndProceedPoolInBlocks(
        params.DatasetReadingParams,
        blockSize,
        [&](const NCB::TDataProviderPtr datasetPart) {
            if (IsFirstBlock) {
                ValidateColumnOutput(params.OutputColumnsIds, *datasetPart);
            }
            auto evalColumnsInfo = CreateEvalColumnsInfo(
                allModels,
                datasetPart,
                iterationsLimit,
                evalPeriod,
                virtualEnsemblesCount,
                params.IsUncertaintyPrediction,
                &executor);

            poolColumnsPrinter->UpdateColumnTypeInfo(datasetPart->MetaInfo.ColumnsInfo);

            auto outputColumnsIds = params.OutputColumnsIds;
            if (params.BlendingExpression.size() > 0) {
                AddBlendedApprox(params.BlendingExpression, &evalColumnsInfo);
                outputColumnsIds.push_back({TString("RawFormulaVal")}); // result of blending
            }

            TSetLoggingSilent inThisScope;
            OutputEvalResultToFile(
                evalColumnsInfo,
                &executor,
                outputColumnsIds,
                *datasetPart,
                outputStream.Get(),
                // TODO: src file columns output is incompatible with block processing
                poolColumnsPrinter,
                /*testFileWhichOf*/ {0, 0},
                IsFirstBlock,
                docIdOffset,
                std::make_pair(evalPeriod, iterationsLimit),
                *params.BinClassLogitThreshold);
            docIdOffset += datasetPart->ObjectsGrouping->GetObjectCount();
            IsFirstBlock = false;
        },
        &executor);

}

NCB::TEvalColumnsInfo NCB::CreateEvalColumnsInfo(
    const TVector<TFullModel>& allModels,
    const TDataProviderPtr datasetPart,
    ui32 iterationsLimit,
    ui32 evalPeriod,
    ui32 virtualEnsemblesCount,
    bool isUncertaintyPrediction,
    NPar::TLocalExecutor* localExecutor
) {
    const auto allApproxes = ApplyAllModels(
        allModels,
        *datasetPart,
        0,
        iterationsLimit,
        evalPeriod,
        virtualEnsemblesCount,
        isUncertaintyPrediction,
        localExecutor);

    const auto modelCount = allModels.size();

    TVector<TExternalLabelsHelper> allLabelHelpers;
    allLabelHelpers.reserve(modelCount);
    TVector<TString> allLossFunctions;
    allLossFunctions.reserve(modelCount);
    for (const auto& model : allModels) {
        allLabelHelpers.push_back(TExternalLabelsHelper(model));
        allLossFunctions.push_back(model.GetLossFunctionName());
    }
    return NCB::TEvalColumnsInfo{allApproxes, allLabelHelpers, allLossFunctions};
}

void NCB::AddBlendedApprox(
    const TString& blendingExpression,
    NCB::TEvalColumnsInfo* evalColumnsInfo
) {
    const auto& frontApproxes = evalColumnsInfo->Approxes[0];
    const auto ensemblesCount = frontApproxes.GetEnsemblesCount(); // same for all models
    NCB::TEvalResult blendedApprox(ensemblesCount);

    using TApprox1D = TVector<double>;
    TVector<TVector<TApprox1D>>& blendedApproxRef = blendedApprox.GetRawValuesRef();

    const auto rawValuesRef = frontApproxes.GetRawValuesConstRef(); // [evalIter][dim][docIdx]
    const auto evalIterCount = rawValuesRef.size(); // same for all models
    const auto docCount = rawValuesRef[0][0].size(); // same for all models
    blendedApproxRef = TVector<TVector<TApprox1D>>(evalIterCount, TVector<TApprox1D>(1, TApprox1D(docCount)));

    THashMap<TString, double> approx;
    for (auto evalIter : xrange(evalIterCount)) {
        for (auto docIdx : xrange(docCount)) {
            int approxIdx = 0;
            for (const auto& modelApprox : evalColumnsInfo->Approxes) {
                for (const auto& approx1D : modelApprox.GetRawValuesConstRef()[evalIter]) {
                    approx[GetApproxKey(approxIdx)] = approx1D[docIdx];
                    approxIdx += 1;
                }
            }
            blendedApproxRef[evalIter][0][docIdx] = CalcExpression(blendingExpression, approx);
        }
    }
    evalColumnsInfo->Approxes.push_back(blendedApprox);
    evalColumnsInfo->LabelHelpers.push_back({}); // will not be used
    evalColumnsInfo->LossFunctions.push_back({}); // will not be used
}
