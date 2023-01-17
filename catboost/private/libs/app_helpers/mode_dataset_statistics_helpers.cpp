#include "mode_dataset_statistics_helpers.h"

#include <library/cpp/getopt/small/last_getopt.h>

#include <util/generic/ptr.h>

using namespace NCB;

void TCalculateStatisticsParams::BindParserOpts(NLastGetopt::TOpts& parser) {
    DatasetReadingParams.BindParserOpts(&parser);
    parser.AddLongOption('o', "output-path", "output result path")
        .StoreResult(&OutputPath)
        .DefaultValue("statistics.json");
    parser.AddLongOption('T', "thread-count", "worker thread count (default: core count)")
        .StoreResult(&ThreadCount);
    parser.AddLongOption("only-group-statistics")
        .OptionalValue("false", "bool")
        .Handler1T<TString>([&](const TString& param) {
            OnlyGroupStatistics = FromString<bool>(param);
        });
}

void TCalculateStatisticsParams::ProcessParams(int argc, const char* argv[], NLastGetopt::TOpts* parserPtr) {
    TCalculateStatisticsParams& params = *this;

    params.DatasetReadingParams.ValidatePoolParams();

    auto parser = parserPtr ? *parserPtr : NLastGetopt::TOpts();
    parser.AddHelpOption();
    params.BindParserOpts(parser);
    parser.SetFreeArgsNum(0);
    NLastGetopt::TOptsParseResult parserResult{&parser, argc, argv};

    params.DatasetReadingParams.ValidatePoolParams();
}

void NCB::CalculateDatasetStaticsSingleHost(const TCalculateStatisticsParams& calculateStatisticsParams) {
    NPar::TLocalExecutor localExecutor;
    localExecutor.RunAdditionalThreads(calculateStatisticsParams.ThreadCount - 1);

    const int blockSize = Max<int>(
        10000, 10000 // ToDo: meaningful estimation
    );

    const NCatboostOptions::TDatasetReadingParams& params = calculateStatisticsParams.DatasetReadingParams;

    auto datasetLoader = NCB::GetProcessor<NCB::IDatasetLoader>(
        params.PoolPath, // for choosing processor

        // processor args
        NCB::TDatasetLoaderPullArgs{
            params.PoolPath,

            NCB::TDatasetLoaderCommonArgs{
                params.PairsFilePath,
                /*GroupWeightsFilePath=*/NCB::TPathWithScheme(),
                /*BaselineFilePath=*/NCB::TPathWithScheme(),
                /*TimestampsFilePath*/ NCB::TPathWithScheme(),
                params.FeatureNamesPath,
                params.PoolMetaInfoPath,
                params.ClassLabels,
                params.ColumnarPoolFormatParams.DsvFormat,
                MakeCdProviderFromFile(params.ColumnarPoolFormatParams.CdFilePath),
                params.IgnoredFeatures,
                NCB::EObjectsOrder::Undefined,
                blockSize,
                NCB::TDatasetSubset::MakeColumns(),
                /*LoadColumnsAsString*/ false,
                params.ForceUnitAutoPairWeights,
                &localExecutor}});

    if (calculateStatisticsParams.OnlyGroupStatistics) {
        auto visitor = MakeHolder<TDatasetStatisticsOnlyGroupVisitor>(/*isLocal*/ true);

        datasetLoader->DoIfCompatible(dynamic_cast<IDatasetVisitor*>(visitor.Get()));

        visitor->OutputResult(calculateStatisticsParams.OutputPath);
    } else {
        auto visitor = MakeHolder<TDatasetStatisticsFullVisitor>(
            NCB::TDataProviderBuilderOptions{},
            /*isLocal*/ true,
            &localExecutor);

        datasetLoader->DoIfCompatible(dynamic_cast<IDatasetVisitor*>(visitor.Get()));

        visitor->OutputResult(calculateStatisticsParams.OutputPath);
    }
}
