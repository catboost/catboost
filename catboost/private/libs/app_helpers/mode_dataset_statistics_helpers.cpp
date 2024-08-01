#include "mode_dataset_statistics_helpers.h"

#include <catboost/libs/data/proceed_pool_in_blocks.h>
#include <catboost/libs/helpers/array_subset.h>

#include <library/cpp/getopt/small/last_getopt.h>

#include <util/generic/cast.h>
#include <util/generic/ptr.h>
#include <util/generic/xrange.h>
#include <util/generic/ymath.h>
#include <util/system/info.h>

using namespace NCB;

void TCalculateStatisticsParams::BindParserOpts(NLastGetopt::TOpts& parser) {
    DatasetReadingParams.BindParserOpts(&parser);
    parser.AddLongOption('o', "output-path", "output result path")
        .StoreResult(&OutputPath)
        .DefaultValue("statistics.json");
    parser.AddLongOption('T', "thread-count", "worker thread count (default: core count)")
        .StoreResult(&ThreadCount);
    parser.AddLongOption("only-group-statistics")
        .OptionalValue("true", "bool")
        .Handler1T<TString>([&](const TString& param) {
            OnlyGroupStatistics = FromString<bool>(param);
        });
    parser.AddLongOption("histograms-path", "output result path")
        .StoreResult(&HistogramPath)
        .DefaultValue("histograms.json");
    parser.AddLongOption("only-light-statistics")
        .OptionalValue("true", "bool")
        .Handler1T<TString>([&](const TString& param) {
            OnlyLightStatistics = FromString<bool>(param);
        });
    parser.AddLongOption("not-convert-string-targets")
        .OptionalValue("true", "bool")
        .Handler1T<TString>([&](const TString& param) {
            ConvertStringTargets = !FromString<bool>(param);
        });
    parser.AddLongOption("custom-feature-limits", "comma separated list of feature limits description in format <feature_id>:<min>:<max>,"
                                                  "for example: 0:0:1,10:-2.1:-1")
        .RequiredArgument("string")
        .Handler1T<TString>([&](const TString& limitsDescription) {
            for (const TStringBuf& ignoredFeature : StringSplitter(limitsDescription).Split(',')) {
                TVector<TString> tokens = StringSplitter(ignoredFeature).Split(':');
                if (tokens.empty()) {
                    continue;
                }
                CB_ENSURE(tokens.size() == 3, "Inappropriate feature limits description: " << TString(ignoredFeature));
                ui32 featureId = FromString<ui32>(tokens[0]);
                double minValue = -std::numeric_limits<float>::infinity();
                double maxValue = std::numeric_limits<float>::infinity();
                if (tokens[1] != "-inf") {
                    minValue = FromString<double>(tokens[1]);
                }
                if (tokens[2] != "inf") {
                    maxValue = FromString<double>(tokens[2]);
                }
                CB_ENSURE(minValue <= maxValue, "Inappropriate feature limits description: " << TString(ignoredFeature));
                CB_ENSURE(FeatureLimits.find(featureId) == FeatureLimits.end(),
                          "Duplicate feature " << featureId << "in custom-feature-limits");
                FeatureLimits[featureId] = {minValue, maxValue};
            }
        });
    parser.AddLongOption("border-counts")
        .RequiredArgument("INT")
        .StoreResult(&BorderCount)
        .DefaultValue("256");
    parser.AddLongOption("spot-size", "size of the single spot part (default: process the whole dataset)")
        .StoreResult(&SpotSize);
    parser.AddLongOption("spot-count", "number of spot parts (default: process the whole dataset)")
        .StoreResult(&SpotCount);
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

    CB_ENSURE((SpotSize == 0) == (SpotCount == 0), "spot size and spot count must be specified together");
}

TVector<TIndexRange<ui64>> NCB::GetSpots(ui64 datasetSize, ui64 spotSize, ui64 spotCount) {
    TVector<TIndexRange<ui64>> spots;
    ui64 partSize = datasetSize / spotCount;
    ui64 spotShift = partSize / spotCount;
    for (auto spotNumber : xrange(spotCount)) {
        auto partBegin = spotNumber * partSize;
        auto partEnd = Min(partBegin + partSize, datasetSize);
        auto spotBegin = Min(partBegin + spotNumber * spotShift, partEnd);
        auto spotEnd = Min(spotBegin + spotSize, partEnd);
        spots.emplace_back(spotBegin, spotEnd);
    }
    return spots;
}


// can return empty holder if no spots are used
static THolder<ILineDataReader> GetSpotsLineDataReader(
    const TPathWithScheme& poolPath,
    const NCB::TDsvFormatOptions& dsvFormat,
    ui32 spotSize,
    ui32 spotCount
) {
    if (spotSize == 0) {
        return THolder<ILineDataReader>();
    }

    auto lineDataReaderPtr = TLineDataReaderFactory::Construct(poolPath.Scheme, TLineDataReaderArgs{poolPath, dsvFormat});
    if (!lineDataReaderPtr) {
        CATBOOST_INFO_LOG << "There's no line data reader for scheme '" << poolPath.Scheme << "', the whole dataset will be processed instead of spots\n";
        return THolder<ILineDataReader>();
    }
    THolder<ILineDataReader> lineDataReader(lineDataReaderPtr);

    auto dataLineCount = lineDataReader->GetDataLineCount();
    if (dataLineCount > ui64(spotCount) * ui64(spotSize)) {
        return MakeHolder<TBlocksSubsetLineDataReader>(
            std::move(lineDataReader),
            GetSpots(dataLineCount, spotSize, spotCount)
        );
    } else {
        return lineDataReader;
    }
}

static THolder<IDatasetLoader> GetDatasetLoader(
    const TCalculateStatisticsParams& calculateStatisticsParams,
    NPar::TLocalExecutor* localExecutor
) {
    const ui32 blockSize = Max<ui32>(
        10000, 10000 // ToDo: meaningful estimation
    );

    const NCatboostOptions::TDatasetReadingParams& params = calculateStatisticsParams.DatasetReadingParams;

    THolder<IDatasetLoader> datasetLoader;

    auto getDatasetLoaderCommonArgs = [&] () {
        return TDatasetLoaderCommonArgs{
            params.PairsFilePath,
            params.GraphFilePath,
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
            /*LoadSampleIds*/ false,
            params.ForceUnitAutoPairWeights,
            localExecutor};
    };

    auto spotsLineDataReader = GetSpotsLineDataReader(
        params.PoolPath,
        params.ColumnarPoolFormatParams.DsvFormat,
        calculateStatisticsParams.SpotSize,
        calculateStatisticsParams.SpotCount);

    if (spotsLineDataReader) {
        auto datasetLoaderPtr = TDatasetLineDataLoaderFactory::Construct(
            params.PoolPath.Scheme,
            TLineDataLoaderPushArgs{std::move(spotsLineDataReader), getDatasetLoaderCommonArgs()});
        if (datasetLoaderPtr) {
            datasetLoader.Reset(datasetLoaderPtr);
        } else {
            CATBOOST_INFO_LOG << "There's no dataset line data loader for scheme '" << params.PoolPath.Scheme
                << "', the whole dataset will be processed instead of spots\n";
        }
    }

    if (!datasetLoader) {
        datasetLoader = GetProcessor<IDatasetLoader>(
            params.PoolPath, // for choosing processor
            // processor args
            NCB::TDatasetLoaderPullArgs{params.PoolPath, getDatasetLoaderCommonArgs()});
    }

    return std::move(datasetLoader);
}

static void CalculateHistogram(
    const TCalculateStatisticsParams& calculateStatisticsParams,
    const TFeatureStatistics& featuresStatistics,
    NPar::TLocalExecutor* localExecutor
) {
    THistograms histograms;

    auto floatFeatureCount = featuresStatistics.FloatFeatureStatistics.size();
    if (floatFeatureCount > 0) {
        // 2nd pass
        auto datasetLoader = GetDatasetLoader(calculateStatisticsParams, localExecutor);

        TVector<size_t> borderCounts(floatFeatureCount, calculateStatisticsParams.BorderCount);

        histograms = InitHistograms(borderCounts, featuresStatistics);

        ReadAndProceedPoolInBlocks(
            std::move(datasetLoader),
            calculateStatisticsParams.DatasetReadingParams.PairsFilePath.Inited(),
            [&] (TDataProviderPtr dataProvider) {
                if (const auto* rawObjectsDataProvider
                        = dynamic_cast<const TRawObjectsDataProvider*>(dataProvider->ObjectsData.Get()))
                {
                    for (auto floatFeatureIdx : xrange(floatFeatureCount)) {
                        auto floatFeatureData = rawObjectsDataProvider->GetFloatFeature(floatFeatureIdx);
                        if (floatFeatureData.Defined()) {
                            auto values = (*floatFeatureData)->ExtractValues(localExecutor);
                            histograms.AddFloatFeatureUniformHistogram(floatFeatureIdx, *values);
                        }
                    }
                } else {
                    CB_ENSURE(false, "Non-raw pool formats are not yet supported for dataset histograms calculation");
                }
            },
            localExecutor
        );
    }

    TFileOutput histOutput(calculateStatisticsParams.HistogramPath);
    WriteJsonWithCatBoostPrecision(histograms.ToJson(), false, &histOutput);
}


void NCB::CalculateDatasetStatisticsSingleHost(const TCalculateStatisticsParams& calculateStatisticsParams) {
    NPar::TLocalExecutor localExecutor;

    int threadCount = (calculateStatisticsParams.ThreadCount == -1) ?
        SafeIntegerCast<int>(NSystemInfo::CachedNumberOfCpus())
        : calculateStatisticsParams.ThreadCount;

    localExecutor.RunAdditionalThreads(threadCount - 1);

    auto datasetLoader = GetDatasetLoader(calculateStatisticsParams, &localExecutor);

    if (calculateStatisticsParams.OnlyGroupStatistics) {
        TDatasetStatisticsOnlyGroupVisitor visitor(/*isLocal*/ true);

        datasetLoader->DoIfCompatible(dynamic_cast<IDatasetVisitor*>(&visitor));

        visitor.OutputResult(calculateStatisticsParams.OutputPath);
    } else {
        auto visitor = MakeHolder<TDatasetStatisticsFullVisitor>(
            NCB::TDataProviderBuilderOptions{},
            /*isLocal*/ true,
            &localExecutor);

        visitor->SetCustomBorders(calculateStatisticsParams.FeatureLimits, /*targetCustomBorders*/ TFeatureCustomBorders());
        visitor->SetConvertStringTargets(calculateStatisticsParams.ConvertStringTargets);

        datasetLoader->DoIfCompatible(dynamic_cast<IDatasetVisitor*>(visitor.Get()));

        visitor->OutputResult(calculateStatisticsParams.OutputPath);

        if (!calculateStatisticsParams.OnlyLightStatistics) {
            CATBOOST_INFO_LOG << "Calculate statistics histogram" << Endl;

            TFeatureStatistics featureStatistics = visitor->GetDatasetStatistics().FeatureStatistics;

            // clean up resources before 2nd pass
            visitor.Destroy();
            datasetLoader.Destroy();

            CalculateHistogram(
                calculateStatisticsParams,
                featureStatistics,
                &localExecutor
            );
        }
    }
}
