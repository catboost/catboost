#include "baseline.h"
#include "cb_dsv_loader.h"
#include "load_data.h"
#include "loader.h"
#include "sampler.h"

#include <catboost/libs/column_description/cd_parser.h>
#include <catboost/libs/helpers/mem_usage.h>
#include <catboost/private/libs/data_util/exists_checker.h>
#include <catboost/private/libs/data_util/line_data_reader.h>
#include <catboost/private/libs/labels/helpers.h>
#include <catboost/private/libs/options/dataset_reading_params.h>
#include <catboost/private/libs/options/pool_metainfo_options.h>

#include <library/cpp/object_factory/object_factory.h>
#include <library/cpp/string_utils/csv/csv.h>

#include <util/generic/algorithm.h>
#include <util/generic/cast.h>
#include <util/generic/strbuf.h>
#include <util/generic/vector.h>
#include <util/generic/xrange.h>
#include <util/stream/file.h>
#include <util/string/split.h>
#include <util/system/compiler.h>
#include <util/system/guard.h>
#include <util/system/types.h>


namespace NCB {

    TCBDsvDataLoader::TCBDsvDataLoader(TDatasetLoaderPullArgs&& args)
        : TCBDsvDataLoader(
            TLineDataLoaderPushArgs {
                GetLineDataReader(args.PoolPath, args.CommonArgs.PoolFormat),
                std::move(args.CommonArgs)
            }
        )
    {
    }

    TCBDsvDataLoader::TCBDsvDataLoader(TLineDataLoaderPushArgs&& args)
        : TAsyncProcDataLoaderBase<TString>(std::move(args.CommonArgs))
        , FieldDelimiter(Args.PoolFormat.Delimiter)
        , NumVectorDelimiter(Args.PoolFormat.NumVectorDelimiter)
        , CsvSplitterQuote(Args.PoolFormat.IgnoreCsvQuoting ? '\0' : '"')
        , LineDataReader(std::move(args.Reader))
    {
        if (Args.BaselineFilePath.Inited()) {
            CB_ENSURE(
                CheckExists(Args.BaselineFilePath),
                "TCBDsvDataLoader:BaselineFilePath does not exist"
            );

            BaselineReader = GetProcessor<IBaselineReader, TBaselineReaderArgs>(
                Args.BaselineFilePath,
                TBaselineReaderArgs{
                    Args.BaselineFilePath,
                    ClassLabelsToStrings(Args.ClassLabels),
                    Args.DatasetSubset.Range
                }
            );
        }

        CB_ENSURE(
            !Args.PairsFilePath.Inited() || CheckExists(Args.PairsFilePath),
            "TCBDsvDataLoader:PairsFilePath does not exist"
        );
        CB_ENSURE(
            !Args.GraphFilePath.Inited() || CheckExists(Args.GraphFilePath),
            "TCBDsvDataLoader:GraphFilePath does not exist"
        );
        CB_ENSURE(
            !Args.GroupWeightsFilePath.Inited() || CheckExists(Args.GroupWeightsFilePath),
            "TCBDsvDataLoader:GroupWeightsFilePath does not exist"
        );
        CB_ENSURE(
            !Args.TimestampsFilePath.Inited() || CheckExists(Args.TimestampsFilePath),
            "TCBDsvDataLoader:TimestampsFilePath does not exist"
        );
        CB_ENSURE(
            !Args.FeatureNamesPath.Inited() || CheckExists(Args.FeatureNamesPath),
            "TCBDsvDataLoader:FeatureNamesPath does not exist"
        );
        CB_ENSURE(
            !Args.PoolMetaInfoPath.Inited() || CheckExists(Args.PoolMetaInfoPath),
            "TCBDsvDataLoader:PoolMetaInfoPath does not exist"
        );

        TMaybe<TString> header = LineDataReader->GetHeader();
        TMaybe<TVector<TString>> headerColumns;
        if (header) {
            headerColumns = TVector<TString>(NCsvFormat::CsvSplitter(*header, FieldDelimiter, CsvSplitterQuote));
        }

        TString firstLine;
        CB_ENSURE(LineDataReader->ReadLine(&firstLine), "TCBDsvDataLoader: no data rows in pool");
        const ui32 columnsCount = TVector<TString>(
            NCsvFormat::CsvSplitter(firstLine, FieldDelimiter, CsvSplitterQuote)
        ).size();

        auto columnsDescription = TDataColumnsMetaInfo{ CreateColumnsDescription(columnsCount) };
        auto targetCount = columnsDescription.CountColumns(EColumn::Label);

        const TVector<TString> featureNames = GetFeatureNames(
            columnsDescription,
            headerColumns,
            Args.FeatureNamesPath
        );

        const auto poolMetaInfoOptions = NCatboostOptions::LoadPoolMetaInfoOptions(Args.PoolMetaInfoPath);

        DataMetaInfo = TDataMetaInfo(
            std::move(columnsDescription),
            targetCount ? ERawTargetType::String : ERawTargetType::None,
            Args.GroupWeightsFilePath.Inited(),
            Args.TimestampsFilePath.Inited(),
            Args.PairsFilePath.Inited(),
            Args.GraphFilePath.Inited(),
            Args.LoadSampleIds,
            Args.ForceUnitAutoPairWeights,
            BaselineReader ? TMaybe<ui32>(BaselineReader->GetBaselineCount()) : Nothing(),
            &featureNames,
            &poolMetaInfoOptions.Tags.Get(),
            args.CommonArgs.ClassLabels
        );

        AsyncRowProcessor.AddFirstLine(std::move(firstLine));

        ProcessIgnoredFeaturesList(
            Args.IgnoredFeatures,
            /*allFeaturesIgnoredMessage*/ Nothing(),
            &DataMetaInfo,
            &FeatureIgnored
        );

        AsyncRowProcessor.ReadBlockAsync(GetReadFunc());
        if (BaselineReader) {
            AsyncBaselineRowProcessor.ReadBlockAsync(GetReadBaselineFunc());
        }
    }

    TVector<TColumn> TCBDsvDataLoader::CreateColumnsDescription(ui32 columnsCount) {
        return Args.CdProvider->GetColumnsDescription(columnsCount);
    }

    ui32 TCBDsvDataLoader::GetObjectCountSynchronized() {
        TGuard g(ObjectCountMutex);
        if (!ObjectCount) {
            const ui64 dataLineCount = LineDataReader->GetDataLineCount();
            CB_ENSURE(
                dataLineCount <= Max<ui32>(), "CatBoost does not support datasets with more than "
                << Max<ui32>() << " objects"
            );
            // cast is safe - was checked above
            ObjectCount = (ui32)dataLineCount;
        }
        return *ObjectCount;
    }

    void TCBDsvDataLoader::StartBuilder(bool inBlock,
                                          ui32 objectCount, ui32 /*offset*/,
                                          IRawObjectsOrderDataVisitor* visitor)
    {
        visitor->Start(
            inBlock,
            DataMetaInfo,
            /*haveUnknownNumberOfSparseFeatures*/ false,
            objectCount,
            Args.ObjectsOrder,
            {}
        );
    }

    inline static TVector<float> ProcessNumVector(TStringBuf token, char delimiter, ui32 featureId) {
        TVector<float> result;

        ui32 fieldIdx = 0;
        for (TStringBuf part: StringSplitter(token).Split(delimiter)) {
            float value;
            CB_ENSURE(
                TryFloatFromString(part, /*parseNonFinite*/true, &value),
                "Sub-field #" << fieldIdx << " of numeric vector for feature " << featureId
                << " cannot be parsed as float. Check data contents or column description"
            );
            result.push_back(value);
            ++fieldIdx;
        }

        return result;
    }

    void TCBDsvDataLoader::ProcessBlock(IRawObjectsOrderDataVisitor* visitor) {
        visitor->StartNextBlock(AsyncRowProcessor.GetParseBufferSize());

        auto& columnsDescription = DataMetaInfo.ColumnsInfo->Columns;

        auto parseLine = [&](TString& line, int lineIdx) {
            const auto& featuresLayout = *DataMetaInfo.FeaturesLayout;
            bool storeStringColumns = DataMetaInfo.StoreStringColumns;

            ui32 featureId = 0;
            ui32 targetId = 0;
            ui32 baselineIdx = 0;

            TVector<float> floatFeatures;
            floatFeatures.resize(featuresLayout.GetFloatFeatureCount());

            TVector<ui32> catFeatures;
            catFeatures.resize(featuresLayout.GetCatFeatureCount());

            TVector<TString> textFeatures;
            textFeatures.resize(featuresLayout.GetTextFeatureCount());

            TVector<TVector<float>> embeddingFeatures;
            embeddingFeatures.resize(featuresLayout.GetEmbeddingFeatureCount());

            size_t tokenIdx = 0;
            try {
                const bool floatFeaturesOnly = catFeatures.empty() && textFeatures.empty();
                auto splitter = NCsvFormat::CsvSplitter(
                    line,
                    FieldDelimiter,
                    floatFeaturesOnly ? '\0' : CsvSplitterQuote
                );
                do {
                    TStringBuf token = splitter.Consume();
                    CB_ENSURE(
                        tokenIdx < columnsDescription.size(),
                        "wrong column count: found token " << token << " with id more than "
                        << columnsDescription.ysize() << " values:\n" << line
                    );
                    try {
                        switch (columnsDescription[tokenIdx].Type) {
                            case EColumn::Categ: {
                                if (!FeatureIgnored[featureId]) {
                                    const ui32 catFeatureIdx = featuresLayout.GetInternalFeatureIdx(featureId);
                                    catFeatures[catFeatureIdx] = visitor->GetCatFeatureValue(lineIdx, featureId, token);
                                }
                                ++featureId;
                                break;
                            }
                            case EColumn::HashedCateg: {
                                if (!FeatureIgnored[featureId]) {
                                    if (!TryFromString<ui32>(
                                            token,
                                            catFeatures[featuresLayout.GetInternalFeatureIdx(featureId)]
                                        ))
                                    {
                                        CB_ENSURE(
                                            false,
                                            "Factor " << featureId << "=" << token << " cannot be parsed as "
                                            "hashed categorical value."
                                            " Try correcting column description file."
                                        );
                                    }
                                }
                                ++featureId;
                                break;
                            }
                            case EColumn::Num: {
                                if (!FeatureIgnored[featureId]) {
                                    if (!TryFloatFromString(
                                            token,
                                            /*parseNonFinite*/true,
                                            &floatFeatures[featuresLayout.GetInternalFeatureIdx(featureId)]
                                         ))
                                    {
                                        CB_ENSURE(
                                            false,
                                            "Factor " << featureId << "=" << token << " cannot be parsed as float."
                                            " Try correcting column description file."
                                        );
                                    }
                                }
                                ++featureId;
                                break;
                            }
                            case EColumn::Text: {
                                if (!FeatureIgnored[featureId]) {
                                    const ui32 textFeatureIdx = featuresLayout.GetInternalFeatureIdx(featureId);
                                    textFeatures[textFeatureIdx] = TString(token);
                                }
                                ++featureId;
                                break;
                            }
                            case EColumn::NumVector: {
                                if (!FeatureIgnored[featureId]) {
                                    const ui32 embeddingFeatureIdx
                                        = featuresLayout.GetInternalFeatureIdx(featureId);
                                    embeddingFeatures[embeddingFeatureIdx] = ProcessNumVector(
                                        token,
                                        NumVectorDelimiter,
                                        featureId
                                    );
                                }
                                ++featureId;
                                break;
                            }
                            case EColumn::Label: {
                                CB_ENSURE(token.length() != 0, "empty values not supported for Label");
                                visitor->AddTarget(targetId, lineIdx, TString(token));
                                ++targetId;
                                break;
                            }
                            case EColumn::Weight: {
                                CB_ENSURE(token.length() != 0, "empty values not supported for weight");
                                visitor->AddWeight(lineIdx, FromString<float>(token));
                                break;
                            }
                            case EColumn::Auxiliary: {
                                break;
                            }
                            case EColumn::GroupId: {
                                CB_ENSURE(token.length() != 0, "empty values not supported for GroupId");
                                if (storeStringColumns) {
                                    visitor->AddGroupId(lineIdx, TString(token));
                                } else {
                                    visitor->AddGroupId(lineIdx, CalcGroupIdFor(token));
                                }
                                break;
                            }
                            case EColumn::GroupWeight: {
                                CB_ENSURE(token.length() != 0, "empty values not supported for GroupWeight");
                                visitor->AddGroupWeight(lineIdx, FromString<float>(token));
                                break;
                            }
                            case EColumn::SubgroupId: {
                                CB_ENSURE(token.length() != 0, "empty values not supported for SubgroupId");
                                if (storeStringColumns) {
                                    visitor->AddSubgroupId(lineIdx, TString(token));
                                } else {
                                    visitor->AddSubgroupId(lineIdx, CalcSubgroupIdFor(token));
                                }
                                break;
                            }
                            case EColumn::Baseline: {
                                CB_ENSURE(token.length() != 0, "empty values not supported for Baseline");
                                visitor->AddBaseline(lineIdx, baselineIdx, FromString<float>(token));
                                ++baselineIdx;
                                break;
                            }
                            case EColumn::SampleId: {
                                if (Args.LoadSampleIds) {
                                    CB_ENSURE(token.length() != 0, "empty values not supported for SampleId");
                                    visitor->AddSampleId(lineIdx, TString(token));
                                }
                                break;
                            }
                            case EColumn::Timestamp: {
                                CB_ENSURE(token.length() != 0, "empty values not supported for Timestamp");
                                visitor->AddTimestamp(lineIdx, FromString<ui64>(token));
                                break;
                            }
                            default: {
                                CB_ENSURE(false, "wrong column type");
                            }
                        }
                    } catch (yexception& e) {
                        throw TCatBoostException() << "Column " << tokenIdx << " (type "
                            << columnsDescription[tokenIdx].Type << ", value = \"" << token
                            << "\"): " << e.what();
                    }
                    ++tokenIdx;
                } while (splitter.Step());
                CB_ENSURE(
                    tokenIdx == columnsDescription.size(),
                    "wrong column count: expected " << columnsDescription.ysize() << ", found " << tokenIdx
                );
                if (!floatFeatures.empty()) {
                    visitor->AddAllFloatFeatures(lineIdx, floatFeatures);
                }
                if (!catFeatures.empty()) {
                    visitor->AddAllCatFeatures(lineIdx, catFeatures);
                }
                if (!textFeatures.empty()) {
                    visitor->AddAllTextFeatures(lineIdx, textFeatures);
                }
                if (!embeddingFeatures.empty()) {
                    for (auto embeddingFeatureIdx : xrange(embeddingFeatures.size())) {
                        visitor->AddEmbeddingFeature(
                            lineIdx,
                            featuresLayout.GetEmbeddingFeatureInternalIdxToExternalIdx()[embeddingFeatureIdx],
                            TMaybeOwningConstArrayHolder<float>::CreateOwning(
                                std::move(embeddingFeatures[embeddingFeatureIdx])
                            )
                        );
                    }
                }
            } catch (yexception& e) {
                throw TCatBoostException() << "Error in dsv data. Line " <<
                    AsyncRowProcessor.GetLinesProcessed() + lineIdx + 1 << ": " << e.what();
            }
        };

        AsyncRowProcessor.ProcessBlock(parseLine);

        if (BaselineReader) {
            auto setBaselineBlock = [&](TObjectBaselineData &data, int inBlockIdx) {
                for (auto baselineIdx : xrange(data.Baseline.size())) {
                    visitor->AddBaseline(inBlockIdx, baselineIdx, data.Baseline[baselineIdx]);
                }
            };

            AsyncBaselineRowProcessor.ProcessBlock(setBaselineBlock);
        }
    }

    size_t GetDsvColumnCount(const TPathWithScheme& pathWithScheme, const TDsvFormatOptions& format) {
        CB_ENSURE_INTERNAL(pathWithScheme.Scheme == "dsv", "Unsupported scheme " << pathWithScheme.Scheme);
        TString firstLine;
        CB_ENSURE(
            GetLineDataReader(pathWithScheme, format)->ReadLine(&firstLine),
            "TCBDsvDataLoader: no data rows in pool"
        );
        return TVector<TString>(
            NCsvFormat::CsvSplitter(firstLine, format.Delimiter, format.IgnoreCsvQuoting ? '\0' : '"')
        ).size();
    }

    static void AugmentWithExternalFeatureNames(
        TConstArrayRef<TString> featureNames,
        TArrayRef<TColumn>* columnsDescription
    ) {
        size_t featureIdx = 0;
        for (auto& column : *columnsDescription) {
            if (IsFactorColumn(column.Type)) {
                if (featureIdx == featureNames.size()) {
                    break;
                }
                if (column.Id.empty()) {
                    column.Id = featureNames[featureIdx];
                } else {
                    CB_ENSURE(
                        column.Id == featureNames[featureIdx],
                        "Feature #" << featureIdx << ": name from columns specification (\""
                        << column.Id
                        << "\") is not equal to external feature name (\""
                        << featureNames[featureIdx] << "\")"
                    );
                }
                ++featureIdx;
            } else {
                CB_ENSURE(column.SubColumns.empty(), "SubColumns not supported");
            }
        }
    }

    TVector<TColumn> CreateDsvColumnsDescription(
        const TPathWithScheme& datasetPathWithScheme,
        const TPathWithScheme& cdPathWithScheme,
        const TPathWithScheme& featureNamesPathWithScheme,
        const TDsvFormatOptions& format
    ) {
        auto lineDataReader = GetLineDataReader(datasetPathWithScheme, format);

        auto csvSplitterQuote = format.IgnoreCsvQuoting ? '\0' : '"';

        TMaybe<TString> header = lineDataReader->GetHeader();
        TMaybe<TVector<TString>> headerColumns;
        int columnCount = 0;
        if (header) {
            headerColumns = TVector<TString>(
                NCsvFormat::CsvSplitter(*header, format.Delimiter, csvSplitterQuote)
            );
            columnCount = SafeIntegerCast<int>(headerColumns->size());
        } else {
            TString firstLine;
            CB_ENSURE(lineDataReader->ReadLine(&firstLine), "no data rows in pool");
            columnCount = SafeIntegerCast<int>(
                TVector<TString>(
                    NCsvFormat::CsvSplitter(firstLine, format.Delimiter, csvSplitterQuote)
                ).size()
            );
        }

        auto dataColumnsMetaInfo = TDataColumnsMetaInfo{
            cdPathWithScheme.Inited() ?
                ReadCD(cdPathWithScheme, TCdParserDefaults(EColumn::Num, columnCount))
                : MakeDefaultColumnsDescription(columnCount)
        };

        auto featureNames = GetFeatureNames(dataColumnsMetaInfo, headerColumns, featureNamesPathWithScheme);
        auto columnsDescription = std::move(dataColumnsMetaInfo.Columns);

        TArrayRef<TColumn> columnsDescriptionRef(columnsDescription);
        AugmentWithExternalFeatureNames(featureNames, &columnsDescriptionRef);

        return columnsDescription;
    }

    namespace {
        TDatasetLoaderFactory::TRegistrator<TCBDsvDataLoader> DefDataLoaderReg("");
        TDatasetLoaderFactory::TRegistrator<TCBDsvDataLoader> CBDsvDataLoaderReg("dsv");

        TDatasetLineDataLoaderFactory::TRegistrator<TCBDsvDataLoader> CBDsvLineDataLoader("dsv");
    }


    class TSampleIdSubsetDsvLineDataReader final : public ILineDataReader {
    public:
        TSampleIdSubsetDsvLineDataReader(
            THolder<ILineDataReader>&& lineDataReader,
            THashMap<TString, ui32>&& subsetSampleIdsToCount,
            TDsvFormatOptions&& dstFormatOptions,
            size_t sampleIdColumnIdx
        )
            : LineDataReader(std::move(lineDataReader))
            , SubsetSampleIdsToCount(std::move(subsetSampleIdsToCount))
            , DsvFormatOptions(std::move(dstFormatOptions))
            , CsvSplitterQuote(DsvFormatOptions.IgnoreCsvQuoting ? '\0' : '"')
            , SampleIdColumnIdx(sampleIdColumnIdx)
            , LineIdx(0)
            , Header(LineDataReader->GetHeader())
        {
            if (!SubsetSampleIdsToCount.empty()) {
                Size = 0;
                for (const auto& [sampleId, count] : SubsetSampleIdsToCount) {
                    Size += count;
                }
            }
        }

        ui64 GetDataLineCount(bool estimate = false) override {
            Y_UNUSED(estimate);
            return Size;
        }

        TMaybe<TString> GetHeader() override {
            return Header;
        }

        bool ReadLine(TString* line, ui64* lineIdx = nullptr) override {
            if (SubsetSampleIdsToCount.empty()) {
                return false;
            }

            while (true) {
                auto it = SubsetSampleIdsToCount.find(CurrentSampleId);
                if (it != SubsetSampleIdsToCount.end()) {
                    if (it->second == 1) {
                        SubsetSampleIdsToCount.erase(it);
                        *line = std::move(LineBuffer);
                    } else {
                        --(it->second);
                        *line = LineBuffer;
                    }

                    if (lineIdx) {
                        *lineIdx = LineIdx;
                    }
                    ++LineIdx;
                    return true;
                }
                NextLine();
            }
        }

    private:
        void NextLine() {
            bool enclosingReadResult = LineDataReader->ReadLine(&LineBuffer);
            CB_ENSURE(enclosingReadResult, "Reached the end of data but not reached the end of subset");

            auto splitter = NCsvFormat::CsvSplitter(LineBuffer, DsvFormatOptions.Delimiter, CsvSplitterQuote);
            for (size_t i = 0; i < SampleIdColumnIdx; ++i) {
                splitter.Consume();
                splitter.Step();
            }
            CurrentSampleId = splitter.Consume();
        }

    private:
        THolder<ILineDataReader> LineDataReader;
        THashMap<TString, ui32> SubsetSampleIdsToCount;
        TDsvFormatOptions DsvFormatOptions;
        char CsvSplitterQuote;
        size_t SampleIdColumnIdx;

        //ui64 EnclosingLineIdx;
        ui64 LineIdx;
        TString CurrentSampleId;
        TMaybe<TString> Header;
        size_t Size = 0;
        TString LineBuffer;
    };


    class TDsvDatasetSampler final : public IDataProviderSampler {
    public:
        TDsvDatasetSampler(TDataProviderSampleParams&& params)
            : Params(std::move(params))
        {}

        TDataProviderPtr SampleByIndices(TConstArrayRef<ui32> indices) override {
            return LinesFileSampleByIndices(Params, indices);
        }

        TDataProviderPtr SampleBySampleIds(TConstArrayRef<TString> sampleIds) override {
            const auto& datasetReadingParams = Params.DatasetReadingParams;
            const auto& dsvFormat = datasetReadingParams.ColumnarPoolFormatParams.DsvFormat;

            auto lineDataReader = MakeHolder<TFileLineDataReader>(
                TLineDataReaderArgs {
                    datasetReadingParams.PoolPath,
                    dsvFormat,
                    /*KeepLineOrder*/ false
                }
            );

            THashMap<TString, ui32> subsetSampleIdsToCount;
            for (const auto& sampleId : sampleIds) {
                subsetSampleIdsToCount[sampleId]++;
            }

            TVector<TColumn> columnsDescription;
            TMaybe<size_t> sampleIdColumnIdx;

            ReadCDForSampler(
                datasetReadingParams.ColumnarPoolFormatParams.CdFilePath,
                Params.OnlyFeaturesData,
                /*loadSampleIds*/ true,
                &columnsDescription,
                &sampleIdColumnIdx
            );

            CB_ENSURE(sampleIdColumnIdx.Defined(), "SampleId column not present in CD file");

            auto subsetLineDataReader = MakeHolder<TSampleIdSubsetDsvLineDataReader>(
                std::move(lineDataReader),
                std::move(subsetSampleIdsToCount),
                TDsvFormatOptions(dsvFormat),
                *sampleIdColumnIdx
            );

            TVector<NJson::TJsonValue> classLabels = datasetReadingParams.ClassLabels;

            auto dataset = ReadDataset(
                std::move(subsetLineDataReader),
                /*pairsFilePath*/ TPathWithScheme(),
                /*groupFilePath*/ TPathWithScheme(),
                /*groupWeightsFilePath*/ TPathWithScheme(),
                /*timestampsFilePath*/ TPathWithScheme(),
                /*baselineFilePath*/ TPathWithScheme(),
                datasetReadingParams.FeatureNamesPath,
                datasetReadingParams.PoolMetaInfoPath,
                dsvFormat,
                columnsDescription,
                /*ignoredFeatures*/ {},  //TODO(akhropov): get unused features from the model
                EObjectsOrder::Ordered,
                /*loadSampleIds*/ true,
                /*forceUnitAutoPairWeights*/ false,
                &classLabels,
                Params.LocalExecutor
            );

            return DataProviderSamplerReorderBySampleIds(Params, dataset, sampleIds);
        }

    private:
        TDataProviderSampleParams Params;
    };

    namespace {
        TDataProviderSamplerFactory::TRegistrator<TDsvDatasetSampler> DefDatasetSampler("");
        TDataProviderSamplerFactory::TRegistrator<TDsvDatasetSampler> CBDsvDatasetSampler("dsv");
    }
}
