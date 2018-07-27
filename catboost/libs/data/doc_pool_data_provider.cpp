
#include "doc_pool_data_provider.h"

#include "load_data.h"
#include "load_helpers.h"
#include "quantized.h"

#include <catboost/libs/column_description/cd_parser.h>

#include <catboost/libs/data_util/exists_checker.h>

#include <catboost/libs/helpers/mem_usage.h>

#include <catboost/libs/quantization_schema/schema.h>
#include <catboost/libs/quantization_schema/serialization.h>
#include <catboost/libs/quantized_pool/pool.h>
#include <catboost/libs/quantized_pool/serialization.h>

#include <library/object_factory/object_factory.h>

#include <util/generic/maybe.h>
#include <util/generic/strbuf.h>
#include <util/generic/vector.h>

#include <util/stream/file.h>

#include <util/system/types.h>


namespace NCB {

    TVector<TPair> ReadPairs(const TPathWithScheme& filePath, int docCount) {
        THolder<ILineDataReader> reader = GetLineDataReader(filePath);

        TVector<TPair> pairs;
        TString line;
        while (reader->ReadLine(&line)) {
            TVector<TString> tokens;
            try {
                Split(line, "\t", tokens);
            }
            catch (const yexception& e) {
                MATRIXNET_DEBUG_LOG << "Got exception " << e.what() << " while parsing pairs line " << line << Endl;
                break;
            }
            if (tokens.empty()) {
                continue;
            }
            CB_ENSURE(tokens.ysize() == 2 || tokens.ysize() == 3, "Each line should have two or three columns. Invalid line number " << line);
            int winnerId = FromString<int>(tokens[0]);
            int loserId = FromString<int>(tokens[1]);
            float weight = 1;
            if (tokens.ysize() == 3) {
                weight = FromString<float>(tokens[2]);
            }
            CB_ENSURE(winnerId >= 0 && winnerId < docCount, "Invalid winner index " << winnerId);
            CB_ENSURE(loserId >= 0 && loserId < docCount, "Invalid loser index " << loserId);
            pairs.emplace_back(winnerId, loserId, weight);
        }

        return pairs;
    }

    void WeightPairs(TConstArrayRef<float> groupWeight, TVector<TPair>* pairs) {
        for (auto& pair: *pairs) {
            pair.Weight *= groupWeight[pair.WinnerId];
        }
    }

    void SetPairs(const TPathWithScheme& pairsPath, bool haveGroupWeights, IPoolBuilder* poolBuilder) {
        DumpMemUsage("After data read");
        if (pairsPath.Inited()) {
            TVector<TPair> pairs = ReadPairs(pairsPath, poolBuilder->GetDocCount());
            if (haveGroupWeights) {
                WeightPairs(poolBuilder->GetWeight(), &pairs);
            }
            poolBuilder->SetPairs(pairs);
        }
    }

    bool IsNanValue(const TStringBuf& s) {
        return s == "nan" || s == "NaN" || s == "NAN" || s == "NA" || s == "Na" || s == "na";
    }

    TTargetConverter::TTargetConverter(const TVector<TString>& classNames)
        : ClassNames(classNames)
    {
    }

    float TTargetConverter::operator()(const TString& word) const {
        if (ClassNames.empty()) {
            CB_ENSURE(!IsNanValue(word), "NaN not supported for target");
            return FromString<float>(word);
        }

        for (int classIndex = 0; classIndex < ClassNames.ysize(); ++classIndex) {
            if (ClassNames[classIndex] == word) {
                return classIndex;
            }
        }

        CB_ENSURE(false, "Unknown class name: " + word);
        return UNDEFINED_CLASS;
    }

    TCBDsvDataProvider::TCBDsvDataProvider(TDocPoolDataProviderArgs&& args)
        : TAsyncProcDataProviderBase<TString>(std::move(args))
        , FieldDelimiter(Args.DsvPoolFormatParams.Format.Delimiter)
        , ConvertTarget(Args.ClassNames)
        , LineDataReader(
            GetLineDataReader(Args.PoolPath, Args.DsvPoolFormatParams.Format)
          )
    {
        CB_ENSURE(!Args.PairsFilePath.Inited() || CheckExists(Args.PairsFilePath),
                  "TCBDsvDataProvider:PairsFilePath does not exist");

        PoolMetaInfo.ColumnsInfo.ConstructInPlace();

        TMaybe<TString> header = LineDataReader->GetHeader();

        TString firstLine;
        CB_ENSURE(LineDataReader->ReadLine(&firstLine), "TCBDsvDataProvider: no data rows in pool");

        ui32 columnsCount = StringSplitter(firstLine).Split(FieldDelimiter).Count();
        AsyncRowProcessor.AddFirstLine(std::move(firstLine));

        auto& columnsInfo = PoolMetaInfo.ColumnsInfo;
        columnsInfo->Columns = CreateColumnsDescription(columnsCount);
        columnsInfo->Validate();

        PoolMetaInfo.BaselineCount = columnsInfo->CountColumns(EColumn::Baseline);
        PoolMetaInfo.HasWeights = columnsInfo->CountColumns(EColumn::Weight) != 0;
        PoolMetaInfo.HasDocIds = columnsInfo->CountColumns(EColumn::DocId) != 0;
        PoolMetaInfo.HasGroupId = columnsInfo->CountColumns(EColumn::GroupId) != 0;
        PoolMetaInfo.HasGroupWeight = columnsInfo->CountColumns(EColumn::GroupWeight) != 0;
        PoolMetaInfo.HasSubgroupIds = columnsInfo->CountColumns(EColumn::SubgroupId) != 0;
        PoolMetaInfo.HasTimestamp = columnsInfo->CountColumns(EColumn::Timestamp) != 0;
        PoolMetaInfo.FeatureCount = (const ui32)CountIf(
            columnsInfo->Columns.begin(),
            columnsInfo->Columns.end(),
            [](const auto x) -> bool {
                return IsFactorColumn(x.Type);
            }
        );
        CB_ENSURE(PoolMetaInfo.FeatureCount > 0, "Pool should have at least one factor");

        int featureCount = static_cast<int>(PoolMetaInfo.FeatureCount);
        int ignoredFeatureCount = 0;
        FeatureIgnored.resize(featureCount, false);
        for (int featureId : Args.IgnoredFeatures) {
            CB_ENSURE(0 <= featureId && featureId < featureCount, "Invalid ignored feature id: " << featureId);
            ignoredFeatureCount += FeatureIgnored[featureId] == false;
            FeatureIgnored[featureId] = true;
        }
        CB_ENSURE(featureCount - ignoredFeatureCount > 0, "All features are requested to be ignored");

        CB_ENSURE(!(PoolMetaInfo.HasWeights && PoolMetaInfo.HasGroupWeight), "Pool must have either Weight column or GroupWeight column");

        CatFeatures = columnsInfo->GetCategFeatures();

        FeatureIds = columnsInfo->GenerateFeatureIds(header, FieldDelimiter);

        AsyncRowProcessor.ReadBlockAsync(GetReadFunc());
    }

    TVector<TColumn> TCBDsvDataProvider::CreateColumnsDescription(ui32 columnsCount) {
        TVector<TColumn> columnsDescription;

        const auto& cdFilePath = Args.DsvPoolFormatParams.CdFilePath;

        if (cdFilePath.Inited()) {
            columnsDescription = ReadCD(cdFilePath, TCdParserDefaults(EColumn::Num, columnsCount));
        } else {
            columnsDescription.assign(columnsCount, TColumn{EColumn::Num, TString()});
            columnsDescription[0].Type = EColumn::Label;
        }

        return columnsDescription;
    }


    void TCBDsvDataProvider::StartBuilder(bool /*inBlock*/,
                                          int docCount, int offset,
                                          IPoolBuilder* poolBuilder)
    {
        poolBuilder->Start(PoolMetaInfo, docCount, CatFeatures);
        if (!FeatureIds.empty()) {
            poolBuilder->SetFeatureIds(FeatureIds);
        }
        if (!PoolMetaInfo.HasDocIds) {
            poolBuilder->GenerateDocIds(offset);
        }
    }


    void TCBDsvDataProvider::ProcessBlock(IPoolBuilder* poolBuilder) {
        poolBuilder->StartNextBlock(AsyncRowProcessor.GetParseBufferSize());

        auto& columnsDescription = PoolMetaInfo.ColumnsInfo->Columns;

        auto parseBlock = [&](TString& line, int lineIdx) {
            ui32 featureId = 0;
            ui32 baselineIdx = 0;
            TVector<float> features;
            features.yresize(PoolMetaInfo.FeatureCount);

            int tokenCount = 0;
            TVector<TStringBuf> tokens = StringSplitter(line).Split(FieldDelimiter).ToList<TStringBuf>();
            for (const auto& token : tokens) {
                switch (columnsDescription[tokenCount].Type) {
                    case EColumn::Categ: {
                        if (!FeatureIgnored[featureId]) {
                            if (IsNanValue(token)) {
                                features[featureId] = poolBuilder->GetCatFeatureValue("nan");
                            } else {
                                features[featureId] = poolBuilder->GetCatFeatureValue(token);
                            }
                        }
                        ++featureId;
                        break;
                    }
                    case EColumn::Num: {
                        if (!FeatureIgnored[featureId]) {
                            float val;
                            if (!TryFromString<float>(token, val)) {
                                if (IsNanValue(token)) {
                                    val = std::numeric_limits<float>::quiet_NaN();
                                } else if (token.length() == 0) {
                                    val = std::numeric_limits<float>::quiet_NaN();
                                } else {
                                    CB_ENSURE(false, "Factor " << featureId << " (column " << tokenCount + 1 << ") is declared `Num`," <<
                                        " but has value '" << token << "' in row "
                                        << AsyncRowProcessor.GetLinesProcessed() + lineIdx + 1
                                        << " that cannot be parsed as float. Try correcting column description file.");
                                }
                            }
                            features[featureId] = val == 0.0f ? 0.0f : val; // remove negative zeros
                        }
                        ++featureId;
                        break;
                    }
                    case EColumn::Label: {
                        CB_ENSURE(token.length() != 0, "empty values not supported for Label. Label should be float.");
                        poolBuilder->AddTarget(lineIdx, ConvertTarget(FromString<TString>(token)));
                        break;
                    }
                    case EColumn::Weight: {
                        CB_ENSURE(token.length() != 0, "empty values not supported for weight");
                        poolBuilder->AddWeight(lineIdx, FromString<float>(token));
                        break;
                    }
                    case EColumn::Auxiliary: {
                        break;
                    }
                    case EColumn::GroupId: {
                        CB_ENSURE(token.length() != 0, "empty values not supported for GroupId");
                        poolBuilder->AddQueryId(lineIdx, CalcGroupIdFor(token));
                        break;
                    }
                    case EColumn::GroupWeight: {
                        CB_ENSURE(token.length() != 0, "empty values not supported for GroupWeight");
                        poolBuilder->AddWeight(lineIdx, FromString<float>(token));
                        break;
                    }
                    case EColumn::SubgroupId: {
                        CB_ENSURE(token.length() != 0, "empty values not supported for SubgroupId");
                        poolBuilder->AddSubgroupId(lineIdx, CalcSubgroupIdFor(token));
                        break;
                    }
                    case EColumn::Baseline: {
                        CB_ENSURE(token.length() != 0, "empty values not supported for Baseline");
                        poolBuilder->AddBaseline(lineIdx, baselineIdx, FromString<double>(token));
                        ++baselineIdx;
                        break;
                    }
                    case EColumn::DocId: {
                        CB_ENSURE(token.length() != 0, "empty values not supported for DocId");
                        poolBuilder->AddDocId(lineIdx, token);
                        break;
                    }
                    case EColumn::Timestamp: {
                        CB_ENSURE(token.length() != 0, "empty values not supported for Timestamp");
                        poolBuilder->AddTimestamp(lineIdx, FromString<ui64>(token));
                        break;
                    }
                    default: {
                        CB_ENSURE(false, "wrong column type");
                    }
                }
                ++tokenCount;
            }
            poolBuilder->AddAllFloatFeatures(lineIdx, features);
            CB_ENSURE(tokenCount == columnsDescription.ysize(), "wrong columns number in pool line " <<
                      AsyncRowProcessor.GetLinesProcessed() + lineIdx + 1 << ": expected " << columnsDescription.ysize() << ", found " << tokenCount);
        };

        AsyncRowProcessor.ProcessBlock(parseBlock);
    }


    // Quantized data provider

    namespace {
        TVector<TFloatFeature> GetFeatureInfo(const TPoolQuantizationSchema& quantizationSchema, size_t targetColumn) {
            Y_ASSERT(quantizationSchema.FeatureIndices.size() == quantizationSchema.Borders.size());
            Y_ASSERT(quantizationSchema.FeatureIndices.size() == quantizationSchema.NanModes.size());
            const auto& quantizedColumns = quantizationSchema.FeatureIndices;
            TVector<TFloatFeature> featureInfo(quantizedColumns.size());
            for (int featureIdx = 0; featureIdx < quantizedColumns.ysize(); ++featureIdx) {
                featureInfo[featureIdx] = TFloatFeature(/*hasNans*/ false, // TODO(yazevnul): add this info to quantized pools
                    /*featureIndex*/ featureIdx,
                    /*flatFeatureIndex*/ quantizedColumns[featureIdx] - (quantizedColumns[featureIdx] > targetColumn), // do not count target column
                    /*borders*/ quantizationSchema.Borders[featureIdx]);
            }
            return featureInfo;
        }
    }

    class TCBQuantizedDataProvider : public IDocPoolDataProvider
    {
    public:
        explicit TCBQuantizedDataProvider(TDocPoolDataProviderArgs&& args);

        void Do(IPoolBuilder* poolBuilder) override {
            poolBuilder->Start(PoolMetaInfo, QuantizedPool.DocumentCount, CatFeatures);
            poolBuilder->StartNextBlock(QuantizedPool.DocumentCount);

            if (!PoolMetaInfo.HasDocIds) {
                poolBuilder->GenerateDocIds(/*offset*/ 0);
            }

            size_t baselineIndex = 0;
            size_t floatIndex = 0;
            size_t targetIndex = 0;
            for (const auto& kv : QuantizedPool.ColumnIndexToLocalIndex) {
                const auto localIndex = kv.second;
                const auto columnType = QuantizedPool.ColumnTypes[localIndex];

                if (QuantizedPool.Chunks[localIndex].empty()) {
                    continue;
                }

                CB_ENSURE(columnType == EColumn::Num || columnType == EColumn::Baseline || columnType == EColumn::Label, "Expected Num, Baseline, or Label");
                QuantizedPool.AddColumn(floatIndex, baselineIndex, columnType, localIndex, poolBuilder);

                baselineIndex += (columnType == EColumn::Baseline);
                floatIndex += (columnType == EColumn::Num);
                if (columnType == EColumn::Label) {
                    targetIndex = localIndex;
                }
            }

            poolBuilder->SetFloatFeatures(GetFeatureInfo(QuantizationSchemaFromProto(QuantizedPool.QuantizationSchema), targetIndex));
            SetPairs(PairsPath, PoolMetaInfo.HasGroupWeight, poolBuilder);
            poolBuilder->Finish();
        }

        bool DoBlock(IPoolBuilder* /*poolBuilder*/) override {
            CB_ENSURE(!PairsPath.Inited(), "TAsyncProcDataProviderBase::DoBlock does not support pairs data");
            CB_ENSURE(false, "Quantized pools do not supported on CPU");
            return false;
        }

    protected:
        TVector<bool> FeatureIgnored; // TODO(espetrov): respect in Do()
        TVector<int> CatFeatures;
        TQuantizedPool QuantizedPool;
        TPathWithScheme PairsPath;
        TPoolMetaInfo PoolMetaInfo;
    };

    TCBQuantizedDataProvider::TCBQuantizedDataProvider(TDocPoolDataProviderArgs&& args)
        : PairsPath(args.PairsFilePath)
    {
        CB_ENSURE(!args.PairsFilePath.Inited() || CheckExists(args.PairsFilePath),
            "TCBQuantizedDataProvider:PairsFilePath does not exist");

        NCB::TLoadQuantizedPoolParameters loadParameters;
        loadParameters.LockMemory = false;
        loadParameters.Precharge = false;
        QuantizedPool = NCB::LoadQuantizedPool(args.PoolPath.Path, loadParameters);

        PoolMetaInfo.ColumnsInfo.ConstructInPlace();
        PoolMetaInfo = GetPoolMetaInfo(QuantizedPool);
        CB_ENSURE(PoolMetaInfo.ColumnsInfo.Defined(), "Missing column description");
        PoolMetaInfo.ColumnsInfo->Validate();

        CB_ENSURE(PoolMetaInfo.FeatureCount > 0, "Pool should have at least one factor");
        const int featureCount = static_cast<int>(PoolMetaInfo.FeatureCount);

        FeatureIgnored.resize(featureCount, false);
        int ignoredFeatureCount = 0;
        for (int featureId : args.IgnoredFeatures) { // esp: GetIgnoredFeatureIndices(const NCB::TQuantizedPool& pool)?
            CB_ENSURE(0 <= featureId && featureId < featureCount, "Invalid ignored feature id: " << featureId);
            ignoredFeatureCount += FeatureIgnored[featureId] == false;
            FeatureIgnored[featureId] = true;
        }
        CB_ENSURE(featureCount - ignoredFeatureCount > 0, "All features are requested to be ignored");

        CB_ENSURE(!(PoolMetaInfo.HasWeights && PoolMetaInfo.HasGroupWeight), "Pool must have either Weight column or GroupWeight column");

        CatFeatures = GetCategoricalFeatureIndices(QuantizedPool);
    }

    namespace {
        TDocDataProviderObjectFactory::TRegistrator<TCBDsvDataProvider> DefDataProviderReg("");
        TDocDataProviderObjectFactory::TRegistrator<TCBDsvDataProvider> CBDsvDataProviderReg("dsv");

        TDocDataProviderObjectFactory::TRegistrator<TCBQuantizedDataProvider> CBQuantizedDataProviderReg("quantized");
    }
}

