#include "doc_pool_data_provider.h"

#include <catboost/libs/column_description/cd_parser.h>
#include <catboost/libs/data_util/exists_checker.h>
#include <catboost/libs/helpers/mem_usage.h>
#include <catboost/libs/quantization_schema/schema.h>
#include <catboost/libs/quantization_schema/serialization.h>
#include <catboost/libs/quantized_pool/pool.h>
#include <catboost/libs/quantized_pool/quantized.h>
#include <catboost/libs/quantized_pool/serialization.h>

#include <library/object_factory/object_factory.h>

#include <util/generic/maybe.h>
#include <util/generic/strbuf.h>
#include <util/generic/vector.h>
#include <util/stream/file.h>
#include <util/string/iterator.h>
#include <util/string/split.h>
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


    TCBDsvDataProvider::TCBDsvDataProvider(TDocPoolPullDataProviderArgs&& args)
        : TCBDsvDataProvider(
            TDocPoolPushDataProviderArgs {
                GetLineDataReader(args.PoolPath, args.CommonArgs.PoolFormat),
                std::move(args.CommonArgs)
            }
        )
    {
    }

    TCBDsvDataProvider::TCBDsvDataProvider(TDocPoolPushDataProviderArgs&& args)
        : TAsyncProcDataProviderBase<TString>(std::move(args.CommonArgs))
        , FieldDelimiter(Args.PoolFormat.Delimiter)
        , LineDataReader(std::move(args.PoolReader))
    {
        CB_ENSURE(!Args.PairsFilePath.Inited() || CheckExists(Args.PairsFilePath),
                  "TCBDsvDataProvider:PairsFilePath does not exist");

        TMaybe<TString> header = LineDataReader->GetHeader();

        TString firstLine;
        CB_ENSURE(LineDataReader->ReadLine(&firstLine), "TCBDsvDataProvider: no data rows in pool");
        const ui32 columnsCount = StringSplitter(firstLine).Split(FieldDelimiter).Count();
        PoolMetaInfo = TPoolMetaInfo(CreateColumnsDescription(columnsCount));

        AsyncRowProcessor.AddFirstLine(std::move(firstLine));

        int featureCount = static_cast<int>(PoolMetaInfo.FeatureCount);
        int ignoredFeatureCount = 0;
        FeatureIgnored.resize(featureCount, false);
        for (int featureId : Args.IgnoredFeatures) {
            CB_ENSURE(0 <= featureId, "Invalid ignored feature id: " << featureId);
            if (featureId >= featureCount) {
                continue;
            }
            ignoredFeatureCount += FeatureIgnored[featureId] == false;
            FeatureIgnored[featureId] = true;
        }
        CB_ENSURE(featureCount - ignoredFeatureCount > 0, "All features are requested to be ignored");

        const auto& columnsInfo = PoolMetaInfo.ColumnsInfo;
        CatFeatures = columnsInfo->GetCategFeatures();
        FeatureIds = columnsInfo->GenerateFeatureIds(header, FieldDelimiter);

        AsyncRowProcessor.ReadBlockAsync(GetReadFunc());
    }

    TVector<TColumn> TCBDsvDataProvider::CreateColumnsDescription(ui32 columnsCount) {
        return Args.CdProvider->GetColumnsDescription(columnsCount);
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
            EConvertTargetPolicy targetPolicy = TargetConverter->GetTargetPolicy();
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
                        CB_ENSURE(token.length() != 0, "empty values not supported for Label");
                        switch (targetPolicy) {
                            case EConvertTargetPolicy::MakeClassNames: {
                                CB_ENSURE(!IsOnlineTargetProcessing,
                                          "Cannot process target online with offline processing policy.");
                                IsOfflineTargetProcessing = true;
                                poolBuilder->AddLabel(lineIdx, token);
                                break;
                            }
                            case EConvertTargetPolicy::UseClassNames:
                            case EConvertTargetPolicy::CastFloat: {
                                CB_ENSURE(!IsOfflineTargetProcessing,
                                          "Cannot process target offline with online processing policy.");
                                IsOnlineTargetProcessing = true;
                                poolBuilder->AddTarget(
                                    lineIdx,
                                    TargetConverter->ConvertLabel(token)
                                );
                                break;
                            }
                            default: {
                                CB_ENSURE(false, "Unsupported convert target policy "
                                                 << ToString<EConvertTargetPolicy>(targetPolicy));
                            }
                        }
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

        int GetTargetIndex(const TVector<EColumn>& columnTypes) {
            const auto targetIndex = FindIndex(columnTypes, EColumn::Label);
            if (targetIndex == NPOS) {
                return 0;
            } else {
                return targetIndex;
            }
        }
    }
    class TCBQuantizedDataProvider : public IDocPoolDataProvider
    {
    public:
        explicit TCBQuantizedDataProvider(TDocPoolPullDataProviderArgs&& args);

        void Do(IPoolBuilder* poolBuilder) override {
            poolBuilder->Start(PoolMetaInfo, QuantizedPool.DocumentCount, CatFeatures);
            poolBuilder->StartNextBlock(QuantizedPool.DocumentCount);

            if (!PoolMetaInfo.HasDocIds) {
                poolBuilder->GenerateDocIds(/*offset*/ 0);
            }

            size_t baselineIndex = 0;
            size_t nonFloatColumnsCount = 0;
            for (const auto& kv : QuantizedPool.ColumnIndexToLocalIndex) {
                const auto localIndex = kv.second;
                const auto columnType = QuantizedPool.ColumnTypes[localIndex];

                if (QuantizedPool.Chunks[localIndex].empty()) {
                    nonFloatColumnsCount += (columnType != EColumn::Num);
                    continue;
                }

                CB_ENSURE(columnType == EColumn::Num || columnType == EColumn::Baseline || columnType == EColumn::Label || columnType == EColumn::Categ || columnType == EColumn::GroupId,
                    "Expected Num, Baseline, Label, or Categ; got " << columnType << " for column " << kv.first);
                if (!IsFeatureIgnored[kv.first - nonFloatColumnsCount]) {
                    QuantizedPool.AddColumn(kv.first - nonFloatColumnsCount, baselineIndex, columnType, localIndex, poolBuilder);
                }

                baselineIndex += (columnType == EColumn::Baseline);
                nonFloatColumnsCount += (columnType != EColumn::Num);
            }

            poolBuilder->SetFloatFeatures(GetFeatureInfo(QuantizationSchemaFromProto(QuantizedPool.QuantizationSchema), GetTargetIndex(QuantizedPool.ColumnTypes)));
            SetPairs(PairsPath, PoolMetaInfo.HasGroupWeight, poolBuilder);
            poolBuilder->Finish();
        }

        bool DoBlock(IPoolBuilder* /*poolBuilder*/) override {
            CB_ENSURE(false, "Quantized pools do not support reading by blocks");
            return false;
        }

    protected:
        TVector<bool> IsFeatureIgnored;
        TVector<int> CatFeatures;
        TQuantizedPool QuantizedPool;
        TPathWithScheme PairsPath;
        TPoolMetaInfo PoolMetaInfo;
    };

    TCBQuantizedDataProvider::TCBQuantizedDataProvider(TDocPoolPullDataProviderArgs&& args)
        : PairsPath(args.CommonArgs.PairsFilePath)
    {
        CB_ENSURE(!args.CommonArgs.PairsFilePath.Inited() || CheckExists(args.CommonArgs.PairsFilePath),
            "TCBQuantizedDataProvider:PairsFilePath does not exist");

        NCB::TLoadQuantizedPoolParameters loadParameters;
        loadParameters.LockMemory = false;
        loadParameters.Precharge = false;
        QuantizedPool = NCB::LoadQuantizedPool(args.PoolPath.Path, loadParameters);

        PoolMetaInfo = GetPoolMetaInfo(QuantizedPool);
        CB_ENSURE(PoolMetaInfo.ColumnsInfo.Defined(), "Missing column description");
        PoolMetaInfo.ColumnsInfo->Validate();

        CB_ENSURE(PoolMetaInfo.FeatureCount > 0, "Pool should have at least one factor");
        const int featureCount = static_cast<int>(PoolMetaInfo.FeatureCount);

        IsFeatureIgnored.resize(featureCount, false);
        int ignoredFeatureCount = 0;
        for (int featureId : args.CommonArgs.IgnoredFeatures) {
            CB_ENSURE(0 <= featureId && featureId < featureCount, "Invalid ignored feature id: " << featureId);
            ignoredFeatureCount += IsFeatureIgnored[featureId] == false;
            IsFeatureIgnored[featureId] = true;
        }
        const int targetColumn = GetTargetIndex(QuantizedPool.ColumnTypes);
        const auto ignoredColumns = GetIgnoredFeatureIndices(QuantizedPool);
        for (int column : ignoredColumns) {
            if (column == targetColumn) { // TODO(yazevnul): do not add target to ignored features
                continue;
            }
            const int featureId = column - (column > targetColumn); // do not count target column
            CB_ENSURE(0 <= featureId && featureId < featureCount, "Invalid ignored feature id: " << featureId);
            ignoredFeatureCount += IsFeatureIgnored[featureId] == false;
            IsFeatureIgnored[featureId] = true;
        }
        CB_ENSURE(featureCount - ignoredFeatureCount > 0, "All features are requested to be ignored");

        CatFeatures = GetCategoricalFeatureIndices(QuantizedPool);
    }

    namespace {
        TDocDataProviderObjectFactory::TRegistrator<TCBDsvDataProvider> DefDataProviderReg("");
        TDocDataProviderObjectFactory::TRegistrator<TCBDsvDataProvider> CBDsvDataProviderReg("dsv");

        TDocDataProviderObjectFactory::TRegistrator<TCBQuantizedDataProvider> CBQuantizedDataProviderReg("quantized");
    }
}

