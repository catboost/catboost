#include "doc_pool_data_provider.h"

#include <catboost/libs/column_description/cd_parser.h>
#include <catboost/libs/data_new/loader.h> // for IsNanValue. TODO(akhropov): to be removed after MLTOOLS-140
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

    static TVector<TPair> ReadPairs(const TPathWithScheme& filePath, ui64 docCount) {
        THolder<ILineDataReader> reader = GetLineDataReader(filePath);

        TVector<TPair> pairs;
        TString line;
        while (reader->ReadLine(&line)) {
            TVector<TString> tokens = StringSplitter(line).Split('\t').ToList<TString>();
            if (tokens.empty()) {
                continue;
            }
            CB_ENSURE(tokens.ysize() == 2 || tokens.ysize() == 3,
                "Each line should have two or three columns. Invalid line number " << line);
            ui64 winnerId = FromString<int>(tokens[0]);
            ui64 loserId = FromString<int>(tokens[1]);
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

    static TVector<float> ReadGroupWeights(
        const TPathWithScheme& filePath,
        TConstArrayRef<TGroupId> groupIds,
        ui64 docCount
    ) {
        CB_ENSURE(groupIds.size() == docCount, "GroupId count should correspond with object count.");
        TVector<float> groupWeights;
        groupWeights.reserve(docCount);
        ui64 groupIdCursor = 0;
        THolder<ILineDataReader> reader = GetLineDataReader(filePath);
        TString line;
        while (reader->ReadLine(&line)) {
            TVector<TString> tokens = StringSplitter(line).Split('\t').ToList<TString>();
            CB_ENSURE(tokens.ysize() == 2,
                "Each line in group weights file should have two columns. Invalid line number " << line);

            const TGroupId groupId = CalcGroupIdFor(tokens[0]);
            const float groupWeight = FromString<float>(tokens[1]);
            ui64 groupSize = 0;
            CB_ENSURE(groupId == groupIds[groupIdCursor],
                "GroupId from the file with group weights do not match GroupId from the dataset.");
            while (groupIdCursor < docCount && groupId == groupIds[groupIdCursor]) {
                ++groupSize;
                ++groupIdCursor;
            }
            groupWeights.insert(groupWeights.end(), groupSize, groupWeight);
        }
        CB_ENSURE(groupWeights.size() == docCount,
            "Group weights file should have as many weights as the objects in the dataset.");

        return groupWeights;
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

    void SetGroupWeights(const TPathWithScheme& groupWeightsPath, IPoolBuilder* poolBuilder) {
        DumpMemUsage("After data read");
        if (groupWeightsPath.Inited()) {
            TVector<float> groupWeights = ReadGroupWeights(
                groupWeightsPath, poolBuilder->GetGroupIds(), poolBuilder->GetDocCount()
            );
            poolBuilder->SetGroupWeights(groupWeights);
        }
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
        CB_ENSURE(!Args.GroupWeightsFilePath.Inited() || CheckExists(Args.GroupWeightsFilePath),
                  "TCBDsvDataProvider:GroupWeightsFilePath does not exist");

        TMaybe<TString> header = LineDataReader->GetHeader();

        TString firstLine;
        CB_ENSURE(LineDataReader->ReadLine(&firstLine), "TCBDsvDataProvider: no data rows in pool");
        const ui32 columnsCount = StringSplitter(firstLine).Split(FieldDelimiter).Count();
        PoolMetaInfo = TPoolMetaInfo(CreateColumnsDescription(columnsCount), Args.GroupWeightsFilePath.Inited());

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
                                          int docCount, int /*offset*/,
                                          IPoolBuilder* poolBuilder)
    {
        poolBuilder->Start(PoolMetaInfo, docCount, CatFeatures);
        if (!FeatureIds.empty()) {
            poolBuilder->SetFeatureIds(FeatureIds);
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
                                poolBuilder->AddLabel(lineIdx, token);
                                break;
                            }
                            case EConvertTargetPolicy::UseClassNames:
                            case EConvertTargetPolicy::CastFloat: {
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
        TVector<TFloatFeature> GetFloatFeatureInfo(int allFeaturesCount, const TQuantizedPool& pool) {
            const auto quantizationSchema = QuantizationSchemaFromProto(pool.QuantizationSchema);
            const auto catFeatureIndices = GetCategoricalFeatureIndices(pool);
            const auto flatFeatureNames = GetFlatFeatureNames(pool);
            const THashSet<int> catFeatureIndicesSet(catFeatureIndices.begin(), catFeatureIndices.end());
            TVector<TFloatFeature> floatFeatures = CreateFloatFeatures(allFeaturesCount, catFeatureIndicesSet, /*featureIds*/ {});
            for (auto& floatFeature : floatFeatures) {
                const auto& flatIndices = quantizationSchema.FeatureIndices;
                const auto localIndex = FindIndex(flatIndices, floatFeature.FlatFeatureIndex);
                if (localIndex == NPOS) {
                    continue;
                }
                const auto& nanModes = quantizationSchema.NanModes;
                floatFeature.HasNans = nanModes[localIndex] != ENanMode::Forbidden;
                const auto& borders = quantizationSchema.Borders;
                floatFeature.Borders = borders[localIndex];
                floatFeature.FeatureId = flatFeatureNames[floatFeature.FlatFeatureIndex];
            }
            return floatFeatures;
        }
    }

    class TCBQuantizedDataProvider final : public IDocPoolDataProvider
    {
    public:
        explicit TCBQuantizedDataProvider(TDocPoolPullDataProviderArgs&& args);

        void Do(IPoolBuilder* poolBuilder) override {
            CB_ENSURE(QuantizedPool.DocumentCount > 0, "Pool is empty");
            poolBuilder->Start(PoolMetaInfo, QuantizedPool.DocumentCount, CatFeatures);
            poolBuilder->SetFeatureIds(GetFlatFeatureNames(QuantizedPool));
            poolBuilder->StartNextBlock(QuantizedPool.DocumentCount);

            size_t baselineIndex = 0;
            const auto columnIndexToFlatIndex = GetColumnIndexToFlatIndexMap(QuantizedPool);
            const auto columnIndexToNumericFeatureIndex = GetColumnIndexToNumericFeatureIndexMap(QuantizedPool);
            for (const auto [columnIndex, localIndex] : QuantizedPool.ColumnIndexToLocalIndex) {
                const auto columnType = QuantizedPool.ColumnTypes[localIndex];

                if (QuantizedPool.Chunks[localIndex].empty()) {
                    continue;
                }

                CB_ENSURE(
                    columnType == EColumn::Num || columnType == EColumn::Baseline ||
                    columnType == EColumn::Label || columnType == EColumn::Categ ||
                    columnType == EColumn::Weight || columnType == EColumn::GroupWeight ||
                    columnType == EColumn::GroupId || columnType == EColumn::SubgroupId,
                    "Expected Num, Baseline, Label, Categ, Weight, GroupWeight, GroupId, or Subgroupid; got "
                    LabeledOutput(columnType, columnIndex));

                const auto it = columnIndexToNumericFeatureIndex.find(columnIndex);
                if (it == columnIndexToNumericFeatureIndex.end()) {
                    QuantizedPool.AddColumn(/*unused featureIndex*/ -1, baselineIndex, columnType, localIndex, poolBuilder);
                    baselineIndex += (columnType == EColumn::Baseline);
                    continue;
                } else if (!IsFeatureIgnored[columnIndexToFlatIndex.at(columnIndex)]) {
                    QuantizedPool.AddColumn(it->second, baselineIndex, columnType, localIndex, poolBuilder);
                }
            }

            poolBuilder->SetFloatFeatures(GetFloatFeatureInfo(PoolMetaInfo.FeatureCount, QuantizedPool));
            QuantizedPool = TQuantizedPool(); // release memory
            SetGroupWeights(GroupWeightsPath, poolBuilder);
            SetPairs(PairsPath, PoolMetaInfo.HasGroupWeight, poolBuilder);
            poolBuilder->Finish();
        }

        bool DoBlock(IPoolBuilder* /*poolBuilder*/) override {
            CB_ENSURE(false, "Quantized pools do not support reading by blocks");
            return false;
        }

    protected:
        static TLoadQuantizedPoolParameters GetLoadParameters() {
            return {/*LockMemory*/ false, /*Precharge*/ false};
        }

        TVector<bool> IsFeatureIgnored;
        TVector<int> CatFeatures;
        TQuantizedPool QuantizedPool;
        TPathWithScheme PairsPath;
        TPathWithScheme GroupWeightsPath;
        TPoolMetaInfo PoolMetaInfo;
    };

    TCBQuantizedDataProvider::TCBQuantizedDataProvider(TDocPoolPullDataProviderArgs&& args)
        : QuantizedPool(std::forward<TQuantizedPool>(LoadQuantizedPool(args.PoolPath.Path, GetLoadParameters())))
        , PairsPath(args.CommonArgs.PairsFilePath)
        , GroupWeightsPath(args.CommonArgs.GroupWeightsFilePath)
    {
        CB_ENSURE(!PairsPath.Inited() || CheckExists(PairsPath),
            "TCBQuantizedDataProvider:PairsFilePath does not exist");
        CB_ENSURE(!GroupWeightsPath.Inited() || CheckExists(GroupWeightsPath),
            "TCBQuantizedDataProvider:GroupWeightsFilePath does not exist");

        PoolMetaInfo = GetPoolMetaInfo(QuantizedPool, GroupWeightsPath.Inited());
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
        for (int featureId : GetIgnoredFlatIndices(QuantizedPool)) {
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

