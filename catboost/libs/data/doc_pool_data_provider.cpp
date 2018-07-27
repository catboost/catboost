
#include "doc_pool_data_provider.h"

#include "load_data.h"
#include "load_helpers.h"

#include <catboost/libs/column_description/cd_parser.h>

#include <catboost/libs/data_util/exists_checker.h>

#include <catboost/libs/helpers/mem_usage.h>

#include <library/object_factory/object_factory.h>

#include <util/generic/maybe.h>
#include <util/generic/strbuf.h>
#include <util/generic/vector.h>

#include <util/string/cast.h>
#include <util/string/iterator.h>
#include <util/string/split.h>

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


    namespace {

    ui32 CountColumns(const TVector<TColumn>& columnsDescription, const EColumn columnType) {
        return CountIf(
            columnsDescription.begin(),
            columnsDescription.end(),
            [&columnType](const auto x) -> bool {
                return x.Type == columnType;
            }
        );
    }


    TVector<int> GetCategFeatures(const TVector<TColumn>& columns) {
        Y_ASSERT(!columns.empty());
        TVector<int> categFeatures;
        int featureId = 0;
        for (const TColumn& column : columns) {
            switch (column.Type) {
                case EColumn::Categ:
                    categFeatures.push_back(featureId);
                    ++featureId;
                    break;
                case EColumn::Num:
                    ++featureId;
                    break;
                case EColumn::Auxiliary:
                case EColumn::Label:
                case EColumn::Baseline:
                case EColumn::Weight:
                case EColumn::DocId:
                case EColumn::GroupId:
                case EColumn::GroupWeight:
                case EColumn::SubgroupId:
                case EColumn::Timestamp:
                    break;
                default:
                    CB_ENSURE(false, "this column type is not supported");
            }
        }
        return categFeatures;
    }

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

        auto& columnsDescription = PoolMetaInfo.ColumnsInfo->Columns;
        columnsDescription = CreateColumnsDescription(columnsCount);

        const ui32 weightColumns = CountColumns(columnsDescription, EColumn::Weight);
        CB_ENSURE(weightColumns <= 1, "Too many Weight columns.");
        PoolMetaInfo.HasWeights = (bool)weightColumns;

        PoolMetaInfo.BaselineCount = CountColumns(columnsDescription, EColumn::Baseline);

        CB_ENSURE(CountColumns(columnsDescription, EColumn::Label) <= 1, "Too many Label columns.");

        const ui32 docIdColumns = CountColumns(columnsDescription, EColumn::DocId);
        CB_ENSURE(docIdColumns <= 1, "Too many DocId columns.");
        PoolMetaInfo.HasDocIds = (bool)docIdColumns;

        const ui32 groupIdColumns = CountColumns(columnsDescription, EColumn::GroupId);
        CB_ENSURE(groupIdColumns <= 1, "Too many GroupId columns. Maybe you've specified QueryId and GroupId, QueryId is synonym for GroupId.");
        PoolMetaInfo.HasGroupId = (bool)groupIdColumns;

        const ui32 groupWeightColumns = CountColumns(columnsDescription, EColumn::GroupWeight);
        CB_ENSURE(groupWeightColumns <= 1, "Too many GroupWeight columns.");
        PoolMetaInfo.HasGroupWeight = (bool)groupWeightColumns;

        const ui32 subgroupIdColumns = CountColumns(columnsDescription, EColumn::SubgroupId);
        CB_ENSURE(subgroupIdColumns <= 1, "Too many SubgroupId columns.");
        PoolMetaInfo.HasSubgroupIds = (bool)subgroupIdColumns;

        const ui32 timestampColumns = CountColumns(columnsDescription, EColumn::Timestamp);
        CB_ENSURE(timestampColumns <= 1, "Too many Timestamp columns.");
        PoolMetaInfo.HasTimestamp = (bool)timestampColumns;

        PoolMetaInfo.FeatureCount = (const ui32)CountIf(
            columnsDescription.begin(),
            columnsDescription.end(),
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

        CatFeatures = GetCategFeatures(columnsDescription);

        InitFeatureIds(header);

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


    // call after PoolMetaInfo.ColumnsInfo->Columns initialization
    void TCBDsvDataProvider::InitFeatureIds(const TMaybe<TString>& header) {
        auto& columnsDescription = PoolMetaInfo.ColumnsInfo->Columns;

        // TODO: this convoluted logic is for compatibility
        if (!AllOf(columnsDescription.begin(), columnsDescription.end(), [](const TColumn& column) {
            return column.Id.empty();
        })) {
            for (auto column : columnsDescription) {
                if (column.Type == EColumn::Categ || column.Type == EColumn::Num) {
                    FeatureIds.push_back(column.Id);
                }
            }
        } else if (header.Defined()) {
            TVector<TStringBuf> words;
            SplitRangeTo<const char, TVector<TStringBuf>>(~(*header), ~(*header) + header->size(), FieldDelimiter, &words);
            for (int i = 0; i < words.ysize(); ++i) {
                if (columnsDescription[i].Type == EColumn::Categ || columnsDescription[i].Type == EColumn::Num) {
                    FeatureIds.push_back(ToString(words[i]));
                }
            }
        }
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

    namespace {

    TDocDataProviderObjectFactory::TRegistrator<TCBDsvDataProvider> DefDataProviderReg("");
    TDocDataProviderObjectFactory::TRegistrator<TCBDsvDataProvider> CBDsvDataProviderReg("dsv");

    }
}

