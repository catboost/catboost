#include "baseline.h"
#include "cb_dsv_loader.h"

#include <catboost/libs/column_description/cd_parser.h>
#include <catboost/private/libs/data_util/exists_checker.h>
#include <catboost/private/libs/labels/helpers.h>
#include <catboost/libs/helpers/mem_usage.h>

#include <library/object_factory/object_factory.h>
#include <library/string_utils/csv/csv.h>

#include <util/generic/strbuf.h>
#include <util/generic/vector.h>
#include <util/stream/file.h>
#include <util/string/split.h>
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
        , CsvSplitterQuote(Args.PoolFormat.IgnoreCsvQuoting ? '\0' : '"')
        , LineDataReader(std::move(args.Reader))
        , BaselineReader(Args.BaselineFilePath, ClassLabelsToStrings(args.CommonArgs.ClassLabels))
    {
        CB_ENSURE(!Args.PairsFilePath.Inited() || CheckExists(Args.PairsFilePath),
                  "TCBDsvDataLoader:PairsFilePath does not exist");
        CB_ENSURE(!Args.GroupWeightsFilePath.Inited() || CheckExists(Args.GroupWeightsFilePath),
                  "TCBDsvDataLoader:GroupWeightsFilePath does not exist");
        CB_ENSURE(!Args.BaselineFilePath.Inited() || CheckExists(Args.BaselineFilePath),
                  "TCBDsvDataLoader:BaselineFilePath does not exist");
        CB_ENSURE(!Args.TimestampsFilePath.Inited() || CheckExists(Args.TimestampsFilePath),
                  "TCBDsvDataLoader:TimestampsFilePath does not exist");
        CB_ENSURE(!Args.FeatureNamesPath.Inited() || CheckExists(Args.FeatureNamesPath),
                  "TCBDsvDataLoader:FeatureNamesPath does not exist");

        TMaybe<TString> header = LineDataReader->GetHeader();
        TMaybe<TVector<TString>> headerColumns;
        if (header) {
            headerColumns = TVector<TString>(NCsvFormat::CsvSplitter(*header, FieldDelimiter, CsvSplitterQuote));
        }

        TString firstLine;
        CB_ENSURE(LineDataReader->ReadLine(&firstLine), "TCBDsvDataLoader: no data rows in pool");
        const ui32 columnsCount = TVector<TString>(NCsvFormat::CsvSplitter(firstLine, FieldDelimiter, CsvSplitterQuote)).size();

        auto columnsDescription = TDataColumnsMetaInfo{ CreateColumnsDescription(columnsCount) };
        auto targetCount = columnsDescription.CountColumns(EColumn::Label);

        const TVector<TString> featureNames = GetFeatureNames(
            columnsDescription,
            headerColumns,
            Args.FeatureNamesPath
        );

        DataMetaInfo = TDataMetaInfo(
            std::move(columnsDescription),
            targetCount ? ERawTargetType::String : ERawTargetType::None,
            Args.GroupWeightsFilePath.Inited(),
            Args.TimestampsFilePath.Inited(),
            Args.PairsFilePath.Inited(),
            BaselineReader.GetBaselineCount(),
            &featureNames,
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
        if (BaselineReader.Inited()) {
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


    void TCBDsvDataLoader::ProcessBlock(IRawObjectsOrderDataVisitor* visitor) {
        visitor->StartNextBlock(AsyncRowProcessor.GetParseBufferSize());

        auto& columnsDescription = DataMetaInfo.ColumnsInfo->Columns;

        auto parseBlock = [&](TString& line, int lineIdx) {
            const auto& featuresLayout = *DataMetaInfo.FeaturesLayout;

            ui32 featureId = 0;
            ui32 targetId = 0;
            ui32 baselineIdx = 0;

            TVector<float> floatFeatures;
            floatFeatures.yresize(featuresLayout.GetFloatFeatureCount());

            TVector<ui32> catFeatures;
            catFeatures.yresize(featuresLayout.GetCatFeatureCount());

            TVector<TString> textFeatures;
            textFeatures.yresize(featuresLayout.GetTextFeatureCount());

            size_t tokenIdx = 0;
            try {
                const bool floatFeaturesOnly = catFeatures.empty() && textFeatures.empty();
                auto splitter = NCsvFormat::CsvSplitter(line, FieldDelimiter, floatFeaturesOnly ? '\0' : CsvSplitterQuote);
                do {
                    TStringBuf token = splitter.Consume();
                    CB_ENSURE(
                        tokenIdx < columnsDescription.size(),
                        "wrong column count: found more than " << columnsDescription.ysize() << " values"
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
                            case EColumn::Num: {
                                if (!FeatureIgnored[featureId]) {
                                    if (!TryParseFloatFeatureValue(
                                            token,
                                            &floatFeatures[featuresLayout.GetInternalFeatureIdx(featureId)]
                                         ))
                                    {
                                        CB_ENSURE(
                                            false,
                                            "Factor " << featureId << " cannot be parsed as float."
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
                                visitor->AddGroupId(lineIdx, CalcGroupIdFor(token));
                                break;
                            }
                            case EColumn::GroupWeight: {
                                CB_ENSURE(token.length() != 0, "empty values not supported for GroupWeight");
                                visitor->AddGroupWeight(lineIdx, FromString<float>(token));
                                break;
                            }
                            case EColumn::SubgroupId: {
                                CB_ENSURE(token.length() != 0, "empty values not supported for SubgroupId");
                                visitor->AddSubgroupId(lineIdx, CalcSubgroupIdFor(token));
                                break;
                            }
                            case EColumn::Baseline: {
                                CB_ENSURE(token.length() != 0, "empty values not supported for Baseline");
                                visitor->AddBaseline(lineIdx, baselineIdx, FromString<float>(token));
                                ++baselineIdx;
                                break;
                            }
                            case EColumn::SampleId: {
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
            } catch (yexception& e) {
                throw TCatBoostException() << "Error in dsv data. Line " <<
                    AsyncRowProcessor.GetLinesProcessed() + lineIdx + 1 << ": " << e.what();
            }
        };

        AsyncRowProcessor.ProcessBlock(parseBlock);

        if (BaselineReader.Inited()) {
            auto parseBaselineBlock = [&](TString &line, int inBlockIdx) {

                auto addBaselineFunc = [&visitor, inBlockIdx](ui32 baselineIdx, float baseline) {
                    visitor->AddBaseline(inBlockIdx, baselineIdx, baseline);
                };
                const auto lineIdx = AsyncBaselineRowProcessor.GetLinesProcessed() + inBlockIdx + 1;

                BaselineReader.Parse(addBaselineFunc, line, lineIdx);
            };

            AsyncBaselineRowProcessor.ProcessBlock(parseBaselineBlock);
        }
    }

    namespace {
        TDatasetLoaderFactory::TRegistrator<TCBDsvDataLoader> DefDataLoaderReg("");
        TDatasetLoaderFactory::TRegistrator<TCBDsvDataLoader> CBDsvDataLoaderReg("dsv");
    }
}

