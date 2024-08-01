#include "libsvm_loader.h"

#include "features_layout.h"
#include "sampler.h"

#include <catboost/libs/column_description/cd_parser.h>
#include <catboost/libs/helpers/mem_usage.h>
#include <catboost/libs/helpers/sparse_array.h>
#include <catboost/private/libs/data_util/exists_checker.h>
#include <catboost/private/libs/data_util/line_data_reader.h>
#include <catboost/private/libs/labels/helpers.h>
#include <catboost/private/libs/options/pool_metainfo_options.h>

#include <library/cpp/object_factory/object_factory.h>

#include <util/generic/algorithm.h>
#include <util/generic/maybe.h>
#include <util/generic/strbuf.h>
#include <util/generic/utility.h>
#include <util/generic/vector.h>
#include <util/generic/xrange.h>
#include <util/stream/file.h>
#include <util/string/split.h>
#include <util/system/compiler.h>
#include <util/system/guard.h>
#include <util/system/types.h>


static TVector<TString> GetFeatureNames(
    const TVector<TString>& featureNamesFromColumnsDescription,
    const NCB::TPathWithScheme& featureNamesPath
) {
    TVector<TString> externalFeatureNames = LoadFeatureNames(featureNamesPath);

    if (externalFeatureNames.empty()) {
        return featureNamesFromColumnsDescription;
    } else {
        const size_t intersectionSize = Min(
            featureNamesFromColumnsDescription.size(),
            externalFeatureNames.size());

        size_t featureIdx = 0;
        for (; featureIdx < intersectionSize; ++featureIdx) {
            CB_ENSURE(
                featureNamesFromColumnsDescription[featureIdx].empty()
                || (featureNamesFromColumnsDescription[featureIdx] == externalFeatureNames[featureIdx]),
                "Feature #" << featureIdx << ": name from columns description (\""
                << featureNamesFromColumnsDescription[featureIdx]
                << "\") is not equal to name from feature names file (\""
                << externalFeatureNames[featureIdx] << "\")"
            );
        }
        for (; featureIdx < featureNamesFromColumnsDescription.size(); ++featureIdx) {
            CB_ENSURE(
                featureNamesFromColumnsDescription[featureIdx].empty(),
                "Feature #" << featureIdx << ": name specified in columns description (\""
                << featureNamesFromColumnsDescription[featureIdx]
                << "\") but not present in feature names file"
            );
        }

        return externalFeatureNames;
    }
}


namespace NCB {

    TLibSvmDataLoader::TLibSvmDataLoader(TDatasetLoaderPullArgs&& args)
        : TLibSvmDataLoader(
            TLineDataLoaderPushArgs {
                GetLineDataReader(args.PoolPath, args.CommonArgs.PoolFormat),
                std::move(args.CommonArgs)
            }
        )
    {
    }

    TLibSvmDataLoader::TLibSvmDataLoader(TLineDataLoaderPushArgs&& args)
        : TAsyncProcDataLoaderBase<TString>(std::move(args.CommonArgs))
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
            "TLibSvmDataLoader:PairsFilePath does not exist");
        CB_ENSURE(
            !Args.GraphFilePath.Inited() || CheckExists(Args.GraphFilePath),
            "TLibSvmDataLoader:GraphFilePath does not exist");
        CB_ENSURE(
            !Args.GroupWeightsFilePath.Inited() || CheckExists(Args.GroupWeightsFilePath),
            "TLibSvmDataLoader:GroupWeightsFilePath does not exist"
        );
        CB_ENSURE(
            !Args.TimestampsFilePath.Inited() || CheckExists(Args.TimestampsFilePath),
            "TLibSvmDataLoader:TimestampsFilePath does not exist"
        );
        CB_ENSURE(
            !Args.FeatureNamesPath.Inited() || CheckExists(Args.FeatureNamesPath),
            "TLibSvmDataLoader:FeatureNamesPath does not exist"
        );
        CB_ENSURE(
            !Args.PoolMetaInfoPath.Inited() || CheckExists(Args.PoolMetaInfoPath),
            "TLibSvmDataLoader:PoolMetaInfoPath does not exist"
        );

        TString firstLine;
        CB_ENSURE(LineDataReader->ReadLine(&firstLine), "TLibSvmDataLoader: no data rows");

        DataMetaInfo.TargetType = ERawTargetType::Float;
        DataMetaInfo.TargetCount = 1;
        DataMetaInfo.BaselineCount = BaselineReader ? BaselineReader->GetBaselineCount() : 0;
        DataMetaInfo.HasGroupId = DataHasGroupId(firstLine);
        DataMetaInfo.HasGroupWeight = Args.GroupWeightsFilePath.Inited();
        DataMetaInfo.HasPairs = Args.PairsFilePath.Inited();
        DataMetaInfo.HasTimestamp = Args.TimestampsFilePath.Inited();

        AsyncRowProcessor.AddFirstLine(std::move(firstLine));

        TVector<ui32> catFeatures;
        TVector<TString> featureNamesFromColumns;
        if (Args.CdProvider->Inited()) {
            ProcessCdData(&catFeatures, &featureNamesFromColumns);
        }

        const TVector<TString> featureNames = GetFeatureNames(featureNamesFromColumns, Args.FeatureNamesPath);
        const auto poolMetaInfoOptions = NCatboostOptions::LoadPoolMetaInfoOptions(Args.PoolMetaInfoPath);

        auto featuresLayout = MakeIntrusive<TFeaturesLayout>(
            (ui32)featureNames.size(),
            catFeatures,
            /*textFeatures*/ TVector<ui32>{},
            /*embeddingFeatures*/ TVector<ui32>{},
            featureNames,
            /*graph*/ false,
            poolMetaInfoOptions.Tags.Get(),
            /*allFeaturesAreSparse*/ true
        );

        ProcessIgnoredFeaturesListWithUnknownFeaturesCount(
            Args.IgnoredFeatures,
            featuresLayout.Get(),
            &FeatureIgnored
        );

        DataMetaInfo.FeaturesLayout = std::move(featuresLayout);

        AsyncRowProcessor.ReadBlockAsync(GetReadFunc());
        if (BaselineReader) {
            AsyncBaselineRowProcessor.ReadBlockAsync(GetReadBaselineFunc());
        }
    }

    ui32 TLibSvmDataLoader::GetObjectCountSynchronized() {
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

    void TLibSvmDataLoader::StartBuilder(
        bool inBlock,
        ui32 objectCount,
        ui32 /*offset*/,
        IRawObjectsOrderDataVisitor* visitor)
    {
        visitor->Start(
            inBlock,
            DataMetaInfo,
            /*haveUnknownNumberOfSparseFeatures*/ true,
            objectCount,
            Args.ObjectsOrder,
            {}
        );

        const auto& featuresLayout = *DataMetaInfo.FeaturesLayout;
        for (auto catFeatureExternalIdx : featuresLayout.GetCatFeatureInternalIdxToExternalIdx()) {
            visitor->AddCatFeatureDefaultValue(catFeatureExternalIdx, TStringBuf("0"));
        }
    }


    void TLibSvmDataLoader::ProcessBlock(IRawObjectsOrderDataVisitor* visitor) {
        visitor->StartNextBlock(AsyncRowProcessor.GetParseBufferSize());

        auto parseBlock = [&](TString& line, int lineIdx) {
            const auto& featuresLayout = *DataMetaInfo.FeaturesLayout;

            TConstArrayRef<TFeatureMetaInfo> featuresMetaInfo = featuresLayout.GetExternalFeaturesMetaInfo();

            TVector<ui32> floatFeatureIndices;
            floatFeatureIndices.reserve(featuresLayout.GetFloatFeatureCount());
            TVector<float> floatFeatureValues;
            floatFeatureValues.reserve(featuresLayout.GetFloatFeatureCount());

            TVector<ui32> catFeatureIndices;
            catFeatureIndices.reserve(featuresLayout.GetCatFeatureCount());
            TVector<ui32> catFeatureValues;
            catFeatureValues.reserve(featuresLayout.GetCatFeatureCount());

            try {
                auto lineSplitter = StringSplitter(line).Split(' ');
                auto lineIterator = lineSplitter.begin();
                auto lineEndIterator = lineSplitter.end();

                size_t tokenCount = 0;
                TStringBuf token;
                ui32 lastFeatureIdxPlus1 = 0; // +1 to allow to compare first featureIdx
                try {
                    CB_ENSURE(lineIterator != lineEndIterator, "line is empty");
                    token = (*lineIterator).Token();

                    CB_ENSURE(token.length() != 0, "empty values not supported for Label");
                    float label;
                    CB_ENSURE(TryFromString(token, label), "Target value must be float");
                    visitor->AddTarget(lineIdx, label);

                    ++tokenCount;
                    ++lineIterator;

                    if (DataMetaInfo.HasGroupId) {
                        CB_ENSURE(lineIterator != lineEndIterator, "line does not contain 'qid' field");
                        token = (*lineIterator).Token();

                        TStringBuf left;
                        TStringBuf right;
                        token.Split(':', left, right);

                        CB_ENSURE(left == "qid"sv, "line does not contain 'qid' field");
                        TGroupId groupId;
                        CB_ENSURE(TryFromString(right, groupId), "'qid' value must be integer");
                        visitor->AddGroupId(lineIdx, groupId);

                        ++tokenCount;
                        ++lineIterator;
                    }

                    for (; lineIterator != lineEndIterator; ++lineIterator, ++tokenCount) {
                        token = (*lineIterator).Token();
                        // allow extra space at the end of line
                        if (token.empty() && std::next(lineIterator) == lineEndIterator) {
                            continue;
                        }

                        TStringBuf left;
                        TStringBuf right;
                        token.Split(':', left, right);

                        ui32 featureIdx;
                        CB_ENSURE(
                            TryFromString(left, featureIdx) && featureIdx,
                            "Feature index must be a positive integer"
                        );
                        CB_ENSURE(
                            featureIdx > lastFeatureIdxPlus1,
                            "Feature indices must be ascending"
                        );
                        --featureIdx; // in libsvm format indices start from 1

                        if ((featureIdx >= FeatureIgnored.size()) || !FeatureIgnored[featureIdx]) {

                            if ((featureIdx < featuresMetaInfo.size()) &&
                                (featuresMetaInfo[featureIdx].Type == EFeatureType::Categorical))
                            {
                                const ui32 catFeatureIdx = featuresLayout.GetInternalFeatureIdx(featureIdx);
                                catFeatureIndices.push_back(catFeatureIdx);
                                catFeatureValues.push_back(visitor->GetCatFeatureValue(lineIdx, featureIdx, right));
                            } else {
                                const TFloatFeatureIdx floatFeatureIdx
                                    = featuresLayout.GetExpandingInternalFeatureIdx<EFeatureType::Float>(
                                        featureIdx
                                    );
                                floatFeatureIndices.push_back(*floatFeatureIdx);
                                floatFeatureValues.yresize(floatFeatureIndices.size());

                                if (!TryFloatFromString(right, /*parseNonFinite*/true, &floatFeatureValues.back())) {
                                    CB_ENSURE(
                                        false,
                                        "Feature value \"" << right << "\" cannot be parsed as float."
                                    );
                                }
                            }
                        }
                    }
                } catch (yexception& e) {
                    throw TCatBoostException() << "Column " << tokenCount << " (value = \""
                        << token << "\"): " << e.what();
                }

                if (!floatFeatureIndices.empty()) {
                    const ui32 floatFeatureCount = floatFeatureIndices.back() + 1;
                    visitor->AddAllFloatFeatures(
                        lineIdx,
                        MakeConstPolymorphicValuesSparseArrayWithArrayIndex<float, float, ui32>(
                            /*size*/ floatFeatureCount,
                            TMaybeOwningConstArrayHolder<ui32>::CreateOwning(std::move(floatFeatureIndices)),
                            TMaybeOwningConstArrayHolder<float>::CreateOwning(std::move(floatFeatureValues))
                        )
                    );
                }

                if (!catFeatureIndices.empty()) {
                    const ui32 catFeatureCount = catFeatureIndices.back() + 1;
                    visitor->AddAllCatFeatures(
                        lineIdx,
                        MakeConstPolymorphicValuesSparseArrayWithArrayIndex<ui32, ui32, ui32>(
                            /*size*/ catFeatureCount,
                            TMaybeOwningConstArrayHolder<ui32>::CreateOwning(std::move(catFeatureIndices)),
                            TMaybeOwningConstArrayHolder<ui32>::CreateOwning(std::move(catFeatureValues))
                        )
                    );
                }

            } catch (yexception& e) {
                throw TCatBoostException() << "Error in libsvm data. Line " <<
                    AsyncRowProcessor.GetLinesProcessed() + lineIdx + 1 << ": " << e.what();
            }
        };

        AsyncRowProcessor.ProcessBlock(parseBlock);

        if (BaselineReader) {
            auto setBaselineBlock = [&](TObjectBaselineData &data, int inBlockIdx) {
                for (auto baselineIdx : xrange(data.Baseline.size())) {
                    visitor->AddBaseline(inBlockIdx, baselineIdx, data.Baseline[baselineIdx]);
                }
            };

            AsyncBaselineRowProcessor.ProcessBlock(setBaselineBlock);
        }
    }

    void TLibSvmDataLoader::ProcessIgnoredFeaturesListWithUnknownFeaturesCount(
        TConstArrayRef<ui32> ignoredFeatures,
        TFeaturesLayout* featuresLayout,
        TVector<bool>* ignoredFeaturesMask
    ) {
        for (auto ignoredFeatureIdx : ignoredFeatures) {
            auto existingFeatureCount = featuresLayout->GetExternalFeatureCount();
            if (ignoredFeatureIdx >= existingFeatureCount) {
                for (auto featureIdx : xrange(existingFeatureCount, ignoredFeatureIdx)) {
                    Y_UNUSED(featureIdx);
                    featuresLayout->AddFeature(
                        TFeatureMetaInfo(EFeatureType::Float, /*name*/ "", /*isSparse*/ true)
                    );
                    ignoredFeaturesMask->push_back(false);
                }
                featuresLayout->AddFeature(
                    TFeatureMetaInfo(
                        EFeatureType::Categorical,
                        /*name*/ "",
                        /*isSparse*/ true,
                        /*isIgnored*/ true
                    )
                );
                ignoredFeaturesMask->push_back(true);
            } else {
                featuresLayout->IgnoreExternalFeature(ignoredFeatureIdx);
                (*ignoredFeaturesMask)[ignoredFeatureIdx] = true;
            }
        }
    }

    bool TLibSvmDataLoader::DataHasGroupId(TStringBuf line) {
        auto splitter = StringSplitter(line).Split(' ');
        auto lineIterator = splitter.begin();
        auto endLineIterator = splitter.end();

        CB_ENSURE(lineIterator != endLineIterator, "Error in libsvm data. Line 0 is empty");
        ++lineIterator;

        if (lineIterator == endLineIterator) {
            return false;
        }

        if ((*lineIterator).Token().Before(':') == "qid"sv) {
            return true;
        }

        return false;
    }

    void TLibSvmDataLoader::ProcessCdData(TVector<ui32>* catFeatures, TVector<TString>* featureNames) {
        catFeatures->clear();

        TVector<TColumn> columns = Args.CdProvider->GetColumnsDescription(/*columnCount*/ Nothing());
        CB_ENSURE(
            columns.size() >= 1,
            "CdProvider has no columns. libsvm format contains at least one column"
        );

        size_t featuresStartColumn = 1;

        if (DataMetaInfo.HasGroupId) {
            CB_ENSURE(
                (columns.size() >= 2) && (columns[1].Type == EColumn::GroupId),
                "libsvm format data contains 'qid' but Column Description doesn't specify it at the second column"
            );
            ++featuresStartColumn;
        }

        for (auto columnIdx : xrange(featuresStartColumn, columns.size())) {
            const auto& column = columns[columnIdx];
            switch (column.Type) {
                case EColumn::Categ:
                    catFeatures->push_back(columnIdx - featuresStartColumn);
                    [[fallthrough]];
                case EColumn::Num:
                    featureNames->push_back(column.Id);
                    break;
                default:
                    CB_ENSURE(
                        false,
                        "Column Description. Column #" << columnIdx
                        << ": Bad type for libsvm format: " << column.Type << ". Expected feature type."
                    );
            }
        }
    }

    namespace {
        TExistsCheckerFactory::TRegistrator<TFSExistsChecker> LibSvmExistsCheckerReg("libsvm");
        TLineDataReaderFactory::TRegistrator<TFileLineDataReader> LibSvmLineDataReaderReg("libsvm");
        TDatasetLoaderFactory::TRegistrator<TLibSvmDataLoader> LibSvmDataLoaderReg("libsvm");
        TDatasetLineDataLoaderFactory::TRegistrator<TLibSvmDataLoader> LibSvmLineDataLoader("libsvm");
    }


    class TLibSvmDatasetSampler final : public IDataProviderSampler {
    public:
        TLibSvmDatasetSampler(TDataProviderSampleParams&& params)
            : Params(std::move(params))
        {}

        TDataProviderPtr SampleByIndices(TConstArrayRef<ui32> indices) override {
            return LinesFileSampleByIndices(Params, indices);
        }

        TDataProviderPtr SampleBySampleIds(TConstArrayRef<TString> sampleIds) override {
            Y_UNUSED(sampleIds);

            CB_ENSURE(false, "libsvm format has no sampleIds, so sampling by them is not supported");
            return TDataProviderPtr();
        }

    private:
        TDataProviderSampleParams Params;
    };

    namespace {
        TDataProviderSamplerFactory::TRegistrator<TLibSvmDatasetSampler> CBDsvDatasetSampler("libsvm");
    }
}
