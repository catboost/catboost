#include "baseline.h"
#include "loader.h"
#include "pairs_data_loaders.h"

#include <catboost/libs/column_description/column.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/helpers/mem_usage.h>
#include <catboost/libs/helpers/vector_helpers.h>

#include <library/cpp/containers/flat_hash/flat_hash.h>

#include <util/charset/unidata.h>
#include <util/generic/algorithm.h>
#include <util/generic/ptr.h>
#include <util/generic/xrange.h>
#include <util/string/cast.h>
#include <util/string/split.h>
#include <util/system/types.h>

#include <limits>
#include <utility>


namespace NCB {


    void ProcessIgnoredFeaturesList(
        TConstArrayRef<ui32> ignoredFeatures, // [flatFeatureIdx]
        TMaybe<TString> allFeaturesIgnoredMessage,
        TDataMetaInfo* dataMetaInfo, // inout, must be inited, only ignored flags are updated
        TVector<bool>* ignoredFeaturesMask // [flatFeatureIdx], out
    ) {
        CB_ENSURE_INTERNAL(
            dataMetaInfo->FeaturesLayout,
            "ProcessIgnoredFeaturesList: TDataMetaInfo must be inited"
        );

        const ui32 featureCount = dataMetaInfo->GetFeatureCount();

        ignoredFeaturesMask->assign((size_t)featureCount, false);

        ui32 ignoredFeaturesInDataCount = 0;
        for (auto ignoredFeatureFlatIdx : ignoredFeatures) {
            if (ignoredFeatureFlatIdx >= featureCount) {
                continue;
            }

            dataMetaInfo->FeaturesLayout->IgnoreExternalFeature(ignoredFeatureFlatIdx);

            // handle possible duplicates
            ignoredFeaturesInDataCount += (*ignoredFeaturesMask)[ignoredFeatureFlatIdx] == false;
            (*ignoredFeaturesMask)[ignoredFeatureFlatIdx] = true;
        }
        CB_ENSURE(
            featureCount > ignoredFeaturesInDataCount,
            (allFeaturesIgnoredMessage.Defined() ?
                *allFeaturesIgnoredMessage
                : "All features are requested to be ignored"));
    }


    static TVector<TVector<float>> ReadBaseline(const TPathWithScheme& filePath, ui64 docCount, TDatasetSubset loadSubset, const TVector<TString>& classNames) {
        THolder<IBaselineReader> reader = GetProcessor<IBaselineReader, TBaselineReaderArgs>(
            filePath,
            TBaselineReaderArgs{filePath, classNames, loadSubset.Range}
        );

        auto baselineCount = reader->GetBaselineCount();

        TVector<TVector<float>> baseline;
        ResizeRank2(baselineCount, docCount, baseline);

        TObjectBaselineData baselineData;
        ui64 objectIdx = 0;
        ui32 objectCount = 0;

        for (; reader->Read(&baselineData, &objectIdx); objectCount++) {
            for (auto approxIdx : xrange(baselineCount)) {
                baseline[approxIdx][objectIdx] = baselineData.Baseline[approxIdx];
            }
        }
        CB_ENSURE(objectCount == docCount,
            "Expected " << docCount << " lines in baseline file starting at offset " << loadSubset.Range.Begin
            << " got " << objectCount);
        return baseline;
    }

    static TVector<float> ReadGroupWeights(
        const TPathWithScheme& filePath,
        TConstArrayRef<TGroupId> groupIds,
        ui64 docCount,
        TDatasetSubset loadSubset
    ) {
        Y_UNUSED(loadSubset);
        CB_ENSURE(groupIds.size() == docCount, "GroupId count should correspond to object count.");
        THolder<ILineDataReader> reader = GetLineDataReader(filePath);
        TString line;
        THashMap<TGroupId, float> groupWeightsByGroupId;
        for (size_t lineNumber = 0; reader->ReadLine(&line); lineNumber++) {
            try {
                TVector<TString> tokens = StringSplitter(line).Split('\t');
                CB_ENSURE(tokens.size() == 2,
                    "Each line should have two columns. This line has " << tokens.size()
                );
                const TGroupId groupId = CalcGroupIdFor(tokens[0]);
                float groupWeight = 1.0f;
                CB_ENSURE(
                    TryFromString(tokens[1], groupWeight),
                    "Invalid group weight: cannot parse as float (" << tokens[1] << ')'
                );
                CB_ENSURE(!groupWeightsByGroupId.contains(groupId), "GroupId at line " << lineNumber << " is repeated in group weights file");
                groupWeightsByGroupId[groupId] = groupWeight;
            } catch (const TCatBoostException& e) {
                throw TCatBoostException() << "Incorrect file with group weights. Invalid line number #"
                    << lineNumber << ": " << e.what();
            }
        }
        TVector<float> groupWeights;
        groupWeights.reserve(docCount);
        for (auto rowIdx : xrange(groupIds.size())) {
            CB_ENSURE(groupWeightsByGroupId.contains(groupIds[rowIdx]), "GroupId from row " << rowIdx << " in dataset is not found in group weights file");
            groupWeights.emplace_back(groupWeightsByGroupId.at(groupIds[rowIdx]));
        }

        return groupWeights;
    }

    static TVector<ui64> ReadGroupTimestamps(
        const TPathWithScheme& filePath,
        TConstArrayRef<TGroupId> groupIds,
        ui64 docCount,
        TDatasetSubset loadSubset
    ) {
        Y_UNUSED(loadSubset);
        THashSet<TGroupId> knownGroups(groupIds.begin(), groupIds.end());

        THashMap<TGroupId, ui64> groupTimestamps;
        ui32 skippedGroupsCount = 0;

        auto reader = GetLineDataReader(filePath);
        TString line;
        for (size_t lineNumber = 0; reader->ReadLine(&line); ++lineNumber) {
            TVector<TString> tokens = StringSplitter(line).Split('\t');
            if (tokens.empty()) {
                continue;
            }
            CB_ENSURE(
                tokens.size() == 2,
                "Timestamps file " << filePath << ", line number " << lineNumber << ": expect two items, got " << tokens.size());
            auto group = CalcGroupIdFor(tokens[0]);
            if (knownGroups.count(group) == 0) {
                ++skippedGroupsCount;
                continue;
            }
            CB_ENSURE(
                groupTimestamps.insert(
                    std::make_pair(
                        group,
                        FromString<ui64>(tokens[1])
                    )
                ).second,
                "Timestamps file " << filePath << ", line number " << lineNumber << ": multiple timestamps for GroupId " << tokens[0]
            );
        }
        CATBOOST_INFO_LOG << "Number of groups from file not in dataset: " << skippedGroupsCount << Endl;

        TVector<ui64> timestamps;
        timestamps.reserve(docCount);
        CB_ENSURE_INTERNAL(groupIds.size() == docCount, "Each object should have GroupId");
        for (auto group : groupIds) {
            CB_ENSURE(
                groupTimestamps.count(group) != 0,
                "Timestamps file " << filePath << ": no timestamp for GroupId with hash " << group);
            timestamps.push_back(groupTimestamps.at(group));
        }
        return timestamps;
    }

    void SetPairs(const TPathWithScheme& pairsPath, TDatasetSubset loadSubset, IDatasetVisitor* visitor) {
        DumpMemUsage("After data read");
        if (pairsPath.Inited()) {
            auto pairsDataLoader = GetProcessor<IPairsDataLoader>(
                pairsPath,
                TPairsDataLoaderArgs{pairsPath, loadSubset}
            );
            if (pairsDataLoader->NeedGroupIdToIdxMap()) {
                auto maybeGroupIds = visitor->GetGroupIds();
                CB_ENSURE(maybeGroupIds, "Cannot load pairs data with group ids for a dataset without groups");
                pairsDataLoader->SetGroupIdToIdxMap(*maybeGroupIds);
            }
            pairsDataLoader->Do(visitor);
        }
    }

    void SetGraph(const TPathWithScheme& graphPath, TDatasetSubset loadSubset, IDatasetVisitor* visitor) {
        DumpMemUsage("Before data read graphPath");
        if (graphPath.Inited()) {
            auto pairsDataLoader = GetProcessor<IPairsDataLoader>(
                graphPath,
                TPairsDataLoaderArgs{graphPath, loadSubset}
            );
            pairsDataLoader->IsPairs = false;
            if (pairsDataLoader->NeedGroupIdToIdxMap()) {
                auto maybeGroupIds = visitor->GetGroupIds();
                CB_ENSURE(maybeGroupIds, "Cannot load graph data with group ids for a dataset without groups");
                pairsDataLoader->SetGroupIdToIdxMap(*maybeGroupIds);
            }
            pairsDataLoader->Do(visitor);
        }
    }

    void SetGroupWeights(
        const TPathWithScheme& groupWeightsPath,
        ui32 objectCount,
        TDatasetSubset loadSubset,
        IDatasetVisitor* visitor
    ) {
        DumpMemUsage("After data read");
        if (groupWeightsPath.Inited()) {
            auto maybeGroupIds = visitor->GetGroupIds();
            CB_ENSURE(maybeGroupIds, "Cannot load group weights data for a dataset without groups");
            TVector<float> groupWeights = ReadGroupWeights(
                groupWeightsPath,
                *maybeGroupIds,
                objectCount,
                loadSubset
            );
            visitor->SetGroupWeights(std::move(groupWeights));
        }
    }

    void SetBaseline(
        const TPathWithScheme& baselinePath,
        ui32 objectCount,
        TDatasetSubset loadSubset,
        const TVector<TString>& classNames,
        IDatasetVisitor* visitor
    ) {
        DumpMemUsage("After data read");
        if (baselinePath.Inited()) {
            visitor->SetBaseline(ReadBaseline(baselinePath, objectCount, loadSubset, classNames));
        }
    }

    void SetTimestamps(
        const TPathWithScheme& timestampsPath,
        ui32 objectCount,
        TDatasetSubset loadSubset,
        IDatasetVisitor* visitor
    ) {
        DumpMemUsage("After data read");
        if (timestampsPath.Inited()) {
            auto maybeGroupIds = visitor->GetGroupIds();
            CB_ENSURE(maybeGroupIds, "Cannot load group timestamps for a dataset without groups");
            TVector<ui64> groupTimestamps = ReadGroupTimestamps(
                timestampsPath,
                *maybeGroupIds,
                objectCount,
                loadSubset
            );
            visitor->SetTimestamps(std::move(groupTimestamps));
        }
    }

    TVector<TString> LoadFeatureNames(const TPathWithScheme& featureNamesPath) {
        TVector<TString> featureNames;

        if (featureNamesPath.Inited()) {
            auto reader = GetLineDataReader(featureNamesPath);
            TString line;
            for (size_t lineNumber = 0; reader->ReadLine(&line); ++lineNumber) {
                TVector<TString> tokens = StringSplitter(line).Split('\t');
                if (tokens.empty()) {
                    continue;
                }
                try {
                    CB_ENSURE(tokens.size() == 2, "expect two items, got " << tokens.size());

                    size_t featureIdx;
                    CB_ENSURE(
                        TryFromString(tokens[0], featureIdx),
                        "Wrong format: first field is not unsigned integer");

                    CB_ENSURE(
                        !tokens[1].empty(),
                        "feature name is empty");

                    if (featureIdx >= featureNames.size()) {
                        featureNames.resize(featureIdx + 1);
                    } else {
                        CB_ENSURE(
                            featureNames[featureIdx].empty(),
                            "feature index " << featureIdx << " specified multiple times");
                    }

                    featureNames[featureIdx] = tokens[1];

                } catch (std::exception& e) {
                    throw TCatBoostException() << "Feature names data from " << featureNamesPath
                        << ", line number " << lineNumber << ": " << e.what();
                }
            }
        }

        return featureNames;
    }

    static size_t GetFeatureCount(TConstArrayRef<TColumn> columns) {
        size_t featureCount = 0;

        for (auto column : columns) {
            if (IsFactorColumn(column.Type)) {
                ++featureCount;
            } else if (column.Type == EColumn::Features) {
                featureCount += GetFeatureCount(column.SubColumns);
            }
        }

        return featureCount;
    }


    TVector<TString> GetFeatureNames(
        const TDataColumnsMetaInfo& columnsDescription,
        const TMaybe<TVector<TString>>& headerColumns,
        const TPathWithScheme& featureNamesPath
    ) {
        // featureNamesFromColumns can be empty
        const TVector<TString> featureNamesFromColumns = columnsDescription.GenerateFeatureIds(headerColumns);
        const size_t featureCount
            = featureNamesFromColumns.empty() ?
                GetFeatureCount(columnsDescription.Columns)
                : featureNamesFromColumns.size();

        TVector<TString> externalFeatureNames = LoadFeatureNames(featureNamesPath);

        if (externalFeatureNames.empty()) {
            return featureNamesFromColumns;
        } else {
            CB_ENSURE(
                externalFeatureNames.size() <= featureCount,
                "feature names file contains index (" << (externalFeatureNames.size() - 1)
                << ") that is not less than the number of features in the dataset (" << featureCount << ')'
            );
            externalFeatureNames.resize(featureCount);
            if (!featureNamesFromColumns.empty()) {
                for (auto featureIdx : xrange(featureCount)) {
                    CB_ENSURE(
                        featureNamesFromColumns[featureIdx].empty()
                        || (featureNamesFromColumns[featureIdx] == externalFeatureNames[featureIdx]),
                        "Feature #" << featureIdx << ": name from columns specification (\""
                        << featureNamesFromColumns[featureIdx]
                        << "\") is not equal to name from feature names file (\""
                        << externalFeatureNames[featureIdx] << "\")");
                }
            }
            return externalFeatureNames;
        }
    }

    bool IsMissingValue(const TStringBuf& s) {
        switch (s.length()) {
            case 0:
                return true;
            case 1:
                return s[0] == '-';
            case 2:
                return (ToLower(s[0]) == 'n') && (
                    s == "NA"sv ||
                    s == "Na"sv ||
                    s == "na"sv ||
                    false);
            case 3:
                return (ToLower(s[0]) == 'n' || ToLower(s[1]) == 'n') && (
                    s == "nan"sv ||
                    s == "NaN"sv ||
                    s == "NAN"sv ||
                    s == "#NA"sv ||
                    s == "N/A"sv ||
                    s == "n/a"sv ||
                    false);
            case 4:
                return (ToLower(s[0]) == 'n' || ToLower(s[1]) == 'n') && (
                    s == "#N/A"sv ||
                    s == "-NaN"sv ||
                    s == "-nan"sv ||
                    s == "NULL"sv ||
                    s == "null"sv ||
                    s == "Null"sv ||
                    s == "none"sv ||
                    s == "None"sv ||
                    false);
            default:
                return
                    s == "#N/A N/A"sv ||
                    s == "-1.#IND"sv ||
                    s == "-1.#QNAN"sv ||
                    s == "1.#IND"sv ||
                    s == "1.#QNAN"sv ||
                    false;
        }
    }

    namespace {
        struct TStupidStringHash {
            size_t operator()(TStringBuf str) {
                Y_ASSERT(str.size() == 3);
                return IntHash(str[0] + (str[2] << 8));
            }
        };

        bool TryFloatFromStringFast(TStringBuf token, float& value) {
            if (token.empty()) {
                return false;
            }
            static const NFH::TFlatHashMap<TStringBuf, float, TStupidStringHash> wellKnown = {
                { TStringBuf("0.0"), 0.0f},
                { TStringBuf("1.0"), 1.0f},
                { TStringBuf("2.0"), 2.0f},
                { TStringBuf("3.0"), 3.0f},
                { TStringBuf("4.0"), 4.0f},
                { TStringBuf("5.0"), 5.0f},
                { TStringBuf("6.0"), 6.0f},
                { TStringBuf("7.0"), 7.0f},
                { TStringBuf("8.0"), 8.0f},
                { TStringBuf("9.0"), 9.0f}
            };
            if (token.size() == 1 && token[0] >= '0' && token[0] <= '9') {
                value = float(token[0] - '0');
                return true;
            } else if (token.size() == 3) {
                if ( auto i = wellKnown.find(token); i != wellKnown.end()) {
                    value = i->second;
                    return true;
                }
            } else if (token[0] == '-' && token.size() == 4) {
                if ( auto i = wellKnown.find(token.substr(1)); i != wellKnown.end()) {
                    value = -i->second;
                    return true;
                }
            }
            return TryFromString<float>(token, value);
        }
    }

    bool TryFloatFromString(TStringBuf token, bool parseNonFinite, float* value) {
        if (TryFloatFromStringFast(token, *value)) {
            if (*value == 0.0f) {
                *value = 0.0f; // remove negative zeros
            }
            return true;
        }
        if (!parseNonFinite) {
            return false;
        }
        if (IsMissingValue(token)) {
            *value = std::numeric_limits<float>::quiet_NaN();
        } else if (TCIEqualTo<TStringBuf>()(token, "inf") || TCIEqualTo<TStringBuf>()(token, "infinity")) {
            *value = std::numeric_limits<float>::infinity();
        } else if (TCIEqualTo<TStringBuf>()(token, "-inf") || TCIEqualTo<TStringBuf>()(token, "-infinity")) {
            *value = -std::numeric_limits<float>::infinity();
        } else {
            return false;
        }
        return true;
    }
}
