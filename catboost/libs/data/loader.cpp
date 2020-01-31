#include "baseline.h"
#include "loader.h"

#include <catboost/libs/column_description/column.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/helpers/mem_usage.h>
#include <catboost/libs/helpers/vector_helpers.h>

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


    static TVector<TPair> ReadPairs(const TPathWithScheme& filePath, ui64 docCount, TDatasetSubset loadSubset) {
        THolder<ILineDataReader> reader = GetLineDataReader(filePath);

        TVector<TPair> pairs;
        TString line;
        for (size_t lineNumber = 0; reader->ReadLine(&line); lineNumber++) {
            TVector<TString> tokens = StringSplitter(line).Split('\t');
            if (tokens.empty()) {
                continue;
            }
            try {
                CB_ENSURE(tokens.ysize() == 2 || tokens.ysize() == 3,
                    "Each line should have two or three columns. This line has " << tokens.size()
                );
                TPair pair;

                size_t tokenIdx = 0;
                auto parseIdFunc = [&](TStringBuf description, ui32* id) {
                    CB_ENSURE(
                        TryFromString(tokens[tokenIdx], *id),
                        "Invalid " << description << " index: cannot parse as nonnegative index ("
                        << tokens[tokenIdx] << ')'
                    );
                    *id -= loadSubset.Range.Begin;
                    if (*id < loadSubset.GetSize()) {
                        CB_ENSURE(
                            *id < docCount,
                            "Invalid " << description << " index (" << *id << "): not less than number of samples"
                            " (" << docCount << ')'
                        );
                    }
                    ++tokenIdx;
                };
                parseIdFunc(AsStringBuf("Winner"), &pair.WinnerId);
                parseIdFunc(AsStringBuf("Loser"), &pair.LoserId);

                pair.Weight = 1.0f;
                if (tokens.ysize() == 3) {
                    CB_ENSURE(
                        TryFromString(tokens[2], pair.Weight),
                        "Invalid weight: cannot parse as float (" << tokens[2] << ')'
                    );
                }
                if (pair.WinnerId < loadSubset.GetSize() && pair.LoserId < loadSubset.GetSize()) {
                    pairs.push_back(std::move(pair));
                } else {
                    CB_ENSURE(
                        pair.WinnerId >= loadSubset.GetSize() && pair.LoserId >= loadSubset.GetSize(),
                        "Load subset " << loadSubset.Range << " must contain loser "
                        << pair.LoserId + loadSubset.Range.Begin << " and winner " << pair.WinnerId + loadSubset.Range.Begin
                    );
                }
            } catch (const TCatBoostException& e) {
                throw TCatBoostException() << "Incorrect file with pairs. Invalid line number #" << lineNumber
                    << ": " << e.what();
            }
        }

        return pairs;
    }

    static TVector<TVector<float>> ReadBaseline(const TPathWithScheme& filePath, ui64 docCount, TDatasetSubset loadSubset, const TVector<TString>& classNames) {
        TBaselineReader reader(filePath, classNames);

        TString line;

        TVector<ui32> tokenIndexes = reader.GetBaselineIndexes();

        TVector<TVector<float>> baseline;
        ResizeRank2(tokenIndexes.size(), docCount, baseline);
        ui32 lineNumber = 0;
        auto addBaselineFunc = [&baseline, &lineNumber, &loadSubset](ui32 approxIdx, float value) {
            baseline[approxIdx][lineNumber - loadSubset.Range.Begin] = value;
        };

        for (; reader.ReadLine(&line); lineNumber++) {
            if (lineNumber < loadSubset.Range.Begin) {
                continue;
            }
            if (lineNumber >= loadSubset.Range.End) {
                break;
            }
            reader.Parse(addBaselineFunc, line, lineNumber - loadSubset.Range.Begin);
        }
        CB_ENSURE(lineNumber - loadSubset.Range.Begin == docCount,
            "Expected " << docCount << " lines in baseline file starting at offset " << loadSubset.Range.Begin
            << " got " << lineNumber - loadSubset.Range.Begin);
        return baseline;
    }

    namespace {
        enum class EReadLocation {
            BeforeSubset,
            InSubset,
            AfterSubset
        };
    }

    static TVector<float> ReadGroupWeights(
        const TPathWithScheme& filePath,
        TConstArrayRef<TGroupId> groupIds,
        ui64 docCount,
        TDatasetSubset loadSubset
    ) {
        Y_UNUSED(loadSubset);
        CB_ENSURE(groupIds.size() == docCount, "GroupId count should correspond to object count.");
        TVector<float> groupWeights;
        groupWeights.reserve(docCount);
        ui64 groupIdCursor = 0;
        EReadLocation readLocation = EReadLocation::BeforeSubset;
        THolder<ILineDataReader> reader = GetLineDataReader(filePath);
        TString line;
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

                if (readLocation == EReadLocation::BeforeSubset) {
                    if (groupId != groupIds[groupIdCursor]) {
                        continue;
                    }
                    readLocation = EReadLocation::InSubset;
                }
                if (readLocation == EReadLocation::InSubset) {
                    if (groupIdCursor < docCount) {
                        ui64 groupSize = 0;
                        CB_ENSURE(
                            groupId == groupIds[groupIdCursor],
                            "GroupId from the file with group weights does not match GroupId from the dataset; "
                            LabeledOutput(groupId, groupIds[groupIdCursor], groupIdCursor));
                        while (groupIdCursor < docCount && groupId == groupIds[groupIdCursor]) {
                            ++groupSize;
                            ++groupIdCursor;
                        }
                        groupWeights.insert(groupWeights.end(), groupSize, groupWeight);
                    } else {
                        readLocation = EReadLocation::AfterSubset;
                    }
                }
            } catch (const TCatBoostException& e) {
                throw TCatBoostException() << "Incorrect file with group weights. Invalid line number #"
                    << lineNumber << ": " << e.what();
            }
        }
        CB_ENSURE(readLocation != EReadLocation::BeforeSubset,
            "Requested group ids are absent or non-consecutive in group weights file.");
        CB_ENSURE(groupWeights.size() == docCount,
            "Group weights file should have as many weights as the objects in the dataset.");

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



    void SetPairs(const TPathWithScheme& pairsPath, ui32 objectCount, TDatasetSubset loadSubset, IDatasetVisitor* visitor) {
        DumpMemUsage("After data read");
        if (pairsPath.Inited()) {
            visitor->SetPairs(ReadPairs(pairsPath, objectCount, loadSubset));
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

    TVector<TString> GetFeatureNames(
        const TDataColumnsMetaInfo& columnsDescription,
        const TMaybe<TVector<TString>>& headerColumns,
        const TPathWithScheme& featureNamesPath
    ) {
        // featureNamesFromColumns can be empty
        const TVector<TString> featureNamesFromColumns = columnsDescription.GenerateFeatureIds(headerColumns);
        const size_t featureCount
            = featureNamesFromColumns.empty() ?
                CountIf(
                    columnsDescription.Columns,
                    [](const TColumn& column) { return IsFactorColumn(column.Type); }
                )
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
                    s == AsStringBuf("NA") ||
                    s == AsStringBuf("Na") ||
                    s == AsStringBuf("na") ||
                    false);
            case 3:
                return (ToLower(s[0]) == 'n' || ToLower(s[1]) == 'n') && (
                    s == AsStringBuf("nan") ||
                    s == AsStringBuf("NaN") ||
                    s == AsStringBuf("NAN") ||
                    s == AsStringBuf("#NA") ||
                    s == AsStringBuf("N/A") ||
                    s == AsStringBuf("n/a") ||
                    false);
            case 4:
                return (ToLower(s[0]) == 'n' || ToLower(s[1]) == 'n') && (
                    s == AsStringBuf("#N/A") ||
                    s == AsStringBuf("-NaN") ||
                    s == AsStringBuf("-nan") ||
                    s == AsStringBuf("NULL") ||
                    s == AsStringBuf("null") ||
                    s == AsStringBuf("Null") ||
                    s == AsStringBuf("none") ||
                    s == AsStringBuf("None") ||
                    false);
            default:
                return
                    s == AsStringBuf("#N/A N/A") ||
                    s == AsStringBuf("-1.#IND") ||
                    s == AsStringBuf("-1.#QNAN") ||
                    s == AsStringBuf("1.#IND") ||
                    s == AsStringBuf("1.#QNAN") ||
                    false;
        }
    }

    bool TryParseFloatFeatureValue(TStringBuf stringValue, float* value) {
        if (!TryFromString<float>(stringValue, *value)) {
            if (IsMissingValue(stringValue)) {
                *value = std::numeric_limits<float>::quiet_NaN();
            } else {
                return false;
            }
        }
        if (*value == 0.0f) {
            *value = 0.0f; // remove negative zeros
        }
        return true;
    }
}
