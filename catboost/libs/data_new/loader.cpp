#include "baseline.h"
#include "loader.h"

#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/helpers/mem_usage.h>
#include <catboost/libs/helpers/vector_helpers.h>

#include <util/generic/ptr.h>
#include <util/string/cast.h>
#include <util/string/split.h>
#include <util/system/types.h>

#include <limits>
#include <utility>


namespace NCB {


    void ProcessIgnoredFeaturesList(
        TConstArrayRef<ui32> ignoredFeatures, // [flatFeatureIdx]
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
        CB_ENSURE(featureCount > ignoredFeaturesInDataCount, "All features are requested to be ignored");
        CB_ENSURE_INTERNAL(featureCount >= ignoredFeaturesInDataCount,
            "Too many ignored features: feature count is " << featureCount
            << " ignored features count is " << ignoredFeaturesInDataCount);
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
                        "Load subset [" << loadSubset.Range.Begin << ", " << loadSubset.Range.End << ") must contain loser "
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

    bool IsMissingValue(const TStringBuf& s) {
        return
            s == AsStringBuf("") ||
            s == AsStringBuf("nan") ||
            s == AsStringBuf("NaN") ||
            s == AsStringBuf("NAN") ||
            s == AsStringBuf("NA") ||
            s == AsStringBuf("Na") ||
            s == AsStringBuf("na") ||
            s == AsStringBuf("#N/A") ||
            s == AsStringBuf("#N/A N/A") ||
            s == AsStringBuf("#NA") ||
            s == AsStringBuf("-1.#IND") ||
            s == AsStringBuf("-1.#QNAN") ||
            s == AsStringBuf("-NaN") ||
            s == AsStringBuf("-nan") ||
            s == AsStringBuf("1.#IND") ||
            s == AsStringBuf("1.#QNAN") ||
            s == AsStringBuf("N/A") ||
            s == AsStringBuf("NULL") ||
            s == AsStringBuf("n/a") ||
            s == AsStringBuf("null") ||
            s == AsStringBuf("Null") ||
            s == AsStringBuf("none") ||
            s == AsStringBuf("None") ||
            s == AsStringBuf("-");
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
