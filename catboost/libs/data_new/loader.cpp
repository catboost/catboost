#include "loader.h"

#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/helpers/mem_usage.h>

#include <util/generic/ptr.h>
#include <util/string/cast.h>
#include <util/string/iterator.h>
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
    }


    static TVector<TPair> ReadPairs(const TPathWithScheme& filePath, ui64 docCount) {
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
                    CB_ENSURE(
                        *id < docCount,
                        "Invalid " << description << " index (" << *id << "): not less than number of samples"
                        " (" << docCount << ')'
                    );
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
                pairs.push_back(std::move(pair));
            } catch (const TCatBoostException& e) {
                throw TCatBoostException() << "Incorrect file with pairs. Invalid line number #" << lineNumber
                    << ": " << e.what();
            }
        }

        return pairs;
    }

    static TVector<float> ReadGroupWeights(
        const TPathWithScheme& filePath,
        TConstArrayRef<TGroupId> groupIds,
        ui64 docCount
    ) {
        CB_ENSURE(groupIds.size() == docCount, "GroupId count should correspond to object count.");
        TVector<float> groupWeights;
        groupWeights.reserve(docCount);
        ui64 groupIdCursor = 0;
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
            } catch (const TCatBoostException& e) {
                throw TCatBoostException() << "Incorrect file with group weights. Invalid line number #"
                    << lineNumber << ": " << e.what();
            }
        }
        CB_ENSURE(groupWeights.size() == docCount,
            "Group weights file should have as many weights as the objects in the dataset.");

        return groupWeights;
    }

    void SetPairs(const TPathWithScheme& pairsPath, ui32 objectCount, IDatasetVisitor* visitor) {
        DumpMemUsage("After data read");
        if (pairsPath.Inited()) {
            visitor->SetPairs(ReadPairs(pairsPath, objectCount));
        }
    }

    void SetGroupWeights(
        const TPathWithScheme& groupWeightsPath,
        ui32 objectCount,
        IDatasetVisitor* visitor
    ) {
        DumpMemUsage("After data read");
        if (groupWeightsPath.Inited()) {
            auto maybeGroupIds = visitor->GetGroupIds();
            CB_ENSURE(maybeGroupIds, "Cannot load group weights data for a dataset without groups");
            TVector<float> groupWeights = ReadGroupWeights(
                groupWeightsPath,
                *maybeGroupIds,
                objectCount
            );
            visitor->SetGroupWeights(std::move(groupWeights));
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
