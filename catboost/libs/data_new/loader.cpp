#include "loader.h"

#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/helpers/mem_usage.h>

#include <util/generic/ptr.h>
#include <util/string/iterator.h>
#include <util/system/types.h>


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

    bool IsNanValue(const TStringBuf& s) {
        return s == "nan" || s == "NaN" || s == "NAN" || s == "NA" || s == "Na" || s == "na";
    }

}
