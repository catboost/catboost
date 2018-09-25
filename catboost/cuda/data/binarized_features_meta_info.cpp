#include "binarized_features_meta_info.h"
#include <catboost/libs/options/enums.h>
#include <catboost/libs/helpers/data_split.h>
#include <catboost/libs/logging/logging.h>
#include <util/generic/vector.h>
#include <util/generic/map.h>
#include <util/generic/string.h>
#include <util/generic/cast.h>
#include <util/string/split.h>
#include <util/stream/file.h>
#include <util/stream/input.h>
#include <util/system/fs.h>

namespace NCatboostCuda {
    template <class TOneLineReaderFunc>
    TBinarizedFloatFeaturesMetaInfo ReadBordersFromFile(TOneLineReaderFunc readLineFunc) {
        TVector<TVector<float>> borders;
        TMap<int, ENanMode> nanModes;
        borders.reserve(2000);

        TString line;
        while (readLineFunc(&line)) {
            TVector<TString> tokens;
            try {
                Split(line, "\t", tokens);
            } catch (const yexception& e) {
                CATBOOST_ERROR_LOG << "Got exception " << e.what() << " while parsing feature descriptions line " << line << Endl;
                ythrow TCatboostException() << "Can't parse borders info";
            }
            CB_ENSURE(tokens.ysize() == 2 || tokens.ysize() == 3, "Each line should have two or three columns. " << line);

            const int featureId = FromString<int>(tokens[0]);
            const float featureBorder = FromString<float>(tokens[1]);
            borders.resize(Max<int>(featureId + 1, borders.ysize()));
            borders[featureId].push_back(featureBorder);

            if (tokens.ysize() == 3) {
                ENanMode nanMode = FromString<ENanMode>(tokens[2]);

                if (nanModes.has(featureId)) {
                    CB_ENSURE(nanModes.at(featureId) == nanMode, "NaN mode should be consistent in borders file");
                } else {
                    nanModes[featureId] = nanMode;
                }
            }
        }

        TBinarizedFloatFeaturesMetaInfo metaInfo;
        metaInfo.Borders = borders;
        metaInfo.BinarizedFeatureIds.resize(borders.size());
        Iota(metaInfo.BinarizedFeatureIds.begin(), metaInfo.BinarizedFeatureIds.end(), 0);
        metaInfo.NanModes.resize(metaInfo.Borders.size(), ENanMode::Forbidden);
        for (const auto& nanMode : nanModes) {
            metaInfo.NanModes[nanMode.first] = nanMode.second;
        }
        for (auto& singleFeatureBorders : metaInfo.Borders) {
            SortUnique(singleFeatureBorders);
        }
        return metaInfo;
    }

    TBinarizedFloatFeaturesMetaInfo LoadBordersFromFromFileInMatrixnetFormat(const TString& path) {
        CB_ENSURE(NFs::Exists(path), "Borders file at [" << path << "] was not found");
        TIFStream in(path);
        return ReadBordersFromFile([&in](TString* line) -> bool {
            return static_cast<bool>(in.ReadLine(*line));
        });
    }

}
