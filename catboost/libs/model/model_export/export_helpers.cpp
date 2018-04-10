#include "export_helpers.h"

#include <util/string/builder.h>
#include <util/string/cast.h>

namespace NCatboostModelExportHelpers {
    int GetBinaryFeatureCount(const TFullModel& model) {
        int binaryFeatureCount = 0;
        for (const auto& floatFeature : model.ObliviousTrees.FloatFeatures) {
            binaryFeatureCount += floatFeature.Borders.size();
        }
        return binaryFeatureCount;
    }

    TString OutputBorderCounts(const TFullModel& model) {
        TStringBuilder outString;
        for (const auto& floatFeature : model.ObliviousTrees.FloatFeatures) {
            outString << floatFeature.Borders.size() << (&floatFeature != &model.ObliviousTrees.FloatFeatures.back() ? "," : "");
        }
        return outString;
    }

    TString OutputBorders(const TFullModel& model) {
        TStringBuilder outString;
        for (const auto& floatFeature : model.ObliviousTrees.FloatFeatures) {
            outString << OutputArrayInitializer(floatFeature.Borders) << (&floatFeature != &model.ObliviousTrees.FloatFeatures.back() ? "," : "");
        }
        return outString;
    }

    TString OutputLeafValues(const TFullModel& model) {
        TStringBuilder outString;
        bool first = true;
        for (const auto& treeLeaf : model.ObliviousTrees.LeafValues) {
            if (!first) {
                outString << Endl << "        ";
            }
            for (const auto leafValue : treeLeaf) {
                if (first) {
                    first = false;
                } else {
                    outString << ",";
                }
                outString << FloatToString(leafValue, PREC_NDIGITS, 16);
            }
        }
        return outString;
    }
}
