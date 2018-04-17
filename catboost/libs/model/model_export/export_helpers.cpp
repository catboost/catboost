#include "export_helpers.h"

#include <util/string/builder.h>
#include <util/string/cast.h>

namespace NCatboostModelExportHelpers {
    template <>
    TString OutputArrayInitializer(const TVector<unsigned char>& values) {
        return OutputArrayInitializer([&values] (size_t i) { return (int)values[i]; }, values.size());
    }

    int GetBinaryFeatureCount(const TFullModel& model) {
        int binaryFeatureCount = 0;
        for (const auto& floatFeature : model.ObliviousTrees.FloatFeatures) {
            binaryFeatureCount += floatFeature.Borders.size();
        }
        return binaryFeatureCount;
    }

    TString OutputBorderCounts(const TFullModel& model) {
        return OutputArrayInitializer([&model] (size_t i) { return model.ObliviousTrees.FloatFeatures[i].Borders.size(); }, model.ObliviousTrees.FloatFeatures.size());
    }

    TString OutputBorders(const TFullModel& model) {
        TStringBuilder outString;
        TSequenceCommaSeparator comma(model.ObliviousTrees.FloatFeatures.size());
        for (const auto& floatFeature : model.ObliviousTrees.FloatFeatures) {
            outString << OutputArrayInitializer(floatFeature.Borders) << comma;
        }
        return outString;
    }

    TString OutputLeafValues(const TFullModel& model, TIndent indent) {
        TStringBuilder outString;
        TSequenceCommaSeparator commaOuter(model.ObliviousTrees.LeafValues.size());
        ++indent;
        for (const auto& treeLeaf : model.ObliviousTrees.LeafValues) {
            outString << Endl << indent;
            outString << OutputArrayInitializer([&treeLeaf] (size_t i) { return FloatToString(treeLeaf[i], PREC_NDIGITS, 16); }, treeLeaf.size());
            outString << commaOuter;
        }
        --indent;
        outString << Endl;
        return outString;
    }
}
