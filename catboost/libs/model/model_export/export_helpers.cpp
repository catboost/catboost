#include "export_helpers.h"

#include <util/string/builder.h>
#include <util/string/cast.h>

static TString FloatToStringWithSuffix(float value, bool addFloatingSuffix) {
    TString str = FloatToString(value, PREC_NDIGITS, 9);
    if (addFloatingSuffix) {
        if (int value; TryFromString<int>(str, value)) {
            str.append('.');
        }
        str.append("f");
    }
    return str;
}

namespace NCatboostModelExportHelpers {
    int GetBinaryFeatureCount(const TFullModel& model) {
        int binaryFeatureCount = 0;
        for (const auto& floatFeature : model.ObliviousTrees->GetFloatFeatures()) {
            if (!floatFeature.UsedInModel()) {
                continue;
            }
            binaryFeatureCount += floatFeature.Borders.size();
        }
        return binaryFeatureCount;
    }

    TString OutputBorderCounts(const TFullModel& model) {
        return OutputArrayInitializer([&model] (size_t i) { return model.ObliviousTrees->GetFloatFeatures()[i].Borders.size(); }, model.ObliviousTrees->GetFloatFeatures().size());
    }

    TString OutputBorders(const TFullModel& model, bool addFloatingSuffix) {
        TStringBuilder outString;
        TSequenceCommaSeparator comma(model.ObliviousTrees->GetFloatFeatures().size(), AddSpaceAfterComma);
        for (const auto& floatFeature : model.ObliviousTrees->GetFloatFeatures()) {
            if (!floatFeature.UsedInModel()) {
                continue;
            }
            outString << OutputArrayInitializer([&floatFeature, addFloatingSuffix] (size_t i) { return FloatToStringWithSuffix(floatFeature.Borders[i], addFloatingSuffix); }, floatFeature.Borders.size()) << comma;
        }
        return outString;
    }

    TString OutputLeafValues(const TFullModel& model, TIndent indent) {
        TStringBuilder outString;
        TSequenceCommaSeparator commaOuter(model.ObliviousTrees->GetTreeSizes().size());
        ++indent;
        auto currentTreeFirstLeafPtr = model.ObliviousTrees->GetLeafValues().data();
        for (const auto& treeSize : model.ObliviousTrees->GetTreeSizes()) {
            const auto treeLeafCount = (1uLL << treeSize) * model.ObliviousTrees->GetDimensionsCount();
            outString << '\n' << indent;
            outString << OutputArrayInitializer([&currentTreeFirstLeafPtr] (size_t i) { return FloatToString(currentTreeFirstLeafPtr[i], PREC_NDIGITS, 16); }, treeLeafCount);
            outString << commaOuter;
            currentTreeFirstLeafPtr += treeLeafCount;
        }
        --indent;
        outString << '\n';
        return outString;
    }
}
