#include "export_helpers.h"

#include <util/string/builder.h>
#include <util/string/cast.h>

static TString FloatToStringWithSuffix(float value, bool addFloatingSuffix) {
    TString str = FloatToString(value, PREC_NDIGITS, 9);
    if (addFloatingSuffix) {
        if (int tmpValue; TryFromString<int>(str, tmpValue)) {
            str.append('.');
        }
        str.append("f");
    }
    return str;
}

namespace NCatboostModelExportHelpers {
    int GetBinaryFeatureCount(const TFullModel& model) {
        int binaryFeatureCount = 0;
        for (const auto& floatFeature : model.ModelTrees->GetFloatFeatures()) {
            if (!floatFeature.UsedInModel()) {
                continue;
            }
            binaryFeatureCount += floatFeature.Borders.size();
        }
        return binaryFeatureCount;
    }

    TString OutputBorderCounts(const TFullModel& model) {
        return OutputArrayInitializer([&model] (size_t i) { return model.ModelTrees->GetFloatFeatures()[i].Borders.size(); }, model.ModelTrees->GetFloatFeatures().size());
    }

    TString OutputBorders(const TFullModel& model, bool addFloatingSuffix) {
        TStringBuilder outString;
        TSequenceCommaSeparator comma(model.ModelTrees->GetFloatFeatures().size(), AddSpaceAfterComma);
        for (const auto& floatFeature : model.ModelTrees->GetFloatFeatures()) {
            if (!floatFeature.UsedInModel()) {
                continue;
            }
            outString << OutputArrayInitializer([&floatFeature, addFloatingSuffix] (size_t i) { return FloatToStringWithSuffix(floatFeature.Borders[i], addFloatingSuffix); }, floatFeature.Borders.size()) << comma;
        }
        return outString;
    }

    TString OutputLeafValues(const TFullModel& model, TIndent indent, EModelType modelType) {
        char bracketStart, bracketEnd;
        switch (modelType) {
            case EModelType::Cpp:
                bracketStart = '{';
                bracketEnd = '}';
                break;
            case EModelType::Python:
                bracketStart = '[';
                bracketEnd = ']';
                break;
            default:
                CB_ENSURE(false, "Can not export to this format");
        }

        TStringBuilder outString;
        TSequenceCommaSeparator commaOuter(model.ModelTrees->GetModelTreeData()->GetTreeSizes().size());
        ++indent;
        auto currentTreeFirstLeafPtr = model.ModelTrees->GetModelTreeData()->GetLeafValues().data();
        for (const auto& treeSize : model.ModelTrees->GetModelTreeData()->GetTreeSizes()) {
            const auto treeLeafCount = 1uLL << treeSize;
            const auto dim = model.ModelTrees->GetDimensionsCount();
            outString << '\n' << indent;
            outString << OutputArrayInitializer([&currentTreeFirstLeafPtr, &dim, &bracketStart, &bracketEnd] (size_t i) {
                return bracketStart + OutputArrayInitializer([&currentTreeFirstLeafPtr, &dim, &i] (size_t j) {
                    return FloatToString(currentTreeFirstLeafPtr[i * dim + j], PREC_NDIGITS, 16);
                }, dim) + bracketEnd;
            }, treeLeafCount);
            outString << commaOuter;
            currentTreeFirstLeafPtr += treeLeafCount * dim;
        }
        --indent;
        outString << '\n';
        return outString;
    }
}
