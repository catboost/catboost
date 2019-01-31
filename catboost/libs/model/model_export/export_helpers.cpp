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
            if (!floatFeature.UsedInModel()) {
                continue;
            }
            binaryFeatureCount += floatFeature.Borders.size();
        }
        return binaryFeatureCount;
    }

    TString OutputBorderCounts(const TFullModel& model) {
        return OutputArrayInitializer([&model] (size_t i) { return model.ObliviousTrees.FloatFeatures[i].Borders.size(); }, model.ObliviousTrees.FloatFeatures.size());
    }

    TString OutputBorders(const TFullModel& model, bool addFloatingSuffix) {
        TStringBuilder outString;
        TSequenceCommaSeparator comma(model.ObliviousTrees.FloatFeatures.size(), AddSpaceAfterComma);
        for (const auto& floatFeature : model.ObliviousTrees.FloatFeatures) {
            if (!floatFeature.UsedInModel()) {
                continue;
            }
            outString << OutputArrayInitializer([&floatFeature, addFloatingSuffix] (size_t i) { return FloatToString(floatFeature.Borders[i], PREC_NDIGITS, 9) + (addFloatingSuffix ? "f" : ""); }, floatFeature.Borders.size()) << comma;
        }
        return outString;
    }

    TString OutputLeafValues(const TFullModel& model, TIndent indent) {
        TStringBuilder outString;
        TSequenceCommaSeparator commaOuter(model.ObliviousTrees.TreeSizes.size());
        ++indent;
        auto currentTreeFirstLeafPtr = model.ObliviousTrees.LeafValues.data();
        for (const auto& treeSize : model.ObliviousTrees.TreeSizes) {
            const auto treeLeafCount = (1uLL << treeSize) * model.ObliviousTrees.ApproxDimension;
            outString << '\n' << indent;
            outString << OutputArrayInitializer([&currentTreeFirstLeafPtr] (size_t i) { return FloatToString(currentTreeFirstLeafPtr[i], PREC_NDIGITS, 16); }, treeLeafCount);
            outString << commaOuter;
            currentTreeFirstLeafPtr += treeLeafCount;
        }
        --indent;
        outString << '\n';
        return outString;
    }

    TVector<TCompressedModelCtr> CompressModelCtrs(const TVector<TModelCtr>& neededCtrs) {
        TVector<TCompressedModelCtr> compressedModelCtrs;
        compressedModelCtrs.emplace_back(TCompressedModelCtr{&neededCtrs[0].Base.Projection, {&neededCtrs[0]}});
        for (size_t i = 1; i < neededCtrs.size(); ++i) {
            Y_ASSERT(neededCtrs[i - 1] < neededCtrs[i]); // needed ctrs should be sorted
            if (*(compressedModelCtrs.back().Projection) != neededCtrs[i].Base.Projection) {
                compressedModelCtrs.emplace_back(TCompressedModelCtr{&neededCtrs[i].Base.Projection, {}});
            }
            compressedModelCtrs.back().ModelCtrs.push_back(&neededCtrs[i]);
        }
        return compressedModelCtrs;
    }
}
