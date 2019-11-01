#include "helpers.h"

#include <util/generic/fwd.h>
#include <util/string/builder.h>
#include <util/string/join.h>

namespace NMonoForest {
    TString ToHumanReadableString(const TBinarySplit& split, const IGrid& grid) {
        TStringBuilder builder;
        builder << "[F" << grid.ExternalFlatFeatureIndex(split.FeatureId);
        builder << (split.SplitType == EBinSplitType::TakeGreater ? " > " : " = ");
        builder << grid.Border(split.FeatureId, split.BinIdx);
        builder << "]";
        return builder;
    }

    TString ToHumanReadableString(const TMonomStructure& structure, const IGrid& grid) {
        TStringBuilder builder;
        for (const auto& split : structure.Splits) {
            builder << ToHumanReadableString(split, grid);
        }
        return builder;
    }

    TString ToHumanReadableString(const TPolynom& polynom, const IGrid& grid) {
        if (polynom.MonomsEnsemble.empty()) {
            return "0";
        }
        TStringBuilder builder;
        bool isFirst = true;
        for (const auto& [structure, stat] : polynom.MonomsEnsemble) {
            if (!isFirst) {
                builder << " + ";
            }
            builder << "("
                    << JoinSeq(", ", stat.Value)
                    << ")";
            if (!structure.Splits.empty()) {
                builder << " * "
                        << ToHumanReadableString(structure, grid);
            }
            isFirst = false;
        }
        return builder;
    }
}
