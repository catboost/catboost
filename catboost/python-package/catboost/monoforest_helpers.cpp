#include "monoforest_helpers.h"

#include <catboost/libs/monoforest/helpers.h>
#include <catboost/libs/monoforest/model_import.h>
#include <catboost/libs/monoforest/polynom.h>

namespace NMonoForest {
    TVector<THumanReadableMonom> ConvertFullModelToPolynom(const TFullModel& fullModel) {
        const auto importer = MakeCatBoostImporter(fullModel);
        TAdditiveModel<TObliviousTree> additiveModel = importer->GetModel();
        TPolynomBuilder polynomBuilder;
        for (auto idx : xrange(additiveModel.Size())) {
            polynomBuilder.AddTree(additiveModel.GetWeakModel(idx));
        }
        TPolynom polynom = polynomBuilder.Build();
        TVector<THumanReadableMonom> monoms;
        monoms.reserve(polynom.MonomsEnsemble.size());
        const IGrid& grid = importer->GetGrid();
        for (const auto& [structure, stat] : polynom.MonomsEnsemble) {
            THumanReadableMonom monom;
            for (const auto& structureSplit : structure.Splits) {
                THumanReadableSplit split;
                split.FeatureIdx = grid.ExternalFlatFeatureIndex(structureSplit.FeatureId);
                split.SplitType = structureSplit.SplitType;
                split.Border = grid.Border(structureSplit.FeatureId, structureSplit.BinIdx);
                monom.Splits.push_back(split);
            }
            monom.Value = stat.Value;
            monom.Weight = stat.Weight;
            monoms.push_back(monom);
        }
        return monoms;
    }

    TString ConvertFullModelToPolynomString(const TFullModel& fullModel) {
        const auto importer = MakeCatBoostImporter(fullModel);
        TAdditiveModel<TObliviousTree> additiveModel = importer->GetModel();
        TPolynomBuilder polynomBuilder;
        for (auto idx : xrange(additiveModel.Size())) {
            polynomBuilder.AddTree(additiveModel.GetWeakModel(idx));
        }
        TPolynom polynom = polynomBuilder.Build();
        return ToHumanReadableString(polynom, importer->GetGrid());
    }
}
