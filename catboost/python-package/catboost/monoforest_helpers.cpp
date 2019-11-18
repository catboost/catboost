#include "monoforest_helpers.h"

#include <catboost/libs/monoforest/helpers.h>
#include <catboost/libs/monoforest/interpretation.h>
#include <catboost/libs/monoforest/model_import.h>
#include <catboost/libs/monoforest/polynom.h>

namespace NMonoForest {
    static TPolynom BuildPolynom(const TAdditiveModel<TObliviousTree>& additiveModel) {
        TPolynomBuilder polynomBuilder;
        for (auto idx : xrange(additiveModel.Size())) {
            polynomBuilder.AddTree(additiveModel.GetWeakModel(idx));
        }
        return polynomBuilder.Build();
    }

    TVector<THumanReadableMonom> ConvertFullModelToPolynom(const TFullModel& fullModel) {
        const auto importer = MakeCatBoostImporter(fullModel);
        const TPolynom polynom = BuildPolynom(importer->GetModel());
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
        const TPolynom polynom = BuildPolynom(importer->GetModel());
        return ToHumanReadableString(polynom, importer->GetGrid());
    }

    TVector<TFeatureExplanation> ExplainFeatures(const TFullModel& fullModel) {
        const auto importer = MakeCatBoostImporter(fullModel);
        const TPolynom polynom = BuildPolynom(importer->GetModel());
        return ExplainFeatures(polynom, importer->GetGrid());
    }
}
