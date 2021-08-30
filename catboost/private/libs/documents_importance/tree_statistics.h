#pragma once

#include <catboost/libs/data/data_provider.h>
#include <catboost/libs/model/model.h>

#include <util/generic/fwd.h>
#include <util/generic/vector.h>
#include <util/system/types.h>


struct TTreeStatistics {
    ui32 LeafCount;
    TVector<ui32> LeafIndices; // [docCount] // leafId for every train docId.
    TVector<TVector<ui32>> LeavesDocId; // [leafCount] // docIds for every leafId.
    TVector<TVector<double>> LeafValues; // [LeavesEstimationIterationsCount][leafCount]
    TVector<TVector<double>> FormulaDenominators; // [LeavesEstimationIterationsCount][leafCount] // Denominator from equation (6).
    TVector<TVector<double>> FormulaNumeratorAdding; // [LeavesEstimationIterationsCount][docCount] // The first term from equation (6).
    TVector<TVector<double>> FormulaNumeratorMultiplier; // [LeavesEstimationIterationsCount][docCount] // The jacobian multiplier from equation (6).
};

// A class that stores all the necessary statistics per each tree.
class ITreeStatisticsEvaluator {
public:
    ITreeStatisticsEvaluator(ui32 docCount)
        : DocCount(docCount)
        , FirstDerivatives(docCount)
        , SecondDerivatives(docCount)
        , ThirdDerivatives(docCount)
    {
    }
    virtual ~ITreeStatisticsEvaluator() = default;

    TVector<TTreeStatistics> EvaluateTreeStatistics(
        const TFullModel& model,
        const NCB::TProcessedDataProvider& processedData,
        const TMaybe<double> startingApprox,
        int logPeriod = 0
    );

private:
    // Compute leaf numerators.
    virtual TVector<double> ComputeLeafNumerators(TConstArrayRef<float> weights) = 0;
    // Compute leaf denominators.
    virtual TVector<double> ComputeLeafDenominators(TConstArrayRef<float> weights, float l2LeafReg) = 0;
    // Compute formula (6) numerator adding.
    virtual TVector<double> ComputeFormulaNumeratorAdding() = 0;
    // Compute formula (6) numerator multiplier.
    virtual TVector<double> ComputeFormulaNumeratorMultiplier(TConstArrayRef<float> weights) = 0;

protected:
    ui32 DocCount;
    TVector<double> FirstDerivatives; // [docCount]
    TVector<double> SecondDerivatives; // [docCount]
    TVector<double> ThirdDerivatives; // [docCount]

    ui32 LeafCount;
    TVector<ui32> LeafIndices; // [docCount]
    TVector<double> LeafValues; // [leafCount]
};

class TGradientTreeStatisticsEvaluator : public ITreeStatisticsEvaluator {
public:
    TGradientTreeStatisticsEvaluator(ui32 docCount)
        : ITreeStatisticsEvaluator(docCount)
    {
    }
    TVector<double> ComputeLeafNumerators(TConstArrayRef<float> weights) override;
    TVector<double> ComputeLeafDenominators(TConstArrayRef<float> weights, float l2LeafReg) override;
    TVector<double> ComputeFormulaNumeratorAdding() override;
    TVector<double> ComputeFormulaNumeratorMultiplier(TConstArrayRef<float> weights) override;
};

class TNewtonTreeStatisticsEvaluator : public ITreeStatisticsEvaluator {
public:
    TNewtonTreeStatisticsEvaluator(ui32 docCount)
        : ITreeStatisticsEvaluator(docCount)
    {
    }
    TVector<double> ComputeLeafNumerators(TConstArrayRef<float> weights) override;
    TVector<double> ComputeLeafDenominators(TConstArrayRef<float> weights, float l2LeafReg) override;
    TVector<double> ComputeFormulaNumeratorAdding() override;
    TVector<double> ComputeFormulaNumeratorMultiplier(TConstArrayRef<float> weights) override;
};

