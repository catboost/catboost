#pragma once

#include "bootstrap.h"
#include "score_function.h"

namespace NCatboostCuda
{
    struct TObliviousTreeLearnerOptions
    {
    public:
        ui32 GetMaxDepth() const
        {
            return MaxDepth;
        }

        ui32 GetLeavesEstimationIters() const
        {
            return LeavesEstimationIters;
        }

        double GetL2Reg() const
        {
            return L2Reg;
        }

        bool IsNormalize() const
        {
            return NormalizeLossInEstimation;
        }

        bool AddRidgeToTargetFunction() const
        {
            return AddRidgeToTargetFunctionFlag;
        }

        bool IsBootstrapTestOnly() const
        {
            return !BootstrapLearn;
        }

        TObliviousTreeLearnerOptions& SetLeavesEstimationIterations(ui32 iters)
        {
            LeavesEstimationIters = iters;
            return *this;
        }

        void SetMaxDepth(const ui32 maxDepth)
        {
            TObliviousTreeLearnerOptions::MaxDepth = maxDepth;
        }

        bool IsUseNewton() const
        {
            return UseNewton;
        }

        EScoreFunction GetScoreFunction() const
        {
            return ScoreFunction;
        }

        const TBootstrapConfig& GetBootstrapConfig() const
        {
            return BootstrapConfig;
        }

        ui32 GetMaxCtrComplexityForBordersCaching() const
        {
            return MaxCtrComplexityForBordersCaching;
        }

        bool IsDumpFreeMemory() const
        {
            return DumpFreeMemoryFlag;
        }

        template<class TConfig>
        friend
        class TOptionsBinder;

        template<class TConfig>
        friend
        class TOptionsJsonConverter;

        void Validate() const
        {
            CB_ENSURE(MaxDepth <= 16, "Maximum depth should be <= 16");
            CB_ENSURE(LeavesEstimationIters > 0);
            BootstrapConfig.Validate();
        }

    private:
        ui32 MaxDepth = 6;
        ui32 LeavesEstimationIters = 10;
        ui32 MaxCtrComplexityForBordersCaching = 1;
        double L2Reg = 0.00001;
        bool DumpFreeMemoryFlag = false;
        bool UseNewton = true;
        EScoreFunction ScoreFunction = EScoreFunction::Correlation;
        TBootstrapConfig BootstrapConfig;
        bool NormalizeLossInEstimation = false;
        bool AddRidgeToTargetFunctionFlag = false;
        bool BootstrapLearn = false;
    };
}
