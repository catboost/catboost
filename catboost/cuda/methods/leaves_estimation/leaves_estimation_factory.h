#pragma once

#include "oracle_interface.h"
#include "leaves_estimation_config.h"
#include <catboost/cuda/targets/oracle_type.h>

namespace NCatboostCuda {
    template <class TObjective>
    class TOracleFactory: public ILeavesEstimationOracleFactory {
    public:
        TOracleFactory(const TObjective& target)
            : Target(&target)
        {
        }

        THolder<ILeavesEstimationOracle> Create(const TLeavesEstimationConfig& config,
                                                TStripeBuffer<const float>&& baseline,
                                                TStripeBuffer<ui32>&& bins,
                                                ui32 binCount,
                                                TGpuAwareRandom& random) const final {
            return TOracle<TObjective>::Create(*Target,
                                               std::move(baseline),
                                               std::move(bins),
                                               binCount,
                                               config,
                                               random);
        }

    private:
        const TObjective* Target;
    };
}
