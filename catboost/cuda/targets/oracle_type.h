#pragma once

namespace NCatboostCuda {
    enum class EOracleType {
        Groupwise,
        Pairwise,
        Pointwise
    };

    template <class TTarget,
              EOracleType Type = TTarget::OracleType()>
    class TOracle;
}
