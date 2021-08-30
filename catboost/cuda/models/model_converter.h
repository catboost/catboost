#pragma once

#include "oblivious_model.h"
#include "additive_model.h"

namespace NCatboostCuda {
    template <class TModel>
    THolder<TAdditiveModel<TObliviousTreeModel>> MakeObliviousModel(THolder<TAdditiveModel<TModel>>&& model, NPar::ILocalExecutor* executor);
}
