#pragma once

#include <catboost/cuda/models/oblivious_model.h>
#include <catboost/cuda/models/non_symmetric_tree.h>
#include <catboost/cuda/models/additive_model.h>
#include <catboost/cuda/data/leaf_path.h>

namespace NCatboostCuda {
    TAdditiveModel<TObliviousTreeModel> MakeOTEnsemble(const TAdditiveModel<TNonSymmetricTree>& ensemble,
                                                       NPar::ILocalExecutor* localExecutor);

}
