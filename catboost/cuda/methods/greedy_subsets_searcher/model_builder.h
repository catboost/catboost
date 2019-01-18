#pragma once

#include <catboost/cuda/data/leaf_path.h>

namespace NCatboostCuda {
    template <class TModel>
    TModel BuildTreeLikeModel(const TVector<TLeafPath>& leaves,
                              const TVector<double>& leavesWeight,
                              const TVector<TVector<float>>& leavesValues);

}
