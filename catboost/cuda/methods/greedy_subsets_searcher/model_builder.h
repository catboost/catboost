#pragma once

#include <catboost/cuda/models/oblivious_model.h>
#include <catboost/cuda/data/leaf_path.h>

namespace NCatboostCuda {


    template <class TModel>
    TModel BuildTreeLikeModel(const TVector<TLeafPath>& leaves,
                              const TVector<double>& leafWeights,
                              const TVector<TVector<float>>& leafValues);


}
