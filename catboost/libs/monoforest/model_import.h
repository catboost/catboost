#pragma once

#include "additive_model.h"
#include "grid.h"
#include "oblivious_tree.h"

#include <catboost/libs/model/model.h>

namespace NMonoForest {
    template <typename TWeakModel>
    class IModelImporter {
    public:
        virtual const IGrid& GetGrid() = 0;
        virtual const TAdditiveModel<TWeakModel>& GetModel() = 0;
        virtual ~IModelImporter() = default;
    };

    class TCatBoostObliviousModelImporter: public IModelImporter<TObliviousTree> {
    public:
        using TModel = TAdditiveModel<TObliviousTree>;

        explicit TCatBoostObliviousModelImporter(TCatBoostGrid&& grid, TModel&& model)
            : Grid(grid)
            , Model(model)
        {
        }

        inline const IGrid& GetGrid() override {
            return Grid;
        }

        inline const TModel& GetModel() override {
            return Model;
        }

    private:
        TCatBoostGrid Grid;
        TModel Model;
    };

    THolder<IModelImporter<TObliviousTree>> MakeCatBoostImporter(const TFullModel& model);
}
