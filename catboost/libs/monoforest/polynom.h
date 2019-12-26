#pragma once

#include "additive_model.h"
#include "monom.h"
#include "oblivious_tree.h"
#include "non_symmetric_tree.h"
#include <util/generic/hash.h>

namespace NMonoForest {
    struct TPolynom {
        THashMap<TMonomStructure, TMonomStat> MonomsEnsemble;

        size_t Dimension() const {
            Y_ASSERT(!MonomsEnsemble.empty());
            return MonomsEnsemble.begin()->second.Value.size();
        }

        Y_SAVELOAD_DEFINE(MonomsEnsemble);
    };

    class TPolynomBuilder {
    public:
        void AddTree(const TObliviousTree& tree);
        void AddTree(const TNonSymmetricTree& tree);
        TPolynom Build();

    private:
        THashMap<TMonomStructure, TMonomStat> MonomsEnsemble;
    };

    template <typename TWeakModel>
    class IPolynomToAdditiveModelConverter {
    public:
        virtual TAdditiveModel<TWeakModel> Convert(const TPolynom& polynom) = 0;
        virtual ~IPolynomToAdditiveModelConverter() = default;
    };

    class TPolynomToObliviousEnsembleConverter: public IPolynomToAdditiveModelConverter<TObliviousTree> {
    public:
        TAdditiveModel<TObliviousTree> Convert(const TPolynom& polynom) override;
    };
}
