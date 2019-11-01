#pragma once

#include <util/ysaveload.h>

namespace NMonoForest {
    template <class TInner>
    class TAdditiveModel {
    public:
        TAdditiveModel() = default;

        TAdditiveModel(const TAdditiveModel& other) = default;

        void AddWeakModel(TInner&& weak) {
            WeakModels.push_back(std::move(weak));
        }

        ui32 OutputDim() const {
            return WeakModels.back().OutputDim();
        }

        const TInner& GetWeakModel(int i) const {
            return WeakModels[i];
        }

        const TInner& operator[](int i) const {
            return WeakModels[i];
        }

        size_t Size() const {
            return WeakModels.size();
        }

        Y_SAVELOAD_DEFINE(WeakModels);

    private:
        TVector<TInner> WeakModels;
    };
}
