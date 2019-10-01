#pragma once

#include <catboost/libs/data/target.h>


namespace NCB {
    namespace NDataNewUT {

    template <class T>
    inline TSharedVector<T> ShareVector(TVector<T>&& data) {
        return MakeAtomicShared<TVector<T>>(std::move(data));
    }

    inline TSharedWeights<float> Share(TWeights<float>&& data) {
        return MakeIntrusive<TWeights<float>>(std::move(data));
    }

    }

}

