#pragma once

#include <util/generic/ptr.h>
#include <util/generic/vector.h>


namespace NCB {

    class IResourceHolder : public TThrRefBase {
    };

    template <class T>
    struct TVectorHolder : public IResourceHolder {
        TVector<T> Data;

    public:
        TVectorHolder() = default;

        explicit TVectorHolder(TVector<T>&& data)
            : Data(std::move(data))
        {}
    };

}

