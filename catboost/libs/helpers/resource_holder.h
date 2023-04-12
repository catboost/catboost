#pragma once

#include <util/generic/ptr.h>
#include <util/generic/vector.h>
#include <util/thread/lfqueue.h>


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

    template <class T>
    struct TLockFreeQueueHolder : public IResourceHolder {
        TLockFreeQueue<T> Data;

    public:
        TLockFreeQueueHolder() = default;
    };

}

