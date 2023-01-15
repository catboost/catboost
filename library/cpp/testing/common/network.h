#pragma once

#include <util/generic/ptr.h>
#include <util/generic/vector.h>

namespace NTesting {

    //@brief network port holder interface
    class IPort {
    public:
        virtual ~IPort() {}

        virtual ui16 Get() = 0;
    };

    class TPortHolder : private THolder<IPort> {
    public:
        using THolder<IPort>::THolder;
        using THolder<IPort>::Release;

        operator ui16() const& {
            return (*this)->Get();
        }

        operator ui16() const&& = delete;
    };

    IOutputStream& operator<<(IOutputStream& out, const TPortHolder& port);

    //@brief Get first free port.
    [[nodiscard]] TPortHolder GetFreePort();

    namespace NLegacy {
        // Do not use this method, it needs only for TPortManager from unittests.
        // Returns continuous sequence of the specified number of ports.
        [[nodiscard]] TVector<TPortHolder> GetFreePortsRange(size_t count);
    }
}
