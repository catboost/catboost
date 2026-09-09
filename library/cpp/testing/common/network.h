#pragma once

#include <util/generic/noncopyable.h>
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
        using TBase = THolder<IPort>;
    public:
        using TBase::TBase;
        using TBase::Release;
        using TBase::Reset;

        operator ui16() const& {
            return (*this)->Get();
        }

        operator ui16() const&& = delete;
    };

    IOutputStream& operator<<(IOutputStream& out, const TPortHolder& port);

    //@brief Get first free port.
    [[nodiscard]] TPortHolder GetFreePort();

    // Owns all allocated ports until the manager is destroyed. Unlike the
    // legacy unittest TPortManager, this class has no test-framework
    // dependencies and may be used by both unittest and gtest binaries.
    class TPortManager: public TNonCopyable {
    public:
        TPortManager();
        ~TPortManager();

        ui16 GetPort(ui16 port = 0);
        ui16 GetTcpPort(ui16 port = 0);
        ui16 GetUdpPort(ui16 port = 0);
        ui16 GetTcpAndUdpPort(ui16 port = 0);
        ui16 GetPortsRange(ui16 startPort, ui16 range);

    private:
        class TImpl;
        THolder<TImpl> Impl_;
    };

    namespace NLegacy {
        // Do not use these methods made for Unittest TPortManager backward compatibility.
        // Returns continuous sequence of the specified number of ports.
        [[nodiscard]] TVector<TPortHolder> GetFreePortsRange(size_t count);
        //@brief Returns port from parameter if NO_RANDOM_PORTS env var is set, otherwise first free port
        [[nodiscard]] TPortHolder GetPort(ui16 port);
    }

    //@brief Reinitialize singleton from environment vars for tests
    void InitPortManagerFromEnv();

    //@brief helper class for inheritance
    struct TFreePortOwner {
        TFreePortOwner() : Port_(GetFreePort()) {}

        ui16 GetPort() const {
            return Port_;
        }

    private:
        TPortHolder Port_;
    };
}
