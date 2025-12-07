#include "generic.h"

#include "thread_local.h"

namespace {
    class TThreadLocalStorage
        : public NThreading::IGenericLocalStorage
    {
    public:
        TData* GetData() const override {
            return Data_.Get();
        }
    private:
        NThreading::TThreadLocalValue<TData, NThreading::EThreadLocalImpl::StdThreadLocal> Data_;
    };

    std::atomic<size_t>& DefaultFactoryUsageCounter() {
        static std::atomic<size_t> v;
        return v;
    }

    auto& genericLocalStorageFactory() {
        static NThreading::TGenericLocalStorageFactory factory = [] {
            DefaultFactoryUsageCounter() += 1;
            return MakeHolder<TThreadLocalStorage>();
        };

        return factory;
    }
}

namespace NThreading {
    void SetGenericLocalStorageFactory(TGenericLocalStorageFactory factory) {
        Y_ENSURE(DefaultFactoryUsageCounter() == 0, "There are some thread local values allocated with default factory");

        genericLocalStorageFactory() = factory;
    }

    THolder<IGenericLocalStorage> MakeGenericLocalStorage() {
        return genericLocalStorageFactory()();
    }
}
