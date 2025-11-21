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
        NThreading::TThreadLocalValue<TData> Data_;
    };

    NThreading::TGenericLocalStorageFactory genericLocalStorageFactory = []() {
        return MakeHolder<TThreadLocalStorage>();
    };
}

namespace NThreading {
    void SetGenericLocalStorageFactory(TGenericLocalStorageFactory factory) {
        genericLocalStorageFactory = factory;
    }

    THolder<IGenericLocalStorage> MakeGenericLocalStorage() {
        return genericLocalStorageFactory();
    }
}
