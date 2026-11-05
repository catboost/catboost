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

    class TThreadLocalContext
        : public NThreading::IGLSContext
    {
        bool IsCurrent() const override {
            return true;
        }

        THolder<NThreading::IGenericLocalStorage> MakeStorage() const override {
            return MakeHolder<TThreadLocalStorage>();
        }
    };

    class TContextRegistry {
    public:
        TContextRegistry() {
            Register(MakeHolder<TThreadLocalContext>());
        }

        size_t Count() const {
            return Count_.load();
        }

        const NThreading::IGLSContext& Get(size_t index) const {
            return *Contexts_[index];
        }

        void Register(THolder<NThreading::IGLSContext> context) {
            with_lock (Lock_) {
                const size_t index = Count_.load();
                Y_ENSURE(index < NThreading::NDetail::MaxGLSContexts, "Too many generic local contexts registered");
                Contexts_[index] = std::move(context);
                Count_.store(index + 1);
            }
        }
    private:
        TAdaptiveLock Lock_;
        std::atomic<size_t> Count_ = 0;
        std::array<THolder<NThreading::IGLSContext>, NThreading::NDetail::MaxGLSContexts> Contexts_ = {};
    };

    TContextRegistry& Registry() {
        static TContextRegistry registry;
        return registry;
    }
}

namespace NThreading {
    void RegisterGLSContext(THolder<IGLSContext> context) {
        Registry().Register(std::move(context));
    }

    namespace NDetail {
        size_t GLSContextCount() {
            return Registry().Count();
        }

        const IGLSContext& GetGLSContext(size_t index) {
            return Registry().Get(index);
        }
    }
}
