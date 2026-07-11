#pragma once

#include <util/generic/ptr.h>
#include <util/generic/vector.h>

#include <array>
#include <functional>
#include <mutex>

namespace NThreading {

    class IGenericLocalStorage {
    public:
        struct TTraits {
            size_t Size = 0;
            std::function<void(void*)> Constructor;
            std::function<void(void*)> Destructor;
        };

        struct TData
            : TNonCopyable
        {
            TVector<char> Memory;
            std::function<void(void*)> Destructor;
        public:
            ~TData() {
                if (Destructor) {
                    Destructor(&Memory[0]);
                }
            }
        };
    public:
        virtual ~IGenericLocalStorage() {};

        void* GetMemory(const TTraits& traits) const {
            TData* data = GetData();
            if (!data->Destructor) {
                data->Destructor = traits.Destructor;
                data->Memory.resize(traits.Size);
                traits.Constructor(&data->Memory[0]);
            }
            return &data->Memory[0];
        }
    private:
        virtual TData* GetData() const = 0;
    };

    class IGLSContext {
    public:
        virtual ~IGLSContext() = default;

        virtual bool IsCurrent() const = 0;
        virtual THolder<IGenericLocalStorage> MakeStorage() const = 0;
    };

    // Later registrations take priority over earlier ones.
    void RegisterGLSContext(THolder<IGLSContext> context);

    namespace NDetail {
        inline constexpr size_t MaxGLSContexts = 4;

        size_t GLSContextCount();
        const IGLSContext& GetGLSContext(size_t index);
    }

    template <typename T>
    class TGenericLocalValue {
    private:
        struct TSlot {
            std::once_flag InitOnce;
            THolder<IGenericLocalStorage> Storage;
        };

        static const auto& Traits() {
            const static IGenericLocalStorage::TTraits traits = {
                .Size = sizeof(T),
                .Constructor = [](void* addr) { new (addr) T(); },
                .Destructor = [](void* addr) { static_cast<T*>(addr)->~T(); }
            };

            return traits;
        };
    public:
        T* Get() const {
            const size_t count = NDetail::GLSContextCount();
            for (size_t index = count; index-- > 0;) {
                const IGLSContext& context = NDetail::GetGLSContext(index);
                if (context.IsCurrent()) {
                    return GetMemory(ContextSlots_[index], context);
                }
            }

            Y_ABORT("unreachable, ContextSlots_[0] always IsCurrent");
        }

        T& GetRef() const {
            return *Get();
        }
    private:
        T* GetMemory(TSlot& slot, const IGLSContext& context) const {
            std::call_once(slot.InitOnce, [&] { slot.Storage = context.MakeStorage(); });
            return static_cast<T*>(slot.Storage->GetMemory(Traits()));
        }
    private:
        mutable std::array<TSlot, NDetail::MaxGLSContexts> ContextSlots_;
    };
}
