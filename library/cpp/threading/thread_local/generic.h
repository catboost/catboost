#pragma once

#include <util/generic/ptr.h>
#include <util/generic/vector.h>

#include <functional>

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

    using TGenericLocalStorageFactory = std::function<THolder<IGenericLocalStorage>()>;

    void SetGenericLocalStorageFactory(TGenericLocalStorageFactory factory);
    THolder<IGenericLocalStorage> MakeGenericLocalStorage();

    template <typename T>
    class TGenericLocalValue {
    private:
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
            return static_cast<T*>(Storage_->GetMemory(Traits()));
        }

        T& GetRef() const {
            return *Get();
        }
    private:
        THolder<IGenericLocalStorage> Storage_ = MakeGenericLocalStorage();
    };
}

