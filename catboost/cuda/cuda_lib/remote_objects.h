#pragma once

#include "cuda_base.h"
#include "memory_provider_trait.h"
#include <util/generic/vector.h>
#include <util/generic/ptr.h>
#include <util/generic/yexception.h>
#include <util/thread/singleton.h>
#include <util/system/spinlock.h>
#include <util/ysaveload.h>
#include <array>

namespace NCudaLib {
    class TObjectByHandleStorage {
    private:
        const static ui64 NULLPTR_HANDLE = 0;
        const static ui64 MAX_HANDLE_COUNT = 1000000;
        std::array<void*, MAX_HANDLE_COUNT> Objects;
        TVector<ui64> FreeHandles;
        TAdaptiveLock Lock;
        ui64 LastFreeHandle = 0;

    public:
        TObjectByHandleStorage() {
            for (ui64 i = 0; i < MAX_HANDLE_COUNT; ++i) {
                Objects[i] = nullptr;
            }
            ++LastFreeHandle;
        }

        TVector<ui64> GetHandle(ui64 count = 1) {
            TGuard<TAdaptiveLock> guard(Lock);

            TVector<ui64> result;
            for (ui64 i = 0; i < count; ++i) {
                if (FreeHandles.size() == 0) {
                    result.push_back(LastFreeHandle++);
                    CB_ENSURE(LastFreeHandle < MAX_HANDLE_COUNT, "Error: too many handles. For performance reasons handle storage has compile-time size limit");
                } else {
                    const ui64 freeHandle = FreeHandles.back();
                    FreeHandles.pop_back();
                    result.push_back(freeHandle);
                }
            }
            return result;
        }

        void FreeHandle(ui64 handle) {
            CB_ENSURE(Objects[handle] == nullptr, "Error: free memory before handle remove");
            TGuard<TAdaptiveLock> guard(Lock);
            FreeHandles.push_back(handle);
        }

        void* GetObjectPtrByHandle(ui64 handle) const {
            CB_ENSURE(handle < Objects.size(), "Handle should be valid " << handle);
            return Objects[handle];
        }

        void SetObjectPtrByHandle(ui64 handle,
                                  void* ptr) {
            if (handle == NULLPTR_HANDLE) {
                CB_ENSURE(ptr == nullptr);
                return;
            }
            Y_ASSERT(handle < Objects.size());
            Objects[handle] = ptr;
        }

        static constexpr bool IsNullptr(ui64 handle) {
            return handle == NULLPTR_HANDLE;
        }
    };

    static inline TObjectByHandleStorage& GetHandleStorage() {
        return *HugeSingleton<TObjectByHandleStorage>();
    }

    namespace NHelpers {
        class THandleCache {
        private:
            mutable void* CachedPtr = nullptr;

        public:
            void* GetPtr(ui64 handle) const {
                if (CachedPtr == nullptr) {
                    CachedPtr = GetHandleStorage().GetObjectPtrByHandle(handle);
                }
                return CachedPtr;
            }
        };
    }

    //it's only view pointer. Objects deallocation will be done by master commands
    template <class T>
    class THandleBasedPointer: public TPointerCommon<THandleBasedPointer<T>, T> {
    private:
        ui64 Handle;
        NHelpers::THandleCache Cache;

    public:
        explicit THandleBasedPointer(ui64 handle)
            : Handle(handle)
        {
        }

        THandleBasedPointer(const THandleBasedPointer&) = default;
        THandleBasedPointer(THandleBasedPointer&&) = default;

        T* Get() const {
            return reinterpret_cast<T*>(Cache.GetPtr(Handle));
        }

        void Reset() {
            Reset(nullptr);
        }

        void Reset(T* ptr) {
            T* obj = Get();
            if (obj != nullptr) {
                delete obj;
            }
            GetHandleStorage().SetObjectPtrByHandle(Handle, ptr);
        }

        static THandleBasedPointer<T> GetNullptr() {
            return THandleBasedPointer<T>(0);
        }

        Y_SAVELOAD_DEFINE(Handle);
    };

    //pointers for memory
    template <class T, EPtrType Type>
    class THandleBasedMemoryPointer {
    private:
        using TRawDataPtr = typename TMemoryProviderImplTrait<Type>::TRawFreeMemory;
        ui64 Handle;
        ui64 Offset;

        friend struct THandleRawPtr;

    public:
        THandleBasedMemoryPointer()
            : Handle(0)
            , Offset(0)
        {
        }

        explicit THandleBasedMemoryPointer(ui64 handle,
                                           ui64 offset = 0)
            : Handle(handle)
            , Offset(offset)
        {
        }

        THandleBasedMemoryPointer(const THandleBasedMemoryPointer& other) = default;

        THandleBasedMemoryPointer(const THandleBasedMemoryPointer& other, ui64 offset)
            : Handle(other.Handle)
            , Offset(other.Offset + offset)
        {
        }

        THandleBasedMemoryPointer& operator=(const THandleBasedMemoryPointer& other) = default;

        operator THandleBasedMemoryPointer<const T, Type>() {
            return THandleBasedMemoryPointer<const T, Type>(Handle, Offset);
        };

        THandleBasedMemoryPointer<char, Type> GetRawHandleBasedPtr() {
            return THandleBasedMemoryPointer<char, Type>(Handle, sizeof(T) * Offset);
        }

        T* Get() const {
            if (Handle == 0) {
                return nullptr;
            }
            char* ptr = THandleBasedPointer<TRawDataPtr>(Handle)->Get();
            return reinterpret_cast<T*>(ptr) + Offset;
        }

        operator T*() {
            return Get();
        };

        Y_SAVELOAD_DEFINE(Handle, Offset);
    };

    struct THandleRawPtr {
        EPtrType Type;
        ui64 Handle = 0;
        ui64 Offset = 0;

        THandleRawPtr(EPtrType type, ui64 handle, ui64 offset)
            : Type(type)
            , Handle(handle)
            , Offset(offset)
        {
        }

        template <class T, EPtrType PtrType>
        explicit THandleRawPtr(THandleBasedMemoryPointer<T, PtrType>& ptr) {
            Type = PtrType;
            Handle = ptr.Handle;
            Offset = sizeof(T) * ptr.Offset;
        }

        THandleRawPtr() = default;
        THandleRawPtr(const THandleRawPtr&) = default;
        THandleRawPtr(THandleRawPtr&&) = default;
        THandleRawPtr& operator=(THandleRawPtr&&) = default;
        THandleRawPtr& operator=(const THandleRawPtr&) = default;

        bool IsNullptr() const {
            return Handle == 0;
        }

        inline char* GetRawPtr() const {
            switch (Type) {
                case NCudaLib::EPtrType::CudaDevice: {
                    return NCudaLib::THandleBasedMemoryPointer<char, NCudaLib::EPtrType::CudaDevice>(Handle, Offset).Get();
                }
                case NCudaLib::EPtrType::CudaHost: {
                    return NCudaLib::THandleBasedMemoryPointer<char, NCudaLib::EPtrType::CudaHost>(Handle, Offset).Get();
                }
                case NCudaLib::EPtrType::Host: {
                    return NCudaLib::THandleBasedMemoryPointer<char, NCudaLib::EPtrType::Host>(Handle, Offset).Get();
                }
                default: {
                    ythrow TCatBoostException() << "Error: unknown ptr type";
                }
            }
            Y_UNREACHABLE();
        }

        Y_SAVELOAD_DEFINE(Type, Handle, Offset);
    };

}
