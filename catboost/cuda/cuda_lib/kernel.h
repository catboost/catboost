#pragma once

#include "cuda_base.h"

namespace NKernelHost {
    using EPtrType = NCudaLib::EPtrType;
    using TCudaStream = NCudaLib::TCudaStream;
    using uchar = unsigned char;
    using ulong = unsigned long;

    class IMemoryManager {
    protected:
        virtual void* AllocateImpl(EPtrType ptrType,
                                   ui64 size) = 0;

    public:
        virtual ~IMemoryManager() {
        }

        template <class T, EPtrType ptrType = EPtrType::CudaDevice>
        T* Allocate(ui64 count) {
            void* ptr = AllocateImpl(ptrType, count * sizeof(T));
            return reinterpret_cast<T*>(ptr);
        };
    };

    namespace NHelpers {
        template <class T>
        class TContextCreator {
        public:
            inline static T* Create() {
                return new T;
            }
        };

        template <>
        class TContextCreator<void> {
        public:
            inline static void* Create() {
                return nullptr;
            }
        };
    }

    template <class TContext = void, bool NeedPostProcessFlag = false>
    class TKernelBase {
    public:
        static constexpr bool NeedPostProcess() {
            return NeedPostProcessFlag;
        }

        inline static TContext* EmptyContext() {
            return NHelpers::TContextCreator<TContext>::Create();
        }
    };

    using TStatelessKernel = TKernelBase<void, false>;

    template <class TBuffer>
    inline void EnsureUnsignedInteger(const TBuffer& buffer) {
        CB_ENSURE(buffer.Size() < (1ULL << 32));
    }
}
