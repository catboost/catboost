#pragma once

#include "cuda_base.h"
#include "remote_objects.h"

namespace NKernelHost {
    using EPtrType = NCudaLib::EPtrType;
    using TCudaStream = NCudaLib::TCudaStream;
    using uchar = unsigned char;
    using ulong = unsigned long;

    class IMemoryManager {
    protected:
        virtual ui64 AllocateImpl(EPtrType ptrType,
                                  ui64 size) = 0;

    public:
        virtual ~IMemoryManager() {
        }

        template <class T, EPtrType PtrType = EPtrType::CudaDevice>
        NCudaLib::THandleBasedMemoryPointer<T, PtrType> Allocate(ui64 count) {
            ui64 handle = AllocateImpl(PtrType, count * sizeof(T));
            return NCudaLib::THandleBasedMemoryPointer<T, PtrType>(handle);
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

    template <class TContext = void>
    class TKernelWithContext {
    public:
        inline static TContext* EmptyContext() {
            return NHelpers::TContextCreator<TContext>::Create();
        }
    };

    template <class TContext = void,
              bool NeedPostProcessFlag = false>
    class TKernelBase: public TKernelWithContext<TContext> {
    public:
        static constexpr bool NeedPostProcess() {
            return NeedPostProcessFlag;
        }
    };

    using TStatelessKernel = TKernelBase<void, false>;

    template <class TBuffer>
    inline void EnsureUnsignedInteger(const TBuffer& buffer) {
        CB_ENSURE(buffer.Size() < (1ULL << 32));
    }

}
