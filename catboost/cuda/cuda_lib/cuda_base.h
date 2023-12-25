#pragma once

#include "fwd.h"

#include <cuda_runtime.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/logging/logging.h>
#include <util/system/types.h>
#include <util/generic/vector.h>
#include <util/generic/noncopyable.h>
#include <util/generic/yexception.h>
#include <util/thread/singleton.h>
#include <util/ysaveload.h>
#include <util/system/spinlock.h>


static_assert(std::is_pod<cudaDeviceProp>::value, "cudaDeviceProp is not pod type");
Y_DECLARE_PODTYPE(cudaDeviceProp);

//cuda-types
Y_DECLARE_PODTYPE(uint2);
Y_DECLARE_PODTYPE(uint4);

#define CUDA_SAFE_CALL(statement)                                                                                    \
    {                                                                                                                \
        cudaError_t errorCode = statement;                                                                           \
        if (errorCode != cudaSuccess && errorCode != cudaErrorCudartUnloading) {                                     \
            ythrow TCatBoostException() << "CUDA error " << (int)errorCode << ": " << cudaGetErrorString(errorCode); \
        }                                                                                                            \
    }

namespace NCudaLib {
    class TCudaStreamsProvider: public TNonCopyable {
    private:
        TVector<cudaStream_t> Streams;

    private:
        cudaStream_t NewStream();

    public:
        class TCudaStream: private TMoveOnly {
        private:
            cudaStream_t Stream = 0;
            TCudaStreamsProvider* Owner = nullptr;

        public:
            TCudaStream(cudaStream_t stream,
                        TCudaStreamsProvider* owner)
                : Stream(stream)
                , Owner(owner)
            {
            }

            TCudaStream(TCudaStream&& other) = default;

            ~TCudaStream() {
                if (Stream && Owner) {
                    Owner->Streams.push_back(Stream);
                }
            }

            void Synchronize() const {
                CUDA_SAFE_CALL(cudaStreamSynchronize(Stream));
            }

            operator cudaStream_t() const {
                return GetStream();
            }

            cudaStream_t GetStream() const {
                return Stream;
            }
        };

        ~TCudaStreamsProvider() noexcept(false) {
            for (auto& stream : Streams) {
                CUDA_SAFE_CALL(cudaStreamDestroy(stream));
            }
        }

        TCudaStream RequestStream() {
            if (Streams.size()) {
                cudaStream_t stream = Streams.back();
                Streams.pop_back();
                return TCudaStream(stream, this);
            } else {
                return TCudaStream(NewStream(), this);
            }
        }
    };

    using TCudaStream = TCudaStreamsProvider::TCudaStream;

    inline TCudaStreamsProvider& GetStreamsProvider() {
        return *FastTlsSingleton<TCudaStreamsProvider>();
    }

    class TDefaultStreamRef {
    private:
        TCudaStream* Stream = nullptr;

    public:
        void SetDefaultStream(TCudaStream& stream) {
            Stream = &stream;
        }

        TCudaStream& Get() {
            CB_ENSURE(Stream != nullptr, "Error: initialize default stream for thread on launch");
            return *Stream;
        }
    };

    static inline TCudaStream& GetDefaultStream() {
        return FastTlsSingleton<TDefaultStreamRef>()->Get();
    }

    static inline void SetDefaultStream(TCudaStream& stream) {
        FastTlsSingleton<TDefaultStreamRef>()->SetDefaultStream(stream);
    }

    inline constexpr bool IsHostPtr(EPtrType type) {
        return type != EPtrType::CudaDevice;
    }

    template <class T>
    EPtrType GetPointerType(const T* ptr) {
        cudaPointerAttributes attributes;
        CUDA_SAFE_CALL(cudaPointerGetAttributes(&attributes, (void*)(ptr)));
        //TODO(noxoomo): currently don't distinguish pinned/non-pinned memory
#ifndef CUDART_VERSION
#error "CUDART_VERSION is not defined: include cuda_runtime_api.h"
#elif (CUDART_VERSION >= 10000)
        return attributes.type == cudaMemoryTypeHost ? EPtrType::CudaHost : EPtrType::CudaDevice;
#else
        return attributes.memoryType == cudaMemoryTypeHost ? EPtrType::CudaHost : EPtrType::CudaDevice;
#endif
    }

    template <EPtrType From, EPtrType To>
    struct TMemoryCopyKind {
        static constexpr cudaMemcpyKind Kind() {
            return cudaMemcpyDefault;
        }
    };

    template <>
    struct TMemoryCopyKind<EPtrType::CudaDevice, EPtrType::CudaDevice> {
        static constexpr cudaMemcpyKind Kind() {
            return cudaMemcpyDeviceToDevice;
        }
    };

    template <>
    struct TMemoryCopyKind<EPtrType::CudaDevice, EPtrType::CudaHost> {
        static constexpr cudaMemcpyKind Kind() {
            return cudaMemcpyDeviceToHost;
        }
    };

    template <>
    struct TMemoryCopyKind<EPtrType::CudaHost, EPtrType::CudaDevice> {
        static constexpr cudaMemcpyKind Kind() {
            return cudaMemcpyHostToDevice;
        }
    };

    template <>
    struct TMemoryCopyKind<EPtrType::CudaHost, EPtrType::CudaHost> {
        static constexpr cudaMemcpyKind Kind() {
            return cudaMemcpyHostToHost;
        }
    };

    template <EPtrType>
    class TCudaMemoryAllocation;

    template <>
    class TCudaMemoryAllocation<EPtrType::CudaDevice> {
    public:
        template <class T>
        static T* Allocate(ui64 size) {
            T* ptr = nullptr;
            CUDA_SAFE_CALL(cudaMalloc((void**)&ptr, size * sizeof(T)));
            return ptr;
        }

        template <class T>
        static void FreeMemory(T* ptr) {
            CUDA_SAFE_CALL(cudaFree((void*)ptr));
        }
    };

    char* CudaHostAllocate(ui64 size);
    void CudaHostFree(char* ptr);

    template <>
    class TCudaMemoryAllocation<EPtrType::CudaHost> {
    public:
        template <class T>
        static T* Allocate(ui64 size) {
            T* ptr = CudaHostAllocate(size * sizeof(T));
            return ptr;
        }

        template <class T>
        static void FreeMemory(T* ptr) {
            CudaHostFree(ptr);
        }
    };

    template <>
    class TCudaMemoryAllocation<EPtrType::Host> {
    public:
        template <class T>
        static T* Allocate(ui64 size) {
            return new T[size];
        }

        template <class T>
        static void FreeMemory(T* ptr) {
            delete[] ptr;
        }
    };

    template <EPtrType From, EPtrType To>
    class TMemoryCopier {
    public:
        template <class T>
        static void CopyMemoryAsync(const T* from, T* to, ui64 size, const TCudaStream& stream) {
            CUDA_SAFE_CALL(cudaMemcpyAsync(static_cast<void*>(to), static_cast<void*>(const_cast<T*>(from)), sizeof(T) * size, TMemoryCopyKind<From, To>::Kind(), stream.GetStream()));
        }

        template <class T>
        static void CopyMemorySync(T* from, T* to, ui64 size) {
            TCudaStream& stream = GetDefaultStream();
            CopyMemoryAsync<T>(from, to, size, stream);
            stream.Synchronize();
        }
    };

    template <class T>
    static void CopyMemoryAsync(const T* from, T* to, ui64 size, const TCudaStream& stream) {
        CUDA_SAFE_CALL(cudaMemcpyAsync(static_cast<void*>(to), static_cast<void*>(const_cast<T*>(from)), sizeof(T) * size, cudaMemcpyDefault, stream.GetStream()));
    }

    inline void DeviceSynchronize() {
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
    }

    inline void SetDevice(int devId) {
        CUDA_SAFE_CALL(cudaSetDevice(devId));
    }

    inline int GetDevice() {
        int devId;
        CUDA_SAFE_CALL(cudaGetDevice(&devId));
        return devId;
    }

    inline void CheckLastError() {
        CUDA_SAFE_CALL(cudaGetLastError());
    }

    template <class T>
    inline int GetDeviceForPointer(const T* ptr) {
        cudaPointerAttributes result;
        CUDA_SAFE_CALL(cudaPointerGetAttributes(&result, (const void*)ptr));
#ifndef CUDART_VERSION
#error "CUDART_VERSION is not defined: include cuda_runtime_api.h"
#elif (CUDART_VERSION >= 10000)
        CB_ENSURE(result.type == cudaMemoryTypeDevice, "Error: this pointer is not GPU pointer");
#else
        CB_ENSURE(result.memoryType == cudaMemoryTypeDevice, "Error: this pointer is not GPU pointer");
#endif
        return result.device;
    }

    namespace NCudaHelpers {
        inline int GetDeviceCount() {
            int deviceCount = 0;
            CUDA_SAFE_CALL(cudaGetDeviceCount(&deviceCount));
            return deviceCount;
        }
    }

    class TCudaDeviceProperties {
    private:
        cudaDeviceProp Props;

    public:
        TCudaDeviceProperties() = default;

        explicit TCudaDeviceProperties(cudaDeviceProp props)
            : Props(props)
        {
        }

        ui64 GetDeviceMemory() const {
            return Props.totalGlobalMem;
        }

        TString GetName() const {
            return Props.name;
        }

        ui64 GetMajor() const {
            return Props.major;
        }

        ui64 GetMinor() const {
            return Props.minor;
        }

        Y_SAVELOAD_DEFINE(Props);
    };

    namespace NCudaHelpers {
        inline TCudaDeviceProperties GetDeviceProps(int dev) {
            cudaDeviceProp deviceProp;
            CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, dev));
            return TCudaDeviceProperties(deviceProp);
        }

        inline TVector<TCudaDeviceProperties> GetDevicesProps() {
            TVector<TCudaDeviceProperties> result;
            for (int dev = 0; dev < GetDeviceCount(); ++dev) {
                result.push_back(NCudaHelpers::GetDeviceProps(dev));
            }
            return result;
        }

        inline ui64 GetDeviceMemory(int dev) {
            auto props = GetDeviceProps(dev);
            return props.GetDeviceMemory();
        }
    }

    class TOutOfMemoryError: public TCatBoostException {
    };

}
