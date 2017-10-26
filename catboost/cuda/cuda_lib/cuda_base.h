#pragma once

#include <cuda_runtime.h>
#include <util/system/types.h>
#include <catboost/libs/logging/logging.h>
#include <util/generic/noncopyable.h>
#include <util/generic/yexception.h>
#include <util/thread/singleton.h>
#include <catboost/libs/helpers/exception.h>

#define CUDA_SAFE_CALL(statement)                                                                            \
    {                                                                                                        \
        cudaError_t errorCode = statement;                                                                   \
        if (errorCode != cudaSuccess && errorCode != cudaErrorCudartUnloading) {                             \
            ythrow yexception() << "CUDA error: " << cudaGetErrorString(errorCode) << " " << (int)errorCode; \
        }                                                                                                    \
    }

namespace NCudaLib {
    class TCudaStream: private TNonCopyable {
    private:
        cudaStream_t Stream;

    public:
        TCudaStream() {
            CUDA_SAFE_CALL(cudaStreamCreate(&Stream));
        }

        void Synchronize() const {
            CUDA_SAFE_CALL(cudaStreamSynchronize(Stream));
        }

        ~TCudaStream() throw (yexception) {
            CUDA_SAFE_CALL(cudaStreamDestroy(Stream));
        }

        cudaStream_t GetStream() const {
            return Stream;
        }
    };

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

    enum EPtrType {
        CudaDevice,
        CudaHost, //pinned cuda memory
        Host      //CPU, non-pinned
    };

    inline constexpr bool IsHostPtr(EPtrType type) {
        return type != CudaDevice;
    }

    template <EPtrType From, EPtrType To>
    struct TMemoryCopyKind {
        static constexpr cudaMemcpyKind Kind() {
            return cudaMemcpyDefault;
        }
    };

    template <>
    struct TMemoryCopyKind<CudaDevice, CudaDevice> {
        static constexpr cudaMemcpyKind Kind() {
            return cudaMemcpyDeviceToDevice;
        }
    };

    template <>
    struct TMemoryCopyKind<CudaDevice, CudaHost> {
        static constexpr cudaMemcpyKind Kind() {
            return cudaMemcpyDeviceToHost;
        }
    };

    template <>
    struct TMemoryCopyKind<CudaHost, CudaDevice> {
        static constexpr cudaMemcpyKind Kind() {
            return cudaMemcpyHostToDevice;
        }
    };

    template <>
    struct TMemoryCopyKind<CudaHost, CudaHost> {
        static constexpr cudaMemcpyKind Kind() {
            return cudaMemcpyHostToHost;
        }
    };

    template <EPtrType>
    class TCudaMemoryAllocation;

    template <>
    class TCudaMemoryAllocation<CudaDevice> {
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

    template <>
    class TCudaMemoryAllocation<CudaHost> {
    public:
        template <class T>
        static T* Allocate(ui64 size) {
            T* ptr = nullptr;
            CUDA_SAFE_CALL(cudaHostAlloc((void**)&ptr, size * sizeof(T), cudaHostAllocPortable));
            return ptr;
        }

        template <class T>
        static void FreeMemory(T* ptr) {
            CUDA_SAFE_CALL(cudaFreeHost((void*)ptr));
        }
    };

    template <>
    class TCudaMemoryAllocation<Host> {
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

    inline void DeviceSynchronize() {
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
    }

    inline void SetDevice(int devId) {
        CUDA_SAFE_CALL(cudaSetDevice(devId));
    }

    inline void CheckLastError() {
        CUDA_SAFE_CALL(cudaGetLastError());
    }

    namespace NCudaHelpers {
        inline int GetDeviceCount() {
            int deviceCount = 0;
            CUDA_SAFE_CALL(cudaGetDeviceCount(&deviceCount))
            return deviceCount;
        }
    }

    class TCudaDeviceProperties {
    private:
        cudaDeviceProp Props;

    public:
        TCudaDeviceProperties(const TCudaDeviceProperties& other) = default;

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
    };

    namespace NCudaHelpers {
        inline TCudaDeviceProperties GetDeviceProps(int dev) {
            cudaDeviceProp deviceProp;
            CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, dev));
            return TCudaDeviceProperties(deviceProp);
        }

        inline ui64 GetDeviceMemory(int dev) {
            auto props = GetDeviceProps(dev);
            return props.GetDeviceMemory();
        }
    }

    class TDevicesList {
    public:
        class TDeviceListIterator {
        private:
            ui64 Mask;
            ui32 Dev;

        public:
            inline TDeviceListIterator()
                : Mask(0)
                , Dev(0)
            {
            }

            inline TDeviceListIterator(const ui64 devMask,
                                       const ui32 dev)
                : Mask(devMask)
                , Dev(dev)
            {
            }

            inline bool operator!=(const TDeviceListIterator& other) {
                Y_ASSERT(Mask == other.Mask);
                return Dev != other.Dev;
            }

            inline const TDeviceListIterator& operator++() {
                ui64 bit = ((ui64)1) << Dev;
                const ui32 maxDevices = sizeof(decltype(bit)) * 8;
                Y_ASSERT(Dev < maxDevices);

                do {
                    ++Dev;
                    bit *= 2;
                } while ((Dev < maxDevices) && !(Mask & bit));
                return *this;
            }

            inline const ui32& operator*() const {
                return Dev;
            }
        };

    public:
        explicit TDevicesList(ui64 mask = 0)
            : Mask(mask)
        {
            if (mask) {
                LastDev = MostSignificantBit(mask);
                Offset = LeastSignificantBit(mask);
                ui64 endFlag = (1ULL << (LastDev + 1));
                CB_ENSURE(!(endFlag & Mask));
                Mask |= endFlag;
            }
        }

        static TDevicesList SingleDevice(ui64 devId) {
            return TDevicesList(1ULL << (devId));
        }

        TDevicesList(TDevicesList&& other) = default;
        TDevicesList(const TDevicesList& other) = default;
        TDevicesList& operator=(TDevicesList&& other) = default;
        TDevicesList& operator=(const TDevicesList& other) = default;

        inline TDeviceListIterator begin() const {
            return TDeviceListIterator(Mask, Offset);
        }

        inline TDeviceListIterator end() const {
            const ui64 end = Mask ? LastDev + 1 : 0;
            return TDeviceListIterator(Mask, end);
        }

        inline bool HasDevice(int devId) const {
            return Mask & (((ui64)1) << devId);
        }

    private:
        ui64 Mask = 0;
        ui32 Offset = 0;
        ui32 LastDev = 0;
    };
}
