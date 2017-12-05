#pragma once

#include "cuda_base.h"
#include "memory_provider_trait.h"
#include "remote_objects.h"
#include "single_device.h"
#include "devices_provider.h"

#include <util/ysafeptr.h>
#include <util/generic/vector.h>

/*
 * Master-slave control and communication routines
 * CudaManager will be thread-local "singleton"
*/
namespace NCudaLib {
    template <class T>
    class TDeviceObjectExtractor {
    public:
        using TRemoteObject = std::remove_const_t<std::remove_reference_t<T>>;

        static TRemoteObject At(ui32 devId, const T& object) {
            Y_UNUSED(devId);
            return object;
        }
    };

    namespace NHelpers {
        template <class T>
        class TEmptyObjectsHelper {
        public:
            static inline bool IsEmpty(T& val) {
                return val == 0;
            }
        };
    }

    template <class T>
    class TDistributedObject {
    public:
        TDistributedObject(TDistributedObject&& other) = default;

        T At(ui64 devId) const {
            return Data[devId];
        }

        void Set(ui64 devId, T t) {
            Data[devId] = t;
        }

        bool IsEmpty(ui64 devId) const {
            return NHelpers::TEmptyObjectsHelper<T>::IsEmpty(At(devId));
        }

        ui64 DeviceCount() const {
            return Data.size();
        }

    private:
        TVector<T> Data;

    private:
        template <class TC>
        friend void swap(TDistributedObject<TC>&, TDistributedObject<TC>&);

        explicit TDistributedObject(ui32 devCount) {
            Data.resize(devCount);
        }

        friend class TCudaManager;
    };

    template <class T>
    void swap(TDistributedObject<T>& lhs, TDistributedObject<T>& rhs) {
        std::swap(lhs.Data, rhs.Data);
    }

    class TCudaProfiler;

    class TCudaManager {
    private:
        struct TCudaManagerState {
            TVector<TCudaSingleDevice*> Devices;
            TVector<ui64> Streams;
            TVector<ui64> FreeStreams;
            TAdaptiveLock Lock;
        };

        TAtomicSharedPtr<TCudaManagerState> State;
        TCudaProfiler* Profiler = nullptr;
        TCudaProfiler* ParentProfiler = nullptr;

        bool IsChildManager = false;
        TDevicesList DevicesList;
        TAutoEvent OnStopChildEvent;

        TVector<bool> IsActiveDevice;
        bool Locked = false;

        void FreeStream(const ui32 stream) {
            for (ui64 dev = 0; dev < State->Devices.size(); ++dev) {
                State->Devices[dev]->FreeStream(stream);
            }
        }

        TCudaManagerState& GetState() {
            CB_ENSURE(State, "Error: uninitialized cuda manager");
            return *State;
        }

        const TCudaManagerState& GetState() const {
            CB_ENSURE(State, "Error: uninitialized cuda manager");
            return *State;
        }

        void SetDevices(TVector<TCudaSingleDevice*>&& devices) {
            CB_ENSURE(!HasDevices(), "Error: CudaManager already has devices");
            GetState().Devices = std::move(devices);
            GetState().Streams.clear();
            GetState().FreeStreams.clear();
            ui32 stream = GetState().Devices[0]->DefaultStream();

            for (ui64 i = 0; i < GetState().Devices.size(); ++i) {
                CB_ENSURE(stream == GetState().Devices[i]->DefaultStream());
            }

            GetState().Streams.push_back(stream);

            ui64 one = 1;
            DevicesList = TDevicesList((one << (GetDeviceCount())) - 1);
            IsActiveDevice.resize(GetDeviceCount(), true);
        }

        void FreeDevices() {
            auto& provider = GetDevicesProvider();
            for (auto dev : GetState().Devices) {
                provider.Free(dev);
            }
            GetState().Devices.resize(0);
        }

        void CreateProfiler();
        void ResetProfiler(bool printInfo);

        template <typename TDistributedObject>
        inline static auto GetDeviceObject(ui32 devId,
                                           TDistributedObject& object) -> decltype(TDeviceObjectExtractor<std::remove_reference_t<TDistributedObject>>::At(devId, object)) {
            return TDeviceObjectExtractor<std::remove_reference_t<TDistributedObject>>::At(devId, object);
        }

        template <class TKernel, class... Args>
        inline static TKernel GetDeviceKernel(ui32 devId,
                                              Args&... args) {
            return {GetDeviceObject<Args>(devId, args)...};
        };

        friend class TDataCopier;
        friend class TSingleHostStreamSync;
        friend class TChildCudaManagerInitializer;

    public:
        ~TCudaManager();

        bool HasPeerAccess(ui64 from, ui64 to) const {
            return GetState().Devices[from]->HasPeerAccess(GetState().Devices[to]);
        }

        template <class TKernel>
        inline void LaunchKernel(TKernel&& kernel,
                                 ui32 dev,
                                 ui64 devStream) const {
            Y_ASSERT(dev < GetState().Devices.size());
            CB_ENSURE(IsActiveDevice[dev]);
            GetState().Devices[dev]->LaunchKernel(std::move(kernel), devStream);
        }

        template <class TKernel, class... Args>
        inline void LaunchKernels(TDevicesList&& devices,
                                  ui32 streamId,
                                  Args&&... args) {
            auto stream = GetState().Streams[streamId];

            for (ui32 devId : devices) {
                auto kernel = GetDeviceKernel<TKernel, Args...>(devId, args...);
                LaunchKernel<TKernel>(std::move(kernel), devId, stream);
            }
        }

        ui32 GetDeviceCount() const {
            return (ui32)GetState().Devices.size();
        }

        TDevicesList GetActiveDevices() const {
            return DevicesList;
        }

        void DumpFreeMemory(TString message) const;

        double MinFreeMemoryFraction() const;

        double FreeMemoryMb(ui32 devId, bool waitComplete = true) const;

        double TotalMemoryMb(ui32 devId) const;

        //waits for finish all work submitted to selected devices
        void WaitComplete(TDevicesList&& devices) {
            TVector<TDeviceFuture<ui64>> waitComplete;

            for (auto dev : devices) {
                CB_ENSURE(dev < GetState().Devices.size());
                CB_ENSURE(IsActiveDevice[dev], "Device should be active");
                waitComplete.push_back(GetState().Devices[dev]->WaitComplete());
            }

            for (auto& event : waitComplete) {
                event.Wait();
            }
        }

        bool IsLocked() const {
            return Locked;
        }

        void WaitStreamSubmit(TDevicesList&& devices, ui32 stream = 0) {
            TVector<TDeviceFuture<ui64>> waitComplete;
            for (auto dev : devices) {
                CB_ENSURE(dev < GetState().Devices.size());
                CB_ENSURE(IsActiveDevice[dev], "Device should be active");
                waitComplete.push_back(GetState().Devices[dev]->WaitStreamSubmit(stream));
            }
            for (auto& event : waitComplete) {
                event.Wait();
            }
        }

        void WaitStreamSubmit(ui32 stream = 0) {
            WaitStreamSubmit(GetActiveDevices(), stream);
        }

        void WaitComplete() {
            WaitComplete(GetActiveDevices());
        }

        void SyncStream(ui32 stream);

        inline void Start() {
            State.Reset(new TCudaManagerState());
            CB_ENSURE(!HasDevices());
            CB_ENSURE(State, "Error: can't start, state already exists");
            SetDevices(GetDevicesProvider().GetDevices());
            CreateProfiler();
        }

        inline void Stop() {
            CB_ENSURE(!IsChildManager);
            WaitComplete();
            for (ui32 i = 1; i < GetState().Streams.size(); ++i) {
                FreeStream(GetState().Streams[i]);
            }
            GetState().Streams.resize(0);
            GetState().FreeStreams.resize(0);
            FreeDevices();
            ResetProfiler(true);
            State = nullptr;
            GetDevicesProvider().Reset();
        }

        void StopChild();

        void StartChild(TCudaManager& parent,
                        const TDevicesList& devices,
                        TAutoEvent& stopEvent);

        TCudaProfiler& GetProfiler() {
            CB_ENSURE(Profiler, "Error: nullptr profiler");
            return *Profiler;
        }

        bool HasDevices() const {
            return GetState().Devices.size() > 0;
        }

        template <class T>
        TDistributedObject<T> CreateDistributedObject() {
            return TDistributedObject<T>(GetDeviceCount());
        }

        template <class T, EPtrType Type>
        TCudaSingleDevice::TSingleBuffer<T, Type> CreateSingleBuffer(ui64 devId, ui64 size) {
            return GetState().Devices[devId]->CreateSingleBuffer<T, Type>(size);
        };

        class TComputationStream: public TMoveOnly {
        private:
            TCudaManager* Owner;
            ui64 Id;

            TComputationStream(ui64 id, TCudaManager* owner)
                : Owner(owner)
                , Id(id)
            {
            }

            friend class TCudaManager;

        public:
            TComputationStream(TComputationStream&& other)
                : Owner(other.Owner)
                , Id(other.Id)
            {
                other.Id = 0;
            }

            TComputationStream& operator=(TComputationStream&& other) noexcept {
                Id = other.Id;
                Owner = other.Owner;
                other.Id = 0;
                return *this;
            }

            ~TComputationStream() {
                if (Id != 0) {
                    Owner->GetState().FreeStreams.push_back(Id);
                }
            }

            //sync on remote. ensures that job done in this stream will be seen for all other jobs submitted to other stream.
            //device-scope, don't guarantee any sync between devices
            void Synchronize(TDevicesList&& devices) {
                auto& stream = Owner->GetState().Streams[Id];
                for (auto dev : devices) {
                    Owner->GetState().Devices[dev]->StreamSynchronize(stream);
                }
            }

            ui32 GetId() const {
                return Id;
            }

            operator ui32() const {
                return static_cast<ui32>(Id);
            }

            ui64 At(ui32 dev) const {
                Y_UNUSED(dev);
                return Owner->GetState().Streams[Id];
            }

            void Synchronize() {
                Synchronize(Owner->GetActiveDevices());
            }
        };

        TComputationStream RequestStream() {
            //for child manager stream support we need lock
            TGuard<TAdaptiveLock> guard(GetState().Lock);

            if (GetState().FreeStreams.size() == 0) {
                TDistributedObject<ui64> stream = CreateDistributedObject<ui64>();
                for (ui64 dev = 0; dev < stream.DeviceCount(); ++dev) {
                    stream.Set(dev, GetState().Devices[dev]->RequestStream());
                }

                for (ui64 dev = 0; dev < stream.DeviceCount(); ++dev) {
                    CB_ENSURE(stream.At(dev) == stream.At(0), "Error: we expect stream identifier to be equal for all devices");
                }
                GetState().FreeStreams.push_back(GetState().Streams.size());
                GetState().Streams.push_back(stream.At(0));
            }

            ui64 id = GetState().FreeStreams.back();
            GetState().FreeStreams.pop_back();
            return TComputationStream(id, this);
        }

        TComputationStream DefaultStream() {
            return TComputationStream(0, this);
        }
    };

    static inline TCudaManager& GetCudaManager() {
        auto& manager = *FastTlsSingleton<TCudaManager>();
        Y_ASSERT(!manager.IsLocked());
        CB_ENSURE(!manager.IsLocked());
        return manager;
    }

    template <class T>
    class TDeviceObjectExtractor<TDistributedObject<T>> {
    public:
        using TRemoteObject = T;

        static TRemoteObject At(int devId, const TDistributedObject<T>& object) {
            return object.At(devId);
        }
    };

    class TChildCudaManagerInitializer: public TNonCopyable {
    public:
        TChildCudaManagerInitializer()
            : Parent(GetCudaManager())
        {
            IsRequested.resize(Parent.GetDeviceCount(), true);
            for (auto dev : Parent.GetActiveDevices()) {
                IsRequested[dev] = false;
            }
            Parent.Locked = true;
        }

        ~TChildCudaManagerInitializer() {
            for (auto& event : Events) {
                event.WaitI();
            }
            Parent.Locked = false;
        }

        struct TStopChildLock {
        public:
            void Acquire() {
            }

            void Release() {
                auto& childManager = GetCudaManager();
                childManager.StopChild();
            }
        };

        TGuard<TStopChildLock> Initialize(const TDevicesList& devices) {
            TGuard<TAdaptiveLock> guard(Lock);

            for (auto& dev : devices) {
                CB_ENSURE(IsRequested[dev] == false);
                IsRequested[dev] = true;
            }

            auto& childManager = GetCudaManager();
            CB_ENSURE(&childManager != &Parent);

            Events.push_back(TAutoEvent());
            StopChildGuards.push_back(TStopChildLock());

            childManager.StartChild(Parent, devices, Events.back());

            return Guard(StopChildGuards.back());
        }

        TGuard<TStopChildLock> Initialize(ui32 dev) {
            return Initialize(TDevicesList(1ULL << dev));
        }

        explicit TChildCudaManagerInitializer(TCudaManager& parent)
            : Parent(parent)
        {
        }

    private:
        TAdaptiveLock Lock;
        TCudaManager& Parent;
        TVector<bool> IsRequested;
        TVector<TAutoEvent> Events;
        TVector<TStopChildLock> StopChildGuards;
    };
}

using TComputationStream = NCudaLib::TCudaManager::TComputationStream;

static inline TComputationStream DefaultStream() {
    return NCudaLib::GetCudaManager().DefaultStream();
}

static inline TComputationStream RequestStream() {
    return NCudaLib::GetCudaManager().RequestStream();
}

template <class TKernel, class... Args>
inline void LaunchKernels(NCudaLib::TDevicesList&& devices,
                          ui32 streamId,
                          Args&&... args) {
    auto& manager = NCudaLib::GetCudaManager();
    manager.LaunchKernels<TKernel>(std::forward<NCudaLib::TDevicesList>(devices), streamId, std::forward<Args>(args)...);
}

template <class TKernel>
inline void LaunchKernel(ui32 device,
                         ui32 streamId,
                         TKernel&& kernel) {
    auto& manager = NCudaLib::GetCudaManager();
    manager.LaunchKernel<TKernel>(std::move(kernel), device, streamId);
}

inline void StartCudaManager(const ELoggingLevel loggingLevel = ELoggingLevel::Info) {
    SetLogingLevel(loggingLevel);
    auto& manager = NCudaLib::GetCudaManager();
    manager.Start();
    manager.WaitComplete();

    ui32 devCount = manager.GetDeviceCount();
    for (ui32 dev = 0; dev < devCount; ++dev) {
        MATRIXNET_INFO_LOG << "Free memory for device #" << dev << " " << manager.FreeMemoryMb(dev) << "MB" << Endl;
    }
}

inline ui32 GetDeviceCount() {
    return NCudaLib::GetCudaManager().GetDeviceCount();
}

inline void StopCudaManager() {
    auto& manager = NCudaLib::GetCudaManager();
    manager.Stop();
}
