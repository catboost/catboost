#pragma once

#include "cuda_base.h"
#include "devices_list.h"
#include "memory_provider_trait.h"
#include "remote_objects.h"
#include "single_device.h"
#include "devices_provider.h"

#include <util/generic/vector.h>

/*
 * Master-slave control and mpi routines
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
            static inline bool IsEmpty(const T& val) {
                return val == 0;
            }
        };
    }

    template <class T>
    class TDistributedObject {
    public:
        TDistributedObject(TDistributedObject&& other) = default;
        TDistributedObject(const TDistributedObject& other) = default;

        TDistributedObject& operator=(TDistributedObject&& other) = default;
        TDistributedObject& operator=(const TDistributedObject& other) = default;

        T At(ui32 devId) const {
            return Data[devId];
        }

        void Set(ui32 devId, T t) {
            Data[devId] = t;
        }

        bool IsEmpty(ui32 devId) const {
            return NHelpers::TEmptyObjectsHelper<T>::IsEmpty(At(devId));
        }

        ui64 DeviceCount() const {
            return Data.size();
        }

        TDistributedObject& operator+=(const TDistributedObject& other) {
            for (ui32 dev = 0; dev < Data.size(); ++dev) {
                Data[dev] += other.Data[dev];
            }
            return *this;
        }

        void Fill(const T& value) {
            std::fill(Data.begin(), Data.end(), value);
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
            TMap<TCudaSingleDevice*, ui32> DevPtrToDevId;

            bool PeersSupportEnabled = false;
            TAdaptiveLock Lock;

            void BuildDevPtrToDevId() {
                DevPtrToDevId.clear();
                for (ui32 dev = 0; dev < Devices.size(); ++dev) {
                    DevPtrToDevId[Devices[dev]] = dev;
                }
            }
        };

        TVector<TDistributedObject<ui32>> Streams;
        TVector<ui32> FreeStreams;

        TAtomicSharedPtr<TCudaManagerState> State;

        TCudaProfiler* Profiler;
        TCudaProfiler* ParentProfiler = nullptr;

        bool IsChildManager = false;
        TDevicesList DevicesList;
        TManualEvent OnStopChildEvent;

        TVector<bool> IsActiveDevice;
        bool Locked = false;

        void FreeStream(const ui32 streamId) {
            TDistributedObject<ui32>& stream = Streams[streamId];

            for (ui64 dev = 0; dev < State->Devices.size(); ++dev) {
                const auto devStreamId = stream.At(dev);
                if (devStreamId != State->Devices[dev]->DefaultStream()) {
                    State->Devices[dev]->FreeStream(devStreamId);
                } else {
                    CB_ENSURE(!IsActiveDevice[dev]);
                }
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

        inline void InitDefaultStream() {
            CB_ENSURE(Streams.size() == 0);

            ui32 defaultStream = 0;
            for (auto& dev : DevicesList) {
                CB_ENSURE(defaultStream == State->Devices[dev]->DefaultStream());
            }
            {
                TDistributedObject<ui32> stream(GetDeviceCount());
                stream.Fill(defaultStream);
                Streams.push_back(std::move(stream));
            }
        }

        void SetDevices(TVector<TCudaSingleDevice*>&& devices) {
            CB_ENSURE(!HasDevices(), "Error: CudaManager already has devices");
            GetState().Devices = std::move(devices);
            CB_ENSURE(Streams.size() == 0);
            CB_ENSURE(FreeStreams.size() == 0);
            const auto deviceCount = GetState().Devices.size();
            DevicesList = TDevicesListBuilder::Range(0, deviceCount);
            IsActiveDevice.clear();
            IsActiveDevice.resize(GetDeviceCount(), true);
            State->BuildDevPtrToDevId();
            InitDefaultStream();
        }

        void FreeDevices() {
            auto& provider = GetDevicesProvider();
            for (auto dev : GetState().Devices) {
                provider.Free(dev);
            }
            GetState().Devices.resize(0);
            GetState().DevPtrToDevId.clear();
        }

        void CreateProfiler();
        void ResetProfiler(bool printInfo);

        template <typename TDistributedObject>
        inline static auto GetDeviceObject(ui64 devId,
                                           TDistributedObject& object) -> decltype(TDeviceObjectExtractor<std::remove_reference_t<TDistributedObject>>::At(devId, object)) {
            return TDeviceObjectExtractor<std::remove_reference_t<TDistributedObject>>::At(devId, object);
        }

        template <class TKernel, class... Args>
        inline static TKernel GetDeviceKernel(ui64 devId,
                                              Args&... args) {
            return {GetDeviceObject<Args>(devId, args)...};
        };

        TCudaSingleDevice* GetDevice(ui32 devId) {
            CB_ENSURE(IsActiveDevice[devId]);
            return GetState().Devices[devId];
        }

        ui32 StreamAt(ui32 streamId, TCudaSingleDevice* singleDev) const {
            return StreamAt(streamId, GetState().DevPtrToDevId.at(singleDev));
        }

        inline void FreeComputationStreams() {
            CB_ENSURE((1 + FreeStreams.size()) == Streams.size(), "Error: not all streams are free");
            for (int i = Streams.size() - 1; i > 0; --i) {
                FreeStream(i);
            }
            Streams.clear();
            FreeStreams.resize(0);
        }

        friend class TDataCopier;
        friend class TChildCudaManagerInitializer;
        friend class TStreamSectionTaskLauncher;

    private:
        void EnablePeers();
        void DisablePeers();

    public:
        ~TCudaManager();

        template <class TKernel>
        inline void LaunchKernel(TKernel&& kernel,
                                 ui64 dev,
                                 ui32 stream) const {
            Y_ASSERT(dev < GetState().Devices.size());
            CB_ENSURE(IsActiveDevice[dev]);
            const ui32 devStreamId = StreamAt(stream, dev);
            GetState().Devices[dev]->LaunchKernel(std::move(kernel), devStreamId);
        }

        template <class TKernel,
                  class... Args>
        inline void LaunchKernels(TDevicesList&& devices,
                                  ui32 streamId,
                                  Args&&... args) {
            for (ui64 devId : devices) {
                auto kernel = GetDeviceKernel<TKernel, Args...>(devId, args...);
                LaunchKernel<TKernel>(std::move(kernel), devId, streamId);
            }
        }

        ui64 GetDeviceCount() const {
            return GetState().Devices.size();
        }

        TVector<ui32> GetDevices(bool onlyLocalIfHasAny = true) const {
            TVector<ui32> devices;
            for (auto& dev : DevicesList) {
                if (onlyLocalIfHasAny && GetState().Devices[dev]->IsRemoteDevice()) {
                    continue;
                }
                devices.push_back(dev);
            }
            if (devices.size() == 0) {
                for (auto& dev : DevicesList) {
                    devices.push_back(dev);
                }
            }
            return devices;
        }

        TDevicesList GetActiveDevices() const {
            return DevicesList;
        }

        double FreeMemoryMb(ui32 devId, bool waitComplete = true) const;

        double TotalMemoryMb(ui32 devId) const;

        //waits for finish all work submitted to selected devices
        void WaitComplete(TDevicesList&& devices) {
            using TEventPtr = THolder<IDeviceFuture<ui64>>;
            TVector<TEventPtr> waitComplete;

            for (auto dev : devices) {
                CB_ENSURE(dev < GetState().Devices.size());
                CB_ENSURE(IsActiveDevice[dev], "Device should be active");
                waitComplete.push_back(GetState().Devices[dev]->WaitComplete());
            }

            for (auto& event : waitComplete) {
                event->Wait();
                Y_VERIFY(event->Has());
            }
        }

        bool IsLocked() const {
            return Locked;
        }

        void WaitComplete() {
            WaitComplete(GetActiveDevices());
        }

        inline void Start(const NCudaLib::TDeviceRequestConfig& config) {
            CB_ENSURE(State == nullptr);
            State.Reset(new TCudaManagerState());
            CB_ENSURE(!HasDevices());
            SetDevices(GetDevicesProvider().RequestDevices(config));
            if (config.EnablePeers) {
                EnablePeers();
                State->PeersSupportEnabled = true;
            }
            CreateProfiler();
        }

        inline void Stop() {
            CB_ENSURE(!IsChildManager);
            CB_ENSURE(State);

            if (State->PeersSupportEnabled) {
                DisablePeers();
            }

            FreeComputationStreams();
            WaitComplete();
            FreeDevices();
            ResetProfiler(true);
            State = nullptr;
        }

        void StopChild();

        void StartChild(TCudaManager& parent,
                        const TDevicesList& devices,
                        TManualEvent& stopEvent);

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

        template <class T>
        TDistributedObject<T> CreateDistributedObject(T defaultValue) {
            auto object = TDistributedObject<T>(GetDeviceCount());
            object.Fill(defaultValue);
            return object;
        }

        template <class T, EPtrType Type>
        TCudaSingleDevice::TSingleBuffer<T, Type> CreateSingleBuffer(ui64 devId, ui64 size) {
            return GetState().Devices[devId]->CreateSingleBuffer<T, Type>(size);
        };

        class TComputationStream: public TMoveOnly {
        private:
            TCudaManager* Owner;
            ui32 Id;

            TComputationStream(ui32 id,
                               TCudaManager* owner)
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
                if (Id) {
                    Owner->FreeStreams.push_back(Id);
                }
            }

            //sync on remote. ensures that job done in this stream will be seen for all other jobs submitted to other stream.
            //device-scope, don't guarantee any sync between devices
            void Synchronize(TDevicesList&& devices) {
                //                auto& stream = Owner->GetState().Streams[Id];
                for (auto dev : devices) {
                    CB_ENSURE(Owner->IsActiveDevice[dev]);
                    Owner->GetState().Devices[dev]->StreamSynchronize(At(dev));
                }
            }

            ui32 GetId() const {
                return Id;
            }

            operator ui32() const {
                return Id;
            }

            ui32 At(ui32 dev) const {
                return Owner->StreamAt(Id, dev);
            }

            void Synchronize() {
                Synchronize(Owner->GetActiveDevices());
            }
        };

        ui32 StreamAt(ui32 streamId, ui32 dev) const {
            CB_ENSURE(IsActiveDevice[dev]);
            return Streams[streamId].At(dev);
        }

        TComputationStream RequestStream() {
            if (FreeStreams.size() == 0) {
                TDistributedObject<ui32> stream = CreateDistributedObject<ui32>();
                for (ui64 dev = 0; dev < stream.DeviceCount(); ++dev) {
                    if (IsActiveDevice[dev]) {
                        stream.Set(dev, GetState().Devices[dev]->RequestStream());
                    } else {
                        stream.Set(dev, 0);
                    }
                }
                FreeStreams.push_back(Streams.size());
                Streams.push_back(stream);
            }

            ui32 id = FreeStreams.back();
            FreeStreams.pop_back();
            return TComputationStream(id, this);
        }

        TDeviceId GetDeviceId(ui32 dev) const {
            return GetState().Devices[dev]->GetDevice();
        }

        TComputationStream DefaultStream() {
            return TComputationStream(0, this);
        }
    };

    inline TCudaManager& GetCudaManager() {
        auto& manager = *FastTlsSingleton<TCudaManager>();
        Y_ASSERT(!manager.IsLocked());
        CB_ENSURE(!manager.IsLocked());
        return manager;
    }

    template <class T>
    class TDeviceObjectExtractor<TDistributedObject<T>> {
    public:
        using TRemoteObject = T;

        static TRemoteObject At(ui32 devId, const TDistributedObject<T>& object) {
            return object.At(devId);
        }
    };

    template <class T>
    class TDeviceObjectExtractor<const TDistributedObject<T>> {
    public:
        using TRemoteObject = T;

        static TRemoteObject At(ui32 devId, const TDistributedObject<T>& object) {
            return object.At(devId);
        }
    };

    struct TStopChildCudaManagerCallback {
        TCudaManager* Owner = nullptr;

        TStopChildCudaManagerCallback(TCudaManager* owner)
            : Owner(owner)
        {
        }

        TStopChildCudaManagerCallback(TStopChildCudaManagerCallback&&) = default;

        void operator()() {
            auto& manager = NCudaLib::GetCudaManager();
            CB_ENSURE(&manager == Owner);
            manager.StopChild();
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

        TFinallyGuard<TStopChildCudaManagerCallback> Initialize(const TDevicesList& devices) {
            TGuard<TAdaptiveLock> guard(Lock);

            for (auto& dev : devices) {
                CB_ENSURE(!IsRequested[dev]);
                IsRequested[dev] = true;
            }

            auto& childManager = GetCudaManager();
            CB_ENSURE(&childManager != &Parent);

            Events.push_back(TManualEvent());
            Events.back().Reset();
            childManager.StartChild(Parent, devices, Events.back());
            return TFinallyGuard<TStopChildCudaManagerCallback>(TStopChildCudaManagerCallback(&childManager));
        }

        TFinallyGuard<TStopChildCudaManagerCallback> Initialize(ui64 dev) {
            return Initialize(TDevicesListBuilder::SingleDevice(dev));
        }

        explicit TChildCudaManagerInitializer(TCudaManager& parent)
            : Parent(parent)
        {
        }

    private:
        TAdaptiveLock Lock;
        TCudaManager& Parent;
        TVector<bool> IsRequested;
        TVector<TManualEvent> Events;
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
    manager.LaunchKernel<TKernel>(std::forward(kernel), device, streamId);
}

struct TStopCudaManagerCallback {
    void operator()() {
        auto& manager = NCudaLib::GetCudaManager();
        manager.Stop();
    }
};

TFinallyGuard<TStopCudaManagerCallback> StartCudaManager(const NCudaLib::TDeviceRequestConfig& requestConfig,
                                                         const ELoggingLevel loggingLevel = ELoggingLevel::Debug);

TFinallyGuard<TStopCudaManagerCallback> StartCudaManager(const ELoggingLevel loggingLevel = ELoggingLevel::Debug);

void RunSlave();

inline ui64 GetDeviceCount() {
    return NCudaLib::GetCudaManager().GetDeviceCount();
}

template <class T>
inline NCudaLib::TDistributedObject<T> CreateDistributedObject() {
    auto object = NCudaLib::GetCudaManager().CreateDistributedObject<T>();
    return object;
}

template <class T>
inline NCudaLib::TDistributedObject<T> CreateDistributedObject(T defaultValue) {
    auto object = NCudaLib::GetCudaManager().CreateDistributedObject<T>(defaultValue);
    return object;
}
