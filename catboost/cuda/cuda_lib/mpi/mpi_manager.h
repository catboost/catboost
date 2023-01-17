#pragma once

#if defined(USE_MPI)

#include <mpi.h>
#include <catboost/cuda/cuda_lib/cuda_base.h>
#include <catboost/cuda/cuda_lib/device_id.h>
#include <catboost/cuda/cuda_lib/serialization/task_factory.h>
#include <catboost/cuda/utils/spin_wait.h>
#include <library/cpp/blockcodecs/codecs.h>
#include <library/cpp/threading/chunk_queue/queue.h>
#include <util/thread/singleton.h>
#include <util/system/types.h>
#include <util/stream/buffer.h>
#include <util/ysaveload.h>
#include <util/system/yield.h>
#include <util/system/mutex.h>
#include <util/stream/file.h>
#include <util/system/event.h>
#include <util/generic/queue.h>
#include <thread>

namespace NCudaLib {
#define MPI_SAFE_CALL(cmd)                                                   \
    {                                                                        \
        int mpiErrNo = (cmd);                                                \
        if (MPI_SUCCESS != mpiErrNo) {                                       \
            char msg[MPI_MAX_ERROR_STRING];                                  \
            int len;                                                         \
            MPI_Error_string(mpiErrNo, msg, &len);                           \
            CATBOOST_ERROR_LOG << "MPI failed with error code :" << mpiErrNo \
                               << " " << msg << Endl;                        \
            MPI_Abort(MPI_COMM_WORLD, mpiErrNo);                             \
        }                                                                    \
    }

    /*
     * This  manager is designed to work correctly only for computation model used in cuda_lib routines
     * It's not general-use class (at least currently) and should not be used anywhere outside cuda_lib
     */
    class TMpiManager {
    public:
        class TMpiRequest: public TNonCopyable, public TThrRefBase {
        public:
            bool IsComplete() const {
                return GetState() == EState::Completed;
            }

            void WaitComplete() const {
                Wait(TDuration::Max());
            }

            ui64 ReceivedBytes() const {
                CB_ENSURE(IsComplete(), "Request is not completed");
                return ReceivedBytesCount;
            }

            void Wait(const TDuration& interval) const {
                if (!IsComplete()) {
                    WaitEvent.WaitT(interval);
                }
                CB_ENSURE(IsComplete(), "Error: event is not complete");
            }

            void Abort() {
                AtomicSet(CancelFlag, 1);
            }

        private:
            enum class EState {
                Created,
                Running,
                Completed,
                Canceled
            };

            EState GetState() const {
                int state = State;
                if (state == 0) {
                    return EState::Created;
                } else if (state == 1) {
                    return EState::Running;
                } else if (state == 2) {
                    return EState::Completed;
                } else {
                    CB_ENSURE(state == 3, "Unexpected request state");
                    return EState::Canceled;
                }
            }

            void SetState(EState state) {
                if (state == EState::Created) {
                    AtomicSet(State, 0);
                } else if (state == EState::Running) {
                    AtomicSet(State, 1);
                } else if (state == EState::Completed) {
                    AtomicSet(State, 2);
                } else {
                    CB_ENSURE(state == EState::Canceled, "Unexpected request state");
                    AtomicSet(State, 3);
                }
            }

        private:
            TMpiRequest() {
            }

            friend TMpiManager;

        private:
            mutable TManualEvent WaitEvent;
            mutable TAtomic CancelFlag = 0;
            mutable TAtomic State = 0;
            mutable MPI_Status Status;
            mutable MPI_Request Request;
            mutable TAtomic ReceivedBytesCount;
        };

        using TMpiRequestPtr = TIntrusivePtr<TMpiRequest>;

        void Start(int* argc, char*** argv);

        void Stop();

        bool IsMaster() const {
            return HostId == 0;
        }

        TMpiRequestPtr ReadAsync(char* data, int dataSize, int sourceRank, int tag);

        void ReadAsync(char* data, ui64 dataSize,
                       int sourceRank, int tag,
                       TVector<TMpiRequestPtr>* requests) {
            Y_ASSERT(dataSize);
            const ui64 blockSize = (int)Min<ui64>(dataSize, 1 << 30);
            ReadAsync(data, dataSize, blockSize, sourceRank, tag, requests);
        }

        //could read 2GB+ data
        void ReadAsync(char* data, ui64 dataSize, ui64 blockSize,
                       int sourceRank, int tag,
                       TVector<TMpiRequestPtr>* requests) {
            Y_ASSERT(dataSize);
            for (ui64 offset = 0; offset < dataSize; offset += blockSize) {
                const auto size = static_cast<const int>(Min<ui64>(blockSize, dataSize - offset));
                requests->push_back(ReadAsync(data + offset, size, sourceRank, tag));
            }
        }

        void WriteAsync(const char* data, ui64 dataSize, int destRank, int tag, TVector<TMpiRequestPtr>* requests) {
            Y_ASSERT(dataSize);
            const ui64 blockSize = (int)Min<ui64>(dataSize, 1 << 30);
            WriteAsync(data, dataSize, blockSize, destRank, tag, requests);
        }

        //could read 2GB+ data
        void WriteAsync(const char* data, ui64 dataSize, ui64 blockSize,
                        int destRank, int tag, TVector<TMpiRequestPtr>* requests) {
            Y_ASSERT(dataSize);
            for (ui64 offset = 0; offset < dataSize; offset += blockSize) {
                const auto size = static_cast<const int>(Min<ui64>(blockSize, dataSize - offset));
                requests->push_back(WriteAsync(data + offset, size, destRank, tag));
            }
        }

        int GetTaskTag(const TDeviceId& deviceId) {
            Y_ASSERT(deviceId.DeviceId >= 0);
            return deviceId.DeviceId + 1;
        }

        void SendTask(const TDeviceId& deviceId,
                      TSerializedTask&& task);

        template <class T>
        TMpiRequestPtr ReceivePodAsync(int rank, int tag, T* dst) {
            CB_ENSURE(std::is_pod<T>::value, "Not a pod type");
            return ReadAsync(reinterpret_cast<char*>(dst), sizeof(T), rank, tag);
        }

        TMpiRequestPtr ReceiveBufferAsync(int rank, int tag, TBuffer* dst) {
            return ReadAsync(dst->Data(), static_cast<int>(dst->Size()), rank, tag);
        }

        void Read(char* data, int dataSize, int sourceRank, int tag) {
            ReadAsync(data, dataSize, sourceRank, tag)->WaitComplete();
        }

        TMpiRequestPtr WriteAsync(const char* data, int dataSize, int destRank, int tag);

        void Write(const char* data, int dataSize, int destRank, int tag) {
            WriteAsync(data, dataSize, destRank, tag)->WaitComplete();
        }

        template <class T>
        void Send(const T& value, int rank, int tag) {
            if (std::is_pod<T>::value) {
                return SendPod(value, rank, tag);
            } else {
                TBuffer buffer;
                {
                    TBufferOutput out(buffer);
                    ::Save(&out, value);
                }
                Write(buffer.Data(), static_cast<int>(buffer.Size()), rank, tag);
            }
        }

        template <class T>
        void SendPod(const T& value, int rank, int tag) {
            CB_ENSURE(std::is_pod<T>::value, "Not a pod type");
            Write(reinterpret_cast<const char*>(&value), sizeof(value), rank, tag);
        }

        int GetHostId() const {
            return HostId;
        }

        static constexpr int GetMasterId() {
            return 0;
        }

        int NextCommunicationTag() {
            Y_ASSERT(IsMaster());
            const int cycleLen = (1 << 16) - 1;
            int tag = static_cast<int>(AtomicIncrement(Counter));
            tag = tag < 0 ? -tag : tag;
            tag %= cycleLen;
            tag = (tag << 10) | 1023;
            //MPI tags should be positive
            return tag;
        }

        const TVector<TDeviceId>& GetDevices() const {
            Y_ASSERT(IsMaster());
            return Devices;
        }

        const TVector<NCudaLib::TCudaDeviceProperties>& GetDeviceProperties() const {
            Y_ASSERT(IsMaster());
            return DeviceProps;
        }

        ui64 GetMinCompressSize() const {
            return MinCompressSize;
        }

        const NBlockCodecs::ICodec* GetCodec() const {
            return CompressCodec;
        }

    private:
        //Every MPI operations are done via proxy thread.
        struct TMemcpySendRequest {
            TMpiRequestPtr Request;
            const char* Data;
            int DataSize;
            int DestRank;
            int Tag;
        };

        struct TMemcpyReceiveRequest {
            TMpiRequestPtr Request;
            char* Data;
            int DataSize;
            int SourceRank;
            int Tag;
        };

        struct TSendTaskRequest {
            TDeviceId DeviceId;
            TSerializedTask Task;
        };

        void ProceedRequests();
        TMpiRequest::EState InvokeRunningRequest(TMpiRequest* request);

    private:
        MPI_Comm Communicator;
        int HostCount;
        int HostId;

        TVector<NCudaLib::TDeviceId> Devices;
        TVector<NCudaLib::TCudaDeviceProperties> DeviceProps;

        TAtomic Counter = 0;
        bool UseBSendForTasks = false;
        static const ui64 BufferSize = 32 * 1024 * 1024; //32MB should be enough for simple kernels

        const NBlockCodecs::ICodec* CompressCodec = nullptr;
        ui64 MinCompressSize = 10000;

        TVector<char> CommandsBuffer;

        TManualEvent HasWorkEvent;
        TAtomic StopFlag = 0;

        THolder<std::thread> MpiProxyThread;

        NThreading::TManyOneQueue<TSendTaskRequest> SendCommands;
        NThreading::TManyOneQueue<TMemcpyReceiveRequest> ReceiveRequests;
        NThreading::TManyOneQueue<TMemcpySendRequest> SendRequests;

        TVector<TMpiRequestPtr> RunningRequests;
    };

    static inline TMpiManager& GetMpiManager() {
        auto& manager = *Singleton<TMpiManager>();
        return manager;
    }

    using TMpiRequestPtr = TMpiManager::TMpiRequestPtr;
}

#endif

namespace NCudaLib {
    inline int GetHostId() {
#if defined(USE_MPI)
        return GetMpiManager().GetHostId();
#else
        return 0;
#endif
    }

#if defined(USE_MPI)
    inline bool AreRequestsComplete(const TVector<TMpiRequestPtr>& MpiRequests) {
        for (const auto& request : MpiRequests) {
            if (!request->IsComplete()) {
                return false;
            }
        }
        return true;
    }
#endif

}
