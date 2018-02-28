#pragma once

#if defined(USE_MPI)

#include <mpi.h>
#include <catboost/cuda/cuda_lib/cuda_base.h>
#include <catboost/cuda/cuda_lib/device_id.h>
#include <catboost/cuda/cuda_lib/serialization/task_factory.h>
#include <catboost/cuda/utils/spin_wait.h>
#include <util/thread/singleton.h>
#include <util/system/types.h>
#include <util/stream/buffer.h>
#include <util/ysaveload.h>
#include <util/system/yield.h>
#include <util/system/mutex.h>
#include <library/blockcodecs/codecs.h>
#include <util/stream/file.h>

namespace NCudaLib {
    //TODO(noxoomo): check different locks. By design MPI should be able to work with multiple threads, but
    //existing implementation could work bad if MPI is accessed from multiple threads
    using TMpiLock = TMutex;
    TMpiLock& GetMpiLock();

#define MPI_SAFE_CALL(cmd)                                                    \
    with_lock (GetMpiLock()) {                                                \
        int mpiErrNo = (cmd);                                                 \
        if (MPI_SUCCESS != mpiErrNo) {                                        \
            char msg[MPI_MAX_ERROR_STRING];                                   \
            int len;                                                          \
            MPI_Error_string(mpiErrNo, msg, &len);                            \
            MATRIXNET_ERROR_LOG << "MPI failed with error code :" << mpiErrNo \
                                << " " << msg << Endl;                        \
            MPI_Abort(MPI_COMM_WORLD, mpiErrNo);                              \
        }                                                                     \
    }

#if defined(WRITE_MPI_MESSAGE_LOG)
    class TOperationsLogger {
    public:
        TOperationsLogger();

        void Write(TString message) {
            (*Out) << TInstant::Now().SecondsFloat() << ":\t" << message << Endl;
        }

    private:
        THolder<TOFStream> Out;
    };

    inline TOperationsLogger& Logger() {
        return *Singleton<TOperationsLogger>();
    }
#endif

    /*
     * This  manager is designed to work correctly only for computation model used in cuda_lib routines
     * It's not general-use class (at least currently) and should not be used anywhere outside cuda_lib
     */
    class TMpiManager {
    public:
        class TMpiRequest: public TMoveOnly {
        public:
            bool IsComplete() const {
                Y_VERIFY(Flag != -1);
                if (!Flag) {
                    CB_ENSURE(*Request != MPI_REQUEST_NULL);
                    MPI_SAFE_CALL(MPI_Test(Request.Get(), &Flag, &Status));
#if defined(WRITE_MPI_MESSAGE_LOGS)
                    if (Flag) {
                        Logger().Write(Msg);
                    } else if ((Now() - StartTime).Seconds() > TDuration::Seconds(20).Seconds()) {
                        Logger().Write("Can't finish " + Msg);
                        StartTime = TInstant::Max();
                    }
#endif
                }
                return static_cast<bool>(Flag);
            }

            void WaitComplete() const {
                CB_ENSURE(Flag != -1);
                if (!Flag) {
                    Wait(TDuration::Max());
                    Flag = true;
                }
            }

            ui64 ReceivedBytes() const {
                CB_ENSURE(Flag != -1 && Flag != 0, "Wait for request first");
                int result;
                MPI_SAFE_CALL(MPI_Get_count(&Status, MPI_CHAR, &result));
                return static_cast<ui64>(result);
            }

            void Wait(const TDuration& interval) const {
                TSpinWaitHelper::Wait(interval, [&]() -> bool {
                    return IsComplete();
                });
            }

            void Abort() {
                Y_VERIFY(Flag != -1);
                if (!Flag) {
                    MPI_SAFE_CALL(MPI_Cancel(Request.Get()));
                    Flag = -1;
                }
            }

            bool IsCreated() {
                return Flag != -1;
            }

            TMpiRequest(TMpiRequest&& other) {
                if (this != &other) {
                    this->Flag = other.Flag;
                    this->Request.Swap(other.Request);
                    this->Status = other.Status;
#if defined(WRITE_MPI_MESSAGE_LOG)
                    this->Msg = other.Msg;
#endif
                    other.Clear();
                }
            }

            TMpiRequest& operator=(TMpiRequest&& other) {
                if (this != &other) {
                    this->Flag = other.Flag;
                    this->Request.Swap(other.Request);
                    this->Status = other.Status;
#if defined(WRITE_MPI_MESSAGE_LOG)
                    this->Msg = other.Msg;
#endif
                    other.Clear();
                }
                return *this;
            }

            TMpiRequest() {
            }

            ~TMpiRequest() {
                if (Flag != -1) {
                    Y_VERIFY(Flag, "Error: unfinished request");
                }
            }

        private:
#if defined(WRITE_MPI_MESSAGE_LOG)
            TMpiRequest(THolder<MPI_Request>&& request, TString msg)
                : Flag(0)
                , Request(std::move(request))
                , Msg(msg)
            {
                IsComplete();
            }
#endif
            TMpiRequest(THolder<MPI_Request>&& request)
                : Flag(0)
                , Request(std::move(request)) {
                IsComplete();
            }

            void Clear() {
                Flag = -1;
                Request = nullptr;
            }

            friend TMpiManager;

        private:
            mutable int Flag = -1;
            THolder<MPI_Request> Request;
            mutable MPI_Status Status;
#if defined(WRITE_MPI_MESSAGE_LOG)
            mutable TInstant StartTime = Now();
            TString Msg;
#endif
        };

        void Start(int* argc, char*** argv);

        void Stop();

        bool IsMaster() const {
            return HostId == 0;
        }

        TMpiRequest ReadAsync(char* data, int dataSize, int sourceRank, int tag) {
            THolder<MPI_Request> request = new MPI_Request;
#if defined(WRITE_MPI_MESSAGE_LOG)
            Logger().Write(TStringBuilder() << "Before IRecv with tag " << tag << " from " << sourceRank << " size " << dataSize);
#endif
            MPI_SAFE_CALL(MPI_Irecv(data, dataSize, MPI_CHAR, sourceRank, tag, Communicator, request.Get()));
#if defined(WRITE_MPI_MESSAGE_LOG)
            Logger().Write(TStringBuilder() << "After IRecv with tag " << tag << " from " << sourceRank << " size " << dataSize);
#endif
            return {
                std::move(request)
#if defined(WRITE_MPI_MESSAGE_LOG)
                    ,
                    TStringBuilder() << "Finished IRecv with tag " << tag << " from " << sourceRank << " size " << dataSize
#endif
            };
        }

        void Read(char* data, int dataSize, int sourceRank, int tag) {
            MPI_SAFE_CALL(MPI_Recv(data, dataSize, MPI_CHAR, sourceRank, tag, Communicator, MPI_STATUS_IGNORE));
        }

        TMpiRequest WriteAsync(const char* data, int dataSize, int destRank, int tag) {
            THolder<MPI_Request> request = new MPI_Request;
#if defined(WRITE_MPI_MESSAGE_LOG)
            Logger().Write(TStringBuilder() << "Before ISend with tag " << tag << " to " << destRank << " size " << dataSize);
#endif
            //issend should be better here for our type of usage
            MPI_SAFE_CALL(MPI_Issend(data, dataSize, MPI_CHAR, destRank, tag, Communicator, request.Get()));
#if defined(WRITE_MPI_MESSAGE_LOG)
            Logger().Write(TStringBuilder() << "After ISend with tag " << tag << " to " << destRank << " size " << dataSize);
#endif
            return {
                std::move(request)
#if defined(WRITE_MPI_MESSAGE_LOG)
                    ,
                    TStringBuilder() << "Finished ISend with tag " << tag << " to " << destRank << " size " << dataSize
#endif
            };
        }

        void Write(const char* data, int dataSize, int destRank, int tag) {
            MPI_SAFE_CALL(MPI_Send(data, dataSize, MPI_CHAR, destRank, tag, Communicator));
        }

        void ReadAsync(char* data, ui64 dataSize,
                       int sourceRank, int tag,
                       TVector<TMpiRequest>* requests) {
            Y_ASSERT(dataSize);
            const ui64 blockSize = (int)Min<ui64>(dataSize, 1 << 30);
            ReadAsync(data, dataSize, blockSize, sourceRank, tag, requests);
        }

        //could read 2GB+ data
        void ReadAsync(char* data, ui64 dataSize, ui64 blockSize,
                       int sourceRank, int tag,
                       TVector<TMpiRequest>* requests) {
            Y_ASSERT(dataSize);
            for (ui64 offset = 0; offset < dataSize; offset += blockSize) {
                const auto size = static_cast<const int>(Min<ui64>(blockSize, dataSize - offset));
                requests->push_back(ReadAsync(data + offset, size, sourceRank, tag));
            }
        }

        void WriteAsync(const char* data, ui64 dataSize, int destRank, int tag, TVector<TMpiRequest>* requests) {
            Y_ASSERT(dataSize);
            const ui64 blockSize = (int)Min<ui64>(dataSize, 1 << 30);
            WriteAsync(data, dataSize, blockSize, destRank, tag, requests);
        }

        //could read 2GB+ data
        void WriteAsync(const char* data, ui64 dataSize, ui64 blockSize,
                        int destRank, int tag, TVector<TMpiRequest>* requests) {
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

        void SendTask(const TSerializedTask& task,
                      const TDeviceId& deviceId) {
            Y_ASSERT(IsMaster());
            const int size = static_cast<const int>(task.Size());
            Y_ASSERT(size < (int)BufferSize);
            Y_ASSERT(size);
#if defined(WRITE_MPI_MESSAGE_LOG)
            Logger().Write(TStringBuilder() << "Before SendTask with tag " << GetTaskTag(deviceId) << " to " << deviceId.HostId << " size " << size);
#endif
            if (UseBSendForTasks) {
                MPI_SAFE_CALL(MPI_Bsend(task.Data(), size, MPI_CHAR,
                                        deviceId.HostId, GetTaskTag(deviceId), Communicator));
            } else {
                MPI_SAFE_CALL(MPI_Send(task.Data(), size, MPI_CHAR,
                                       deviceId.HostId, GetTaskTag(deviceId), Communicator));
            }
#if defined(WRITE_MPI_MESSAGE_LOG)
            Logger().Write(TStringBuilder() << "After SendTask with tag " << GetTaskTag(deviceId) << " to " << deviceId.HostId << " size " << size);
#endif
        }

        void Wait(int rank, int tag, const TDuration& interval) {
            TSpinWaitHelper::Wait(interval, [&]() -> bool {
                return HasMessage(rank, tag);
            });
        }

        TBuffer DynamicReceive(int rank, int tag) {
            MPI_Status status;
            MPI_SAFE_CALL(MPI_Probe(rank, tag, Communicator, &status));
            int size = 0;
            MPI_SAFE_CALL(MPI_Get_count(&status, MPI_CHAR, &size));
            TBuffer data;
            data.Resize(static_cast<size_t>(size));
            MPI_SAFE_CALL(MPI_Recv(data.Data(), size, MPI_CHAR, rank, tag, Communicator, &status));
            return data;
        }

        template <class T>
        TMpiRequest ReceivePodAsync(int rank, int tag, T* dst) {
            CB_ENSURE(std::is_pod<T>::value, "Not a pod type");
            return ReadAsync(reinterpret_cast<char*>(dst), sizeof(T), rank, tag);
        }

        TMpiRequest ReceiveBufferAsync(int rank, int tag, TBuffer* dst) {
            return ReadAsync(dst->Data(), static_cast<int>(dst->Size()), rank, tag);
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
        T Receive(int rank, int tag) {
            T result;
            if (std::is_pod<T>::value) {
                ReceivePodAsync<T>(rank, tag, &result).WaitComplete();
            } else {
                TBuffer buffer = DynamicReceive(rank, tag);
                TBufferInput input(buffer);
                ::Load(&input, result);
            }
            return result;
        }

        template <class T>
        void SendPod(const T& value, int rank, int tag) {
            CB_ENSURE(std::is_pod<T>::value, "Not a pod type");
            Write(reinterpret_cast<const char*>(&value), sizeof(value), rank, tag);
        }

        bool HasMessage(int rank, int tag) {
            int flag = false;
            MPI_SAFE_CALL(MPI_Iprobe(rank, tag, Communicator, &flag, MPI_STATUS_IGNORE));
            return static_cast<bool>(flag);
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

        TMpiLock& GetLock() {
            return Lock;
        }

    private:
        TMpiLock Lock;
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
    };

    static inline TMpiManager& GetMpiManager() {
        auto& manager = *Singleton<TMpiManager>();
        return manager;
    }

    using TMpiRequest = TMpiManager::TMpiRequest;
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
    inline bool AreRequestsComplete(const TVector<TMpiRequest>& MpiRequests) {
        for (const auto& request : MpiRequests) {
            if (!request.IsComplete()) {
                return false;
            }
        }
        return true;
    }
#endif

}
