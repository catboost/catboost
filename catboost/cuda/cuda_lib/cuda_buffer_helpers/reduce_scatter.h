#pragma once

#include <catboost/cuda/cuda_lib/kernel.h>
#include <catboost/cuda/cuda_lib/cuda_kernel_buffer.h>
#include <catboost/cuda/cuda_lib/mapping.h>
#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_lib/kernel/reduce.cuh>
#include <catboost/cuda/cuda_lib/stream_section_tasks_launcher.h>
#include <catboost/cuda/cuda_lib/peer_devices.h>
#include <cmath>
#include <catboost/cuda/cuda_lib/tasks_impl/memory_copy_staged_operation.h>
#include <catboost/cuda/utils/helpers.h>

#include <catboost/libs/helpers/math_utils.h>

#include <util/generic/bitops.h>

namespace NKernelHost {
    template <typename T>
    class TShiftMemoryKernel: public TKernelBase<NKernel::TKernelWithTempBufferContext<T>, false> {
    private:
        TCudaBufferPtr<T> Data;
        TSlice Slice;

    public:
        using TKernelContext = typename NKernel::TKernelWithTempBufferContext<T>;

        TShiftMemoryKernel() = default;

        TShiftMemoryKernel(TCudaBufferPtr<T> data,
                           TSlice slice)
            : Data(data)
            , Slice(slice)
        {
        }

        Y_SAVELOAD_DEFINE(Data, Slice);

        THolder<TKernelContext> PrepareContext(IMemoryManager& memoryManager) const {
            auto context = MakeHolder<TKernelContext>();

            if (Slice.Size() && Slice.Left) {
                context->TempBuffer = memoryManager.Allocate<T>(Data.SliceMemorySize(Slice));
            }
            return context;
        }

        void Run(const TCudaStream& stream,
                 TKernelContext& context) const {
            if (context.TempBuffer != nullptr) {
                NCudaLib::TMemoryCopier<EPtrType::CudaDevice, EPtrType::CudaDevice>::template CopyMemoryAsync<T>(Data.Get() + Data.SliceMemoryOffset(Slice), context.TempBuffer,
                                                                                                                 Data.SliceMemorySize(Slice),
                                                                                                                 stream);

                NCudaLib::TMemoryCopier<EPtrType::CudaDevice, EPtrType::CudaDevice>::template CopyMemoryAsync<T>(context.TempBuffer, Data.Get(),
                                                                                                                 Data.SliceMemorySize(Slice),
                                                                                                                 stream);
            }
        }
    };

}

namespace NCudaLib {
#if defined(USE_MPI)

    template <class T>
    struct TReduceOperator {
        ui64 BlockSize = 0;

        inline void operator()(T* dst, const T* data, ui64 size, const TCudaStream& stream) {
            NKernel::ReduceBinary(dst, dst, data, size, stream.GetStream());
        }

        ui64 GetBlockSize(ui64) {
            return BlockSize;
        }
    };

    template <class T>
    using TRemoteReadAndReduceTask = TThroughHostStagedRecvTask<T, TReduceOperator<T>>;

#endif
    enum class EReduceAlgorithm {
        Ring,
        Tree
    };

    template <class T>
    struct TReduceBinaryContext {
        NCudaLib::THandleRawPtr LocalTempBuffer;
        TVector<bool> IsPeerLocalTask;
        bool AreLocalTaskSend = false;

#if defined(USE_MPI)
        TVector<THolder<IStagedTask>> RunningStagedTasks;
#endif
    };

    template <typename T>
    class TReduceBinaryStreamTask: public NKernelHost::TKernelWithContext<TReduceBinaryContext<T>> {
    private:
        const ui64 LocalBufferSize = 16 * 1024 * 1024 / sizeof(T);

    public:
        using TKernelContext = TReduceBinaryContext<T>;

        struct TLocalHostReduce {
            NKernelHost::TCudaBufferPtr<T> Source;
            NKernelHost::TCudaBufferPtr<T> Dest;

            Y_SAVELOAD_DEFINE(Source, Dest);
        };

#if defined(USE_MPI)
        struct TRemoteHostReduce {
            NKernelHost::TCudaBufferPtr<T> Source;
            NKernelHost::TCudaBufferPtr<T> Dest;
            int Tag = -1;
            bool IsSendTask = false;
            bool Compress = false;

            Y_SAVELOAD_DEFINE(Source, Dest, Tag, IsSendTask, Compress);
        };

#endif

        TReduceBinaryStreamTask() = default;

        TReduceBinaryStreamTask(TReduceBinaryStreamTask&& other) = default;
        TReduceBinaryStreamTask(const TReduceBinaryStreamTask& other) = default;

        TReduceBinaryStreamTask& operator=(TReduceBinaryStreamTask&& other) = default;
        TReduceBinaryStreamTask& operator=(const TReduceBinaryStreamTask& other) = default;

        THolder<TKernelContext> PrepareContext(NKernelHost::IMemoryManager& memoryManager) const {
            auto context = MakeHolder<TKernelContext>();

            for (ui32 localTask = 0; localTask < LocalReduces.size(); ++localTask) {
                auto& src = LocalReduces[localTask].Source;
                auto& dst = LocalReduces[localTask].Dest;
                CB_ENSURE(src.Size() == dst.Size());
                auto sourceDevice = src.GetDeviceId();
                auto destDevice = dst.GetDeviceId();
                CB_ENSURE(sourceDevice.HostId == destDevice.HostId);
                const bool hasPeerAccess = (NCudaLib::GetPeerDevicesHelper().HasPeerAccess(sourceDevice.DeviceId, destDevice.DeviceId));
                context->IsPeerLocalTask.push_back(hasPeerAccess);

                if (!hasPeerAccess && context->LocalTempBuffer.IsNullptr()) {
                    auto ptr = memoryManager.Allocate<T, EPtrType::CudaDevice>(LocalBufferSize);
                    context->LocalTempBuffer = NCudaLib::THandleRawPtr(ptr);
                }
            }

#if defined(USE_MPI)

            TReduceOperator<T> op;
            op.BlockSize = 4 * 1024 * 1024 / sizeof(T);

            for (ui32 remoteTask = 0; remoteTask < RemoteReduces.size(); ++remoteTask) {
                const TRemoteHostReduce& task = RemoteReduces[remoteTask];
                auto& source = task.Source;
                auto& dest = task.Dest;

                if (task.IsSendTask) {
                    const ui64 size = task.Source.Size();
                    context->RunningStagedTasks.push_back(BlockedSendTask(task.Source.Get(),
                                                                          size,
                                                                          op.GetBlockSize(size),
                                                                          dest.GetDeviceId().HostId,
                                                                          task.Tag,
                                                                          memoryManager,
                                                                          task.Compress));
                } else {
                    using TTask = TRemoteReadAndReduceTask<T>;
                    const ui64 size = task.Source.Size();
                    context->RunningStagedTasks.push_back(MakeHolder<TTask>(task.Dest.Get(), size,
                                                                            op,
                                                                            source.GetDeviceId().HostId,
                                                                            task.Tag,
                                                                            memoryManager,
                                                                            task.Compress));
                }
            }
#endif
            return context;
        }

        bool Exec(const TCudaStream& stream,
                  TKernelContext* contextPtr) const {
            CB_ENSURE(contextPtr);
            TKernelContext& context = *contextPtr;

            if (!context.AreLocalTaskSend) {
                auto* tempBuffer = (T*)context.LocalTempBuffer.GetRawPtr();

                //different PCI root complex devices
                if (tempBuffer != nullptr) {
                    TCudaStream helperStream = NCudaLib::GetStreamsProvider().RequestStream();
                    {
                        auto event = NCudaLib::CudaEventProvider().Create();
                        event->Record(stream);
                        event->StreamWait(helperStream);
                    }

                    const ui64 blockSize = LocalBufferSize / 2;
                    bool needSync = false;

                    for (ui32 localTask = 0; localTask < LocalReduces.size(); ++localTask) {
                        auto& src = LocalReduces[localTask].Source;
                        auto& dst = LocalReduces[localTask].Dest;

                        if (!context.IsPeerLocalTask[localTask]) {
                            for (ui64 iter = 0; iter * blockSize < dst.Size(); ++iter) {
                                const ui64 offset = iter * blockSize;
                                const TCudaStream* currentStream = (iter % 2 == 0) ? &stream : &helperStream;
                                const ui64 tempBufferOffset = (iter % 2 == 0) ? 0 : blockSize;
                                const ui64 size = Min<ui64>(blockSize, dst.Size() - offset);
                                if (size) {
                                    CopyMemoryAsync(src.Get() + offset,
                                                    tempBuffer + tempBufferOffset,
                                                    size,
                                                    *currentStream);
                                    NKernel::ReduceBinary(dst.Get() + offset,
                                                          dst.Get() + offset,
                                                          tempBuffer + tempBufferOffset,
                                                          size,
                                                          currentStream->GetStream());
                                }
                            }
                            needSync |= dst.Size() >= blockSize;
                        }
                    }

                    if (needSync) {
                        auto event = NCudaLib::CudaEventProvider().Create();
                        event->Record(helperStream);
                        event->StreamWait(stream);
                    }
                }

                //easy with peer access
                for (ui32 localTask = 0; localTask < LocalReduces.size(); ++localTask) {
                    if (context.IsPeerLocalTask[localTask]) {
                        auto& src = LocalReduces[localTask].Source;
                        auto& dst = LocalReduces[localTask].Dest;

                        NKernel::ReduceBinary(dst.Get(), dst.Get(), src.Get(), dst.Size(), stream.GetStream());
                    }
                }
                context.AreLocalTaskSend = true;
            }

#if defined(USE_MPI)

            if (context.RunningStagedTasks.size()) {
                ExecStagedTasks(stream, &context.RunningStagedTasks);
            }
            return context.RunningStagedTasks.size() == 0;
#else
            return true;
#endif
        }

#if defined(USE_MPI)
        Y_SAVELOAD_DEFINE(LocalReduces, RemoteReduces);
#else
        Y_SAVELOAD_DEFINE(LocalReduces);
#endif
    private:
        template <class TBuffer, EReduceAlgorithm>
        friend class TReducer;

        TVector<TLocalHostReduce> LocalReduces;
#if defined(USE_MPI)
        TVector<TRemoteHostReduce> RemoteReduces;
#endif
    };

    template <class TBuffer, EReduceAlgorithm Method = EReduceAlgorithm::Tree>
    class TReducer {
    private:
    public:
        using T = typename TBuffer::TValueType;

        TReducer& SetCustomStream(ui32 stream) {
            Y_UNUSED(stream);
            return *this;
        }

        TReducer& operator()(TBuffer& data) {
            Y_UNUSED(data);
            return *this;
        }
    };

    struct TReduceTask {
        ui32 ReadDevice;
        ui32 WriteDevice;

        TSlice FromSlice;
        TSlice ToSlice;
    };

    template <EReduceAlgorithm>
    class TPassTasksGenerator;

    template <>
    class TPassTasksGenerator<EReduceAlgorithm::Ring> {
    private:
        const TStripeMapping& ResultMapping;
        ui32 DevCount;

        ui64 GetSrcOffset(ui32 dev) const {
            return ResultMapping.GetObjectsSlice().Size() * dev;
        }

    public:
        TPassTasksGenerator(const TStripeMapping& resultMapping,
                            ui32 devCount)
            : ResultMapping(resultMapping)
            , DevCount(devCount)
        {
        }

        //
        inline TVector<TReduceTask> PassTasks(ui32 pass) const {
            pass = DevCount - pass - 1;

            TVector<TReduceTask> tasks;
            for (ui32 dev = 0; dev < DevCount; ++dev) {
                //pass0: DevCount - 1
                //pass1: DevCount - 2
                //pass3: DevCount - 3
                //...
                //last: DevCount - (DevCount - 2) - 1 => 1
                const ui32 workingPart = (dev + pass) % DevCount;
                TReduceTask reduceTask;

                //this dev
                reduceTask.ReadDevice = dev;
                //next dev in ring
                reduceTask.WriteDevice = (dev + 1) % DevCount;

                TSlice slice = ResultMapping.DeviceSlice(workingPart);

                reduceTask.FromSlice = slice;
                reduceTask.FromSlice += GetSrcOffset(reduceTask.ReadDevice);

                reduceTask.ToSlice = slice;
                reduceTask.ToSlice += GetSrcOffset(reduceTask.WriteDevice);

                tasks.push_back(reduceTask);
            }
            return tasks;
        }

        inline ui32 GetPassCount() const {
            return DevCount - 1;
        }
    };

    template <>
    class TPassTasksGenerator<EReduceAlgorithm::Tree> {
    private:
        const TStripeMapping& ResultMapping;
        ui32 DevCount;
        ui32 PassCount;

        ui64 GetSrcOffset(ui32 dev) const {
            return ResultMapping.GetObjectsSlice().Size() * dev;
        }

    public:
        TPassTasksGenerator(const TStripeMapping& resultMapping,
                            ui32 devCount)
            : ResultMapping(resultMapping)
            , DevCount(devCount)
            , PassCount(NCB::IntLog2(DevCount))
        {
        }

        //on each pass parts with pass-bit == 0 will flow left, and with pass bit == 1 will flow right
        //on each pass — reduce between dev and (dev | mask) (pass bit in first dev should be zero)
        inline TVector<TReduceTask> PassTasks(ui32 pass) const {
            const ui32 mask = 1 << pass;
            TVector<TReduceTask> tasks;

            for (ui32 dev = 0; dev < (1U << PassCount); ++dev) {
                if (mask & dev) {
                    continue; //pass bit is not zero
                }

                ui32 fixedBits = dev & ((1 << pass) - 1);
                ui32 partsCount = 1 << (PassCount - pass);

                for (ui32 rest = 0; rest < partsCount; ++rest) {
                    const ui32 partId = rest << pass | fixedBits;
                    TReduceTask reduceTask;

                    bool flowRight = (bool)(partId & mask);

                    reduceTask.ReadDevice = flowRight ? dev : (dev | mask);
                    reduceTask.WriteDevice = flowRight ? (dev | mask) : dev;

                    if (reduceTask.ReadDevice >= DevCount || reduceTask.WriteDevice >= DevCount) {
                        continue;
                    }

                    TSlice slice = ResultMapping.DeviceSlice(partId);
                    if (slice.IsEmpty()) {
                        continue;
                    }

                    reduceTask.FromSlice = slice;
                    reduceTask.FromSlice += GetSrcOffset(reduceTask.ReadDevice);

                    reduceTask.ToSlice = slice;
                    reduceTask.ToSlice += GetSrcOffset(reduceTask.WriteDevice);

                    tasks.push_back(reduceTask);
                }
            }
            return tasks;
        }

        inline ui32 GetPassCount() const {
            return PassCount;
        }
    };

    //make from (dev0, ABCD) (dev1, ABCD),  (dev2 , ABCD) (dev3, ABCD) (dev0, A), (dev1, B), (dev2, C), (dev3, D)
    //this version trade off some speed to memory (use memory for one extra block if we don't have peer access support
    template <class T, EReduceAlgorithm ReduceType>
    class TReducer<TCudaBuffer<T, TStripeMapping>, ReduceType> {
    private:
        ui32 Stream = 0;

        using TKernel = TReduceBinaryStreamTask<T>;

    public:
        using TBuffer = TCudaBuffer<T, TStripeMapping>;

        TReducer(ui32 stream = 0)
            : Stream(stream)
        {
        }

        TReducer& operator()(TBuffer& data,
                             const TStripeMapping& resultMapping,
                             const bool compressFlag = false) {
#ifndef USE_MPI
            Y_UNUSED(compressFlag);
#endif

            auto& manager = GetCudaManager();
            const auto& beforeMapping = data.GetMapping();
            const ui64 devCount = GetDeviceCount();

            if (devCount == 1) {
                return *this;
            }

            {
                ui64 firstDevSize = beforeMapping.DeviceSlice(0).Size();
                for (auto dev : beforeMapping.NonEmptyDevices()) {
                    CB_ENSURE(beforeMapping.DeviceSlice(dev).Size() == firstDevSize,
                              "Error: Buffer dev sizes should be equal for reduce");
                }
                const ui64 columnCount = data.GetColumnCount();
                CB_ENSURE(columnCount == 1, "Error: expected 1 column for this operation");
            }

            TPassTasksGenerator<ReduceType> tasksGenerator(resultMapping, devCount);
            for (ui32 pass = 0; pass < tasksGenerator.GetPassCount(); ++pass) {
                auto tasks = tasksGenerator.PassTasks(pass);
                TStreamSectionTaskLauncher streamSectionLauncher;

                TVector<TKernel> kernels(devCount);
                TDevicesListBuilder workingDevs;

                for (const TReduceTask& task : tasks) {
                    auto fromView = data.SliceView(task.FromSlice);
                    auto toView = data.SliceView(task.ToSlice);
                    auto fromBuffer = fromView.At(task.ReadDevice);
                    auto toBuffer = toView.At(task.WriteDevice);

                    workingDevs.AddDevice(task.ReadDevice);
                    workingDevs.AddDevice(task.WriteDevice);
                    streamSectionLauncher.Group(task.ReadDevice,
                                                task.WriteDevice);

                    const bool isInterHostReduce = manager.GetDeviceId(task.ReadDevice).HostId != manager.GetDeviceId(task.WriteDevice).HostId;
                    if (isInterHostReduce) {
#if defined(USE_MPI)
                        const int tag = GetMpiManager().NextCommunicationTag();
                        typename TKernel::TRemoteHostReduce sendTask;
                        sendTask.Tag = tag;
                        sendTask.IsSendTask = true;
                        sendTask.Source = fromBuffer;
                        sendTask.Dest = toBuffer;
                        sendTask.Compress = compressFlag;

                        kernels[task.ReadDevice].RemoteReduces.push_back(std::move(sendTask));

                        typename TKernel::TRemoteHostReduce receiveTask;
                        receiveTask.Tag = tag;
                        receiveTask.IsSendTask = false;
                        receiveTask.Source = fromBuffer;
                        receiveTask.Dest = toBuffer;
                        receiveTask.Compress = compressFlag;
                        kernels[task.WriteDevice].RemoteReduces.push_back(std::move(receiveTask));
#else
                        CB_ENSURE(false, "MPI support is not enabled");
#endif
                    } else {
                        typename TKernel::TLocalHostReduce receiveTask;
                        receiveTask.Source = fromBuffer;
                        receiveTask.Dest = toBuffer;
                        kernels[task.WriteDevice].LocalReduces.push_back(std::move(receiveTask));
                    }
                }

                streamSectionLauncher.LaunchTask(workingDevs.Build(), [&](ui32 dev) {
                    return std::move(kernels[dev]);
                },
                                                 Stream);
            }

            auto localShifts = manager.CreateDistributedObject<TSlice>(TSlice(0, 0));
            for (auto dev : resultMapping.NonEmptyDevices()) {
                auto slice = resultMapping.DeviceSlice(dev);
                localShifts.Set(dev, slice);
            }

            using TMemShiftKernel = ::NKernelHost::TShiftMemoryKernel<T>;
            LaunchKernels<TMemShiftKernel>(resultMapping.NonEmptyDevices(), Stream, data, localShifts);
            TBuffer::SetMapping(resultMapping, data, false);
            return *this;
        }

        TReducer& operator()(TBuffer& data, bool compressFlag = false) {
            TStripeMapping mapping = data.GetMapping();
            TStripeMapping afterMapping = TStripeMapping::SplitBetweenDevices(mapping.DeviceSlice(0).Size(), mapping.SingleObjectSize());
            return (*this)(data, afterMapping, compressFlag);
        }
    };

    template <class T, EReduceAlgorithm Algorithm>
    inline void RunReduceScatter(TCudaBuffer<T, NCudaLib::TStripeMapping>& data,
                                 NCudaLib::TStripeMapping& reducedMapping,
                                 bool compress,
                                 ui32 streamId) {
        NCudaLib::TReducer<TCudaBuffer<T, NCudaLib::TStripeMapping>, Algorithm> reducer(streamId);
        reducer(data,
                reducedMapping,
                compress);
    }
}

template <class T>
inline void ReduceScatter(TCudaBuffer<T, NCudaLib::TStripeMapping>& data,
                          NCudaLib::TStripeMapping& reducedMapping,
                          bool compress,
                          ui32 streamId) {
    const bool isPowerOfTwoDevice = IsPowerOf2(NCudaLib::GetCudaManager().GetDeviceCount());
    //TODO(noxoomo): tree-reduce for non power of two devices + performance check
    if (isPowerOfTwoDevice) {
        NCudaLib::RunReduceScatter<T, NCudaLib::EReduceAlgorithm::Tree>(data, reducedMapping, compress, streamId);
    } else {
        NCudaLib::RunReduceScatter<T, NCudaLib::EReduceAlgorithm::Ring>(data, reducedMapping, compress, streamId);
    }
}
