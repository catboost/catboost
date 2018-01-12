#pragma once

#include <catboost/cuda/cuda_lib/kernel/reduce.cuh>
#include <catboost/cuda/cuda_lib/kernel.h>
#include <catboost/cuda/cuda_lib/cuda_kernel_buffer.h>
#include <catboost/cuda/cuda_lib/mapping.h>
#include <catboost/cuda/cuda_lib/cuda_buffer.h>

namespace NKernelHost {
    template <typename T>
    class TSingleHostReduceBinaryKernel: public TKernelBase<NKernel::TKernelWithTempBufferContext<T>, false> {
    private:
        TCudaBufferPtr<const T> Src;
        TCudaBufferPtr<T> Dst;

        mutable bool HasPeerAccess = false;

    public:
        using TKernelContext = typename NKernel::TKernelWithTempBufferContext<T>;

        TSingleHostReduceBinaryKernel() = default;

        TSingleHostReduceBinaryKernel(TCudaBufferPtr<const T> src,
                                      TCudaBufferPtr<T> dst)
            : Src(src)
            , Dst(dst)
        {
        }

        TSingleHostReduceBinaryKernel(TSingleHostReduceBinaryKernel&& other) = default;

        Y_SAVELOAD_DEFINE(Dst, Src);

        THolder<TKernelContext> PrepareContext(IMemoryManager& memoryManager) const {
            auto context = MakeHolder<TKernelContext>();
            CB_ENSURE(Src.Size() == Dst.Size());

            ui32 sourceDevId = NCudaLib::GetDeviceForPointer(Src.Get());
            ui32 dstDevId = NCudaLib::GetDeviceForPointer(Dst.Get());

            HasPeerAccess = NCudaLib::GetPeerDevicesHelper().HasPeerAccess(sourceDevId, dstDevId);

            if (!HasPeerAccess) {
                //TODO(noxoomo): make temp memory more robust
                context->TempBuffer = memoryManager.Allocate<T>(Src.Size()).Get();
            }
            return context;
        }

        void Run(const TCudaStream& stream,
                 TKernelContext& context) const {
            if (context.TempBuffer == nullptr) {
                CB_ENSURE(HasPeerAccess);
                NKernel::ReduceBinary(Dst.Get(), Dst.Get(), Src.Get(), (ui32)Dst.Size(), stream.GetStream());
            } else {
                NCudaLib::TMemoryCopier<EPtrType::CudaDevice, EPtrType::CudaDevice>::template CopyMemoryAsync<T>(Src.Get(), context.TempBuffer, Src.Size(), stream);
                NKernel::ReduceBinary(Dst.Get(), Dst.Get(), context.TempBuffer, Dst.Size(), stream.GetStream());
            }
        }
    };

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
                context->TempBuffer = memoryManager.Allocate<T>(Data.SliceMemorySize(Slice)).Get();
            }
            return context;
        }

        void Run(const TCudaStream& stream,
                 TKernelContext& context) const {
            if (context.TempBuffer != nullptr) {
                NCudaLib::TMemoryCopier<EPtrType::CudaDevice, EPtrType::CudaDevice>::template CopyMemoryAsync<T>(Data.Get() + Data.SliceMemoryOffset(Slice), context.TempBuffer,
                                                                                                                 Data.SliceMemorySize(
                                                                                                                     Slice),
                                                                                                                 stream);
                NCudaLib::TMemoryCopier<EPtrType::CudaDevice, EPtrType::CudaDevice>::template CopyMemoryAsync<T>(context.TempBuffer, Data.Get(),
                                                                                                                 Data.SliceMemorySize(
                                                                                                                     Slice),
                                                                                                                 stream);
            }
        }
    };
}

namespace NCudaLib {
    template <class TBuffer>
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

    //make from (dev0, ABCD) (dev1, ABCD),  (dev2 , ABCD) (dev3, ABCD) (dev0, A), (dev1, B), (dev2, C), (dev3, D)
    //this version trade off some speed to memory (use memory for one extra block if we don't have peer access support
    //and works fro single-host only
    template <class T>
    class TReducer<TCudaBuffer<T, TStripeMapping>> {
    private:
        ui32 Stream = 0;
        using TKernel = typename NKernelHost::TSingleHostReduceBinaryKernel<T>;

        struct TReduceTask {
            ui32 ReadDevice;
            ui32 WriteDevice;

            TSlice FromSlice;
            TSlice ToSlice;
        };

        class TPassTasksGenerator {
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
                , PassCount((ui32)log2(DevCount))
            {
            }

            //on each pass parts with pass-bit == 0 will flow left, and with pass bit == 1 will flow right
            //on each pass — reduce between dev and (dev | mask) (pass bit in first dev should be zero)
            inline TVector<TReduceTask> PassTasks(ui32 pass) const {
                const ui32 mask = 1 << pass;
                TVector<TReduceTask> tasks;

                for (ui32 dev = 0; dev < DevCount; ++dev) {
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
                        reduceTask.WriteDevice = flowRight ? dev | mask : dev;

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

    public:
        using TBuffer = TCudaBuffer<T, TStripeMapping>;

        TReducer(ui32 stream = 0)
            : Stream(stream)
        {
        }

        TReducer& operator()(TBuffer& data,
                             const TStripeMapping& resultMapping) {
            TSingleHostStreamSync streamSync(Stream);
            auto& manager = GetCudaManager();
            streamSync();

            const auto& beforeMapping = data.GetMapping();
            const ui64 devCount = GetDeviceCount();

            if (devCount == 1) {
                return *this;
            }
            for (ui32 dev = 0; dev < devCount; ++dev) {
                streamSync.AddDevice(dev);
            }

            {
                ui64 firstDevSize = beforeMapping.DeviceSlice(0).Size();
                for (auto dev : beforeMapping.NonEmptyDevices()) {
                    CB_ENSURE(beforeMapping.DeviceSlice(dev).Size() == firstDevSize,
                              "Error: Buffer dev sizes should be equal for reduce");
                }
                const ui64 columnCount = data.GetColumnSlice().Size();
                CB_ENSURE(columnCount == 1, "Error: expected 1 column for this operation");
            }

            TPassTasksGenerator tasksGenerator(resultMapping, devCount);
            for (ui32 pass = 0; pass < tasksGenerator.GetPassCount(); ++pass) {
                auto tasks = tasksGenerator.PassTasks(pass);

                for (const TReduceTask& task : tasks) {
                    auto fromView = data.SliceView(task.FromSlice);
                    auto toView = data.SliceView(task.ToSlice);
                    auto fromBuffer = fromView.At(task.ReadDevice);
                    auto toBuffer = toView.At(task.WriteDevice);

                    auto kernel = TKernel(fromBuffer, toBuffer);
                    LaunchKernel<TKernel>(task.WriteDevice, Stream, std::move(kernel));
                }
                streamSync();
            }

            auto localShifts = manager.CreateDistributedObject<TSlice>();
            for (auto dev : resultMapping.NonEmptyDevices()) {
                auto slice = resultMapping.DeviceSlice(dev);
                localShifts.Set(dev, slice);
            }
            manager.WaitComplete();

            using TMemShiftKernel = ::NKernelHost::TShiftMemoryKernel<T>;
            LaunchKernels<TMemShiftKernel>(resultMapping.NonEmptyDevices(), Stream, data, localShifts);
            TBuffer::SetMapping(resultMapping, data, false);
            return *this;
        }

        TReducer& operator()(TBuffer& data) {
            TStripeMapping mapping = data.GetMapping();
            TStripeMapping afterMapping = TStripeMapping::SplitBetweenDevices(mapping.DeviceSlice(0).Size(), mapping.SingleObjectSize());
            return (*this)(data, afterMapping);
        }
    };
}
