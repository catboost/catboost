#pragma once

#include <catboost/cuda/cuda_lib/slice.h>
#include <catboost/cuda/cuda_lib/memory_copy_performance.h>
#include <util/generic/set.h>
#include <cmath>

namespace NCudaLib {
    template <class TFromBuffer,
              class TToBuffer>
    class TCudaBufferResharding {
    private:
        const TFromBuffer& Src;
        TToBuffer& Dst;
        ui32 Stream;
        TSlice ColumnSlice;
        bool ShiftColumnsToZeroInDst = false;
        using T = typename TFromBuffer::TValueType;
        bool CompressFlag = false;

        inline ui64 GetDstColumn(ui64 column) {
            if (ShiftColumnsToZeroInDst) {
                return column - ColumnSlice.Left;
            }
            return column;
        }

        ui64 OptimalNumBlockSize(ui64 dataSize,
                                 const TVector<ui32>& devices) const {
            if (devices.size() <= 1) {
                return 1;
            }

#ifndef USE_MPI
            if (TFromBuffer::PtrType() != EPtrType::CudaDevice) {
                return 1;
            }
#endif

            const auto& stats = GetMemoryCopyPerformance<TFromBuffer::PtrType(), TToBuffer::PtrType()>();

            double bandwidth = 0;
            double latency = 0;
            {
                for (ui64 i = 0; i < devices.size(); ++i) {
                    for (ui64 j = 0; j < i; ++j) {
                        auto from = devices[i];
                        auto to = devices[j];

                        bandwidth = std::max(bandwidth, stats.Bandwidth(from, to));
                        latency = std::max(stats.Latency(from, to), latency);
                    }
                }
            }
            //magic const. works better then theoretical result
            latency *= 4;

            ui64 blockCount = floor(sqrt(bandwidth * dataSize * (devices.size() + 1) / latency));
            return blockCount < 1 ? 1 : blockCount;
        }

        void BroadcastSlice(const TSlice& slice, ui32 device,
                            const TVector<ui32>& devices) {
            const auto& srcMapping = Src.GetMapping();
            const auto& dstMapping = Dst.GetMapping();

            const i64 dataSize = dstMapping.MemorySize(slice);
            const ui64 dataSizeInBytes = sizeof(T) * dataSize;
            const i64 numBlocks = OptimalNumBlockSize(dataSizeInBytes, devices);

            const ui64 srcDeviceMemorySize = srcMapping.MemorySize(srcMapping.DeviceSlice(device));

            if (numBlocks == 1) {
                //naive copy from src to all
                TDataCopier copier(Stream);
                for (ui64 column = ColumnSlice.Left; column < ColumnSlice.Right; ++column) {
                    const ui64 readOffset = srcMapping.DeviceMemoryOffset(device, slice) +
                                            NAligment::ColumnShift(srcDeviceMemorySize, column);

                    const ui64 writeSize = srcMapping.MemorySize(slice);

                    for (ui64 i = 0; i < devices.size(); ++i) {
                        const ui32 dev = devices[i];
                        TSlice deviceSlice = dstMapping.DeviceSlice(dev);
                        Y_ASSERT(deviceSlice.Contains(slice));
                        Y_ASSERT(writeSize == dstMapping.MemorySize(slice));

                        const ui64 writeOffset = dstMapping.DeviceMemoryOffset(dev, slice) +
                                                 NAligment::ColumnShift(dstMapping.MemorySize(deviceSlice), GetDstColumn(column));

                        copier.AddAsyncMemoryCopyTask(Src.GetBuffer(device), readOffset,
                                                      Dst.GetBuffer(dev), writeOffset, writeSize);
                    }
                }
                copier.SubmitCopy();
                return;
            }
            //ring algorithm for broadcasting data
            CB_ENSURE(numBlocks > 1);

            const i64 blockSize = ::NHelpers::CeilDivide(dataSize, numBlocks);
            ui64 iterCount = numBlocks + devices.size();

            // on iter copy from i'th device (i - iter) block to i'th + 1 device.
            for (ui64 iter = 0; iter < iterCount; ++iter) {
                TDataCopier copier(Stream);
                copier.SetCompressFlag(CompressFlag);

                for (ui64 i = 0; i < devices.size(); ++i) {
                    ui32 dstDev = devices[i];
                    i64 blockToCopy = (i64)iter - i;
                    if (!(blockToCopy >= 0 && (blockToCopy < numBlocks))) {
                        continue;
                    }
                    TSlice copySlice = TSlice(slice.Left + blockToCopy * blockSize, slice.Left + (blockToCopy + 1) * blockSize);
                    copySlice.Left = std::min(copySlice.Left, slice.Right);
                    copySlice.Right = std::min(copySlice.Right, slice.Right);
                    ui64 writeSize = dstMapping.MemorySize(copySlice);

                    TSlice deviceSlice = dstMapping.DeviceSlice(dstDev);
                    Y_ASSERT(deviceSlice.Contains(copySlice));

                    for (ui64 column = ColumnSlice.Left; column < ColumnSlice.Right; ++column) {
                        const ui64 dstColumn = GetDstColumn(column);

                        const ui64 writeOffset = dstMapping.DeviceMemoryOffset(dstDev, copySlice) +
                                                 NAligment::ColumnShift(dstMapping.MemorySize(deviceSlice), dstColumn);

                        if (i == 0) {
                            const ui64 readOffset = srcMapping.DeviceMemoryOffset(device, copySlice) +
                                                    NAligment::ColumnShift(srcDeviceMemorySize, column);

                            copier.AddAsyncMemoryCopyTask(Src.GetBuffer(device), readOffset,
                                                          Dst.GetBuffer(dstDev), writeOffset,
                                                          writeSize);
                        } else {
                            ui32 srcDev = devices[i - 1];
                            const ui64 dstDeviceMemorySize = dstMapping.MemorySize(dstMapping.DeviceSlice(srcDev));

                            const ui64 readOffset = dstMapping.DeviceMemoryOffset(srcDev, copySlice) +
                                                    NAligment::ColumnShift(dstDeviceMemorySize, dstColumn);

                            copier.AddAsyncMemoryCopyTask(Dst.GetBuffer(srcDev), readOffset,
                                                          Dst.GetBuffer(dstDev), writeOffset,
                                                          writeSize);
                        }
                    }
                }
                copier.SubmitCopy();
            }
        }

        struct TBroadcastTask {
            ui32 Device;
            TSlice Slice;

            bool operator<(const TBroadcastTask& task) const {
                return Slice.Size() < task.Slice.Size();
            }
        };

        TVector<TBroadcastTask> GetSrcSlicesByDevices(const TSlice& slice) {
            TVector<TSlice> currentSlices;
            currentSlices.push_back(slice);
            const auto& srcMapping = Src.GetMapping();

            TVector<TBroadcastTask> tasks;
            for (ui32 dev : Src.NonEmptyDevices()) {
                const TSlice devSlice = srcMapping.DeviceSlice(dev);

                TVector<TSlice> nextCurrent;

                for (auto current : currentSlices) {
                    auto intersection = TSlice::Intersection(devSlice, current);
                    if (!intersection.IsEmpty()) {
                        tasks.push_back({dev, intersection});
                    }
                    for (auto next : TSlice::Remove(current, intersection)) {
                        nextCurrent.push_back(next);
                    }
                }
                if (nextCurrent.size() == 0) {
                    break;
                }
                currentSlices.swap(nextCurrent);
            }
            return tasks;
        }

    public:
        TCudaBufferResharding(const TFromBuffer& from,
                              TToBuffer& to,
                              ui32 stream = 0)
            : Src(from)
            , Dst(to)
            , Stream(stream)
        {
            Y_ASSERT(from.GetMapping().GetObjectsSlice() == to.GetMapping().GetObjectsSlice());
            CB_ENSURE(from.GetMapping().GetObjectsSlice() == to.GetMapping().GetObjectsSlice(), TStringBuilder() << from.GetMapping().GetObjectsSlice() << "≠" << to.GetMapping().GetObjectsSlice());
            ColumnSlice = TSlice(0, from.GetColumnCount());
        }

        TCudaBufferResharding& SetShiftColumnsToZeroFlag(bool flag) {
            ShiftColumnsToZeroInDst = flag;
            return *this;
        }

        TCudaBufferResharding& SetCompressFlag(bool flag) {
            CompressFlag = flag;
            return *this;
        }

        TCudaBufferResharding& SetColumnSlice(const TSlice& columnSlice) {
            ColumnSlice = columnSlice;
            return *this;
        }

        void Run() {
            //TODO(noxoomo): tune for efficient resharding for several hosts
            const auto& srcMapping = Src.GetMapping();
            const auto& dstMapping = Dst.GetMapping();

            TVector<ui32> writeDevices;
            std::vector<TBroadcastTask> tasks;

            for (const auto dev : Dst.NonEmptyDevices()) {
                writeDevices.push_back(dev);
            }

            {
                TDataCopier copier(Stream);
                copier.SetCompressFlag(CompressFlag);

                for (ui64 i = 0; i < writeDevices.size(); ++i) {
                    ui32 dev = writeDevices[i];
                    const TSlice srcSlice = srcMapping.DeviceSlice(dev);
                    const TSlice deviceSlice = dstMapping.DeviceSlice(dev);
                    TSlice fastCopySlice = TSlice::Intersection(deviceSlice, srcSlice);

                    auto restSlices = TSlice::Remove(deviceSlice, fastCopySlice);
                    for (auto restSlice : restSlices) {
                        tasks.push_back({dev, restSlice});
                    }

                    if (fastCopySlice.Size()) {
                        const ui64 sliceMemorySize = dstMapping.MemorySize(fastCopySlice);
                        for (ui64 column = ColumnSlice.Left; column < ColumnSlice.Right; ++column) {
                            const ui64 readOffset = srcMapping.DeviceMemoryOffset(dev, fastCopySlice) + NAligment::ColumnShift(srcMapping.MemorySize(srcSlice), column);

                            const ui64 writeOffset = dstMapping.DeviceMemoryOffset(dev, fastCopySlice) +
                                                     NAligment::ColumnShift(dstMapping.MemorySize(deviceSlice), GetDstColumn(column));

                            copier.AddAsyncMemoryCopyTask(Src.GetBuffer(dev), readOffset,
                                                          Dst.GetBuffer(dev), writeOffset,
                                                          sliceMemorySize);
                        }
                    }
                }
                copier.SubmitCopy();
            }

            while (tasks.size()) {
                std::sort(tasks.begin(), tasks.end());

                TBroadcastTask task = tasks[0];
                std::vector<TBroadcastTask> nextTasks;

                TSlice broadcastSlice = task.Slice;

                for (ui32 i = 1; i < tasks.size(); ++i) {
                    auto& otherSlice = tasks[i].Slice;
                    auto intersection = TSlice::Intersection(broadcastSlice, otherSlice);
                    if (!intersection.IsEmpty()) {
                        broadcastSlice = intersection;
                    }
                }

                TVector<ui32> devicesToBroadcast;

                for (ui32 i = 0; i < tasks.size(); ++i) {
                    TSlice taskSlice = tasks[i].Slice;

                    if (taskSlice.Contains(broadcastSlice)) {
                        devicesToBroadcast.push_back(tasks[i].Device);
                    }

                    const TSlice intersection = TSlice::Intersection(taskSlice, broadcastSlice);
                    for (auto restSlice : TSlice::Remove(taskSlice, intersection)) {
                        nextTasks.push_back({tasks[i].Device, restSlice});
                    }
                }

                CB_ENSURE(!broadcastSlice.IsEmpty());
                Y_ASSERT(devicesToBroadcast.size() == TSet<ui32>(devicesToBroadcast.begin(), devicesToBroadcast.end()).size());

                TVector<TBroadcastTask> slices = GetSrcSlicesByDevices(broadcastSlice);
                CB_ENSURE(slices.size());
                for (auto broadcastTask : slices) {
                    BroadcastSlice(broadcastTask.Slice, broadcastTask.Device, devicesToBroadcast);
                }
                tasks.swap(nextTasks);
            }
        }
    };

    template <class TSrcBuffer, class TDstBuffer>
    inline void Reshard(const TSrcBuffer& src, TDstBuffer& dst, ui32 stream = 0, bool compress = false) {
        TCudaBufferResharding<TSrcBuffer, TDstBuffer> worker(src, dst, stream);
        worker.SetCompressFlag(compress)
            .Run();
    };

}
