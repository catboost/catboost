#pragma once

#include <catboost/cuda/cuda_lib/cuda_base.h>
#include <catboost/cuda/cuda_lib/memory_provider_trait.h>
#include <catboost/cuda/cuda_lib/remote_objects.h>
#include <catboost/cuda/cuda_lib/single_device.h>
#include <catboost/cuda/cuda_lib/cuda_kernel_buffer.h>
#include <catboost/cuda/cuda_lib/cuda_manager.h>
#include <catboost/cuda/cuda_lib/tasks_impl/memory_copy_tasks.h>
#include <catboost/cuda/cuda_lib/mapping.h>
#include <util/generic/vector.h>

namespace NCudaLib {
    template <class TCudaBuffer>
    class TCudaBufferWriter: public TMoveOnly {
    private:
        using T = typename TCudaBuffer::TValueType;
        const T* Src;
        TCudaBuffer* Dst;
        ui64 SrcMaxSize;
        TSlice WriteSlice;
        ui32 Stream = 0;
        bool Async = false;
        TVector<THolder<IDeviceRequest>> WriteDone;
        ui64 SrcOffset = 0;
        TSlice ColumnWriteSlice;

    public:
        TCudaBufferWriter(const TVector<T>& src,
                          TCudaBuffer& dst)
            : Src(~src)
            , Dst(&dst)
            , SrcMaxSize(src.size())
            , WriteSlice(dst.GetMapping().GetObjectsSlice())
            , ColumnWriteSlice(dst.GetColumnSlice())
        {
        }

        TCudaBufferWriter(TCudaBufferWriter&& other) = default;

        ~TCudaBufferWriter() throw (yexception) {
            for (auto& event : WriteDone) {
                CB_ENSURE(event->IsComplete());
            }
        }

        TCudaBufferWriter& SetWriteSlice(const TSlice& slice) {
            WriteSlice = slice;
            return *this;
        }

        TCudaBufferWriter& SetCustomWritingStream(ui32 stream) {
            Stream = stream;
            return *this;
        }

        TCudaBufferWriter& SetColumnWriteSlice(const TSlice& slice) {
            ColumnWriteSlice = slice;
            return *this;
        }

        TCudaBufferWriter& SetCustomSrcOffset(ui64 offset) {
            SrcOffset = offset;
            return *this;
        }

        //TODO(noxoomo): compressed write support
        void Write() {
            const auto& mapping = Dst->GetMapping();
            for (auto dev : Dst->NonEmptyDevices()) {
                ui64 columnOffset = 0;

                for (ui64 column = ColumnWriteSlice.Left; column < ColumnWriteSlice.Right; ++column) {
                    auto deviceSlice = mapping.DeviceSlice(dev);
                    auto intersection = TSlice::Intersection(WriteSlice, deviceSlice);
                    const ui64 memoryUsageAtDevice = mapping.MemoryUsageAt(dev);

                    if (!intersection.IsEmpty()) {
                        const auto localWriteOffset = mapping.DeviceMemoryOffset(dev, intersection) + memoryUsageAtDevice * column;
                        const ui64 writeSize = mapping.MemorySize(intersection);
                        ui64 readOffset = mapping.MemoryOffset(intersection);
                        CB_ENSURE(readOffset >= SrcOffset);
                        readOffset -= SrcOffset;
                        Y_ASSERT(writeSize <= SrcMaxSize);
                        CB_ENSURE(writeSize <= SrcMaxSize);

                        WriteDone.push_back(TDataCopier::AsyncWrite(Src + readOffset + columnOffset,
                                                                    Dst->GetBuffer(dev),
                                                                    Stream,
                                                                    localWriteOffset,
                                                                    writeSize));
                    }
                    columnOffset += memoryUsageAtDevice;
                }
            }

            if (!Async) {
                WaitComplete();
            }
        }

        void WaitComplete() {
            for (auto& task : WriteDone) {
                task->WaitComplete();
            }
        }
    };

}
