#pragma once

#include <catboost/cuda/cuda_lib/cuda_base.h>
#include <catboost/cuda/cuda_lib/remote_objects.h>
#include <catboost/cuda/cuda_lib/single_device.h>
#include <catboost/cuda/cuda_lib/cuda_kernel_buffer.h>
#include <catboost/cuda/cuda_lib/cuda_manager.h>
#include <catboost/cuda/cuda_lib/tasks_impl/memory_copy_tasks.h>
#include <catboost/cuda/cuda_lib/mapping.h>
#include <catboost/cuda/cuda_lib/cpu_reducers.h>
#include <util/generic/vector.h>

namespace NCudaLib {
    template <class TCudaBuffer>
    class TCudaBufferReader: public TMoveOnly {
    private:
        const TCudaBuffer* Buffer;
        ui32 Stream = 0;
        TVector<THolder<IDeviceRequest>> ReadDone;

        TSlice FactorSlice;
        //slice to read: ReadSlice read all slices, that equal mod ReduceSlice to read slice
        TSlice ReadSlice;
        //columns to read
        TSlice ColumnReadSlice;

    protected:
        ui64 ReadMemorySize(const TSlice& slice) const {
            auto& mapping = Buffer->GetMapping();
            return mapping.MemorySize(slice);
        }

        TVector<TSlice> GetReadSlices() {
            Y_ASSERT(ReadSlice.Right <= FactorSlice.Right);
            CB_ENSURE(ReadSlice.Right <= FactorSlice.Right);

            auto& mapping = Buffer->GetMapping();
            TSlice objectsSlice = mapping.GetObjectsSlice();

            TVector<TSlice> readSlices;
            if (ReadSlice.IsEmpty()) {
                return readSlices;
            }

            ui64 step = FactorSlice.Size();
            for (TSlice cursor = ReadSlice; cursor.Right <= objectsSlice.Right; cursor += step) {
                readSlices.push_back(cursor);
            }

            return readSlices;
        }

    public:
        using T = typename TCudaBuffer::TValueType;
        using NonConstT = std::remove_const_t<typename TCudaBuffer::TValueType>;

        TCudaBufferReader(const TCudaBuffer& buffer)
            : Buffer(&buffer)
            , FactorSlice(buffer.GetMapping().GetObjectsSlice())
            , ReadSlice(buffer.GetMapping().GetObjectsSlice())
            , ColumnReadSlice(TSlice(0, buffer.GetColumnCount()))
        {
        }

        TCudaBufferReader& SetReadSlice(const TSlice& slice) {
            ReadSlice = slice;
            return *this;
        }

        TCudaBufferReader& SetFactorSlice(const TSlice& slice) {
            FactorSlice = slice;
            return *this;
        }

        TCudaBufferReader& SetColumnReadSlice(const TSlice& slice) {
            ColumnReadSlice = slice;
            return *this;
        }

        TCudaBufferReader& SetCustomReadingStream(ui32 stream) {
            Stream = stream;
            return *this;
        }

        void WaitComplete() {
            for (auto& task : ReadDone) {
                task->WaitComplete();
            }
        }

        void ReadAsync(TVector<NonConstT>& dst) {
            auto readSlices = GetReadSlices();
            ui64 singleSliceSize = ReadMemorySize(ReadSlice) * ColumnReadSlice.Size();
            dst.resize(readSlices.size() * singleSliceSize);

            for (ui64 i = 0; i < readSlices.size(); ++i) {
                SubmitReadAsync(dst.data() + i * singleSliceSize, readSlices[i]);
            }
        }

        void Read(TVector<NonConstT>& dst) {
            ReadAsync(dst);
            WaitComplete();
        }

        template <class TReducer = NReducers::TSumReducer<NonConstT>>
        void ReadReduce(TVector<NonConstT>& dst) {
            Read(dst);
            ui64 reduceStep = FactorSlice.Size();
            auto& mapping = Buffer->GetMapping();
            TSlice objectsSlice = mapping.GetObjectsSlice();

            auto reduceSize = mapping.MemorySize(ReadSlice);

            TSlice reduceSlice = ReadSlice;
            reduceSlice += reduceStep;

            for (; reduceSlice.Right <= objectsSlice.Right; reduceSlice += reduceStep) {
                auto appendOffset = mapping.MemoryOffset(reduceSlice);
                auto appendSize = mapping.MemorySize(reduceSlice);
                CB_ENSURE(appendSize == reduceSize, "Error: reduce size should be equal append size");
                TReducer::Reduce(dst.data(), dst.data() + appendOffset, appendSize);
            }
            dst.resize(reduceSize);
        }

        void SubmitReadAsync(NonConstT* to, const TSlice& readSlice) {
            if (readSlice.Size()) {
                auto& mapping = Buffer->GetMapping();
                const ui64 skipOffset = mapping.MemoryOffset(readSlice);

                for (ui64 column = ColumnReadSlice.Left; column < ColumnReadSlice.Right; ++column) {
                    TVector<TSlice> currentSlices;
                    currentSlices.push_back(readSlice);

                    for (auto dev : Buffer->NonEmptyDevices()) {
                        TVector<TSlice> nextSlices;

                        for (auto& slice : currentSlices) {
                            const TSlice& deviceSlice = mapping.DeviceSlice(dev);
                            auto deviceSlicePart = TSlice::Intersection(slice, deviceSlice);

                            if (!deviceSlicePart.IsEmpty()) {
                                if (slice.Left < deviceSlicePart.Left) {
                                    nextSlices.push_back({slice.Left, deviceSlicePart.Left});
                                }

                                if (slice.Right > deviceSlicePart.Right) {
                                    nextSlices.push_back({deviceSlicePart.Right, slice.Right});
                                }

                                const ui64 localDataOffset = mapping.DeviceMemoryOffset(dev, deviceSlicePart) +
                                                             NAligment::ColumnShift(mapping.MemoryUsageAt(dev), column);

                                const ui64 writeOffset = mapping.MemoryOffset(deviceSlicePart) - skipOffset;

                                ReadDone.push_back(
                                    TDataCopier::AsyncRead(Buffer->GetBuffer(dev),
                                                           Stream,
                                                           localDataOffset,
                                                           to + writeOffset,
                                                           mapping.MemorySize(deviceSlicePart)));
                            } else {
                                nextSlices.push_back(slice);
                            }
                        }

                        if (nextSlices.size() == 0) {
                            break;
                        }
                        currentSlices.swap(nextSlices);
                    }

                    to += mapping.MemorySize(readSlice);
                }
            }
        }
    };

}
