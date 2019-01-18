#include "reorder_bins.h"

#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_lib/cuda_kernel_buffer.h>
#include <catboost/cuda/cuda_lib/kernel.h>
#include <catboost/cuda/cuda_util/kernel/sort.cuh>
#include <catboost/cuda/cuda_util/kernel/transform.cuh>
#include <catboost/cuda/cuda_util/kernel/reorder_one_bit.cuh>
#include <catboost/libs/helpers/exception.h>

namespace NKernelHost {
    template <typename T>
    class TReorderOneBitKernel: public TKernelBase<NKernel::TReorderOneBitContext<ui32, T>, false> {
    private:
        TCudaBufferPtr<ui32> Keys;
        TCudaBufferPtr<T> Values;
        int Bit = 0;

    public:
        using TKernelContext = NKernel::TReorderOneBitContext<ui32, T>;
        Y_SAVELOAD_DEFINE(Keys, Values, Bit);

        THolder<TKernelContext> PrepareContext(IMemoryManager& memoryManager) const {
            CB_ENSURE(Keys.Size() == Values.Size());
            CB_ENSURE(Keys.Size() < (1u << 31));

            auto context = MakeHolder<TKernelContext>();
            context->ScanTempBufferSize = NKernel::ReorderBitTempSize(Keys.Size());
            context->ScanTempBuffer = memoryManager.Allocate<char>(context->ScanTempBufferSize);

            context->TempKeys = memoryManager.Allocate<ui32>(Keys.Size());
            context->Offsets = memoryManager.Allocate<i32>(Keys.Size());
            context->TempValues = memoryManager.Allocate<T>(Keys.Size());
            return context;
        }

        TReorderOneBitKernel() = default;

        TReorderOneBitKernel(TCudaBufferPtr<ui32> keys,
                             TCudaBufferPtr<T> values,
                             int bit)
            : Keys(keys)
            , Values(values)
            , Bit(bit)
        {
        }

        void Run(const TCudaStream& stream, TKernelContext& context) {
            NKernel::ReorderOneBit(Keys.Size(), context, Keys.Get(), Values.Get(), Bit, stream.GetStream());
        }
    };
}

using NCudaLib::TMirrorMapping;
using NCudaLib::TSingleMapping;
using NCudaLib::TStripeMapping;
using NKernelHost::IMemoryManager;
using NKernelHost::TCudaBufferPtr;
using NKernelHost::TCudaStream;
using NKernelHost::TKernelBase;
using NKernelHost::uchar;

template <class TMapping>
void ReorderOneBitImpl(
    NCudaLib::TCudaBuffer<ui32, TMapping>& bins,
    NCudaLib::TCudaBuffer<ui32, TMapping>& indices,
    int offset,
    ui32 stream = 0) {
    using TKernel = NKernelHost::TReorderOneBitKernel<ui32>;
    LaunchKernels<TKernel>(bins.NonEmptyDevices(), stream, bins, indices, offset);
}

#define Y_CATBOOST_CUDA_IMPL(TMapping)                 \
    template <>                                        \
    void ReorderOneBit<TMapping>(                      \
        TCudaBuffer<ui32, TMapping> & bins,            \
        TCudaBuffer<ui32, TMapping> & indices,         \
        i32 bit,                                       \
        ui32 stream) {                                 \
        ReorderOneBitImpl(bins, indices, bit, stream); \
    }

Y_MAP_ARGS(
    Y_CATBOOST_CUDA_IMPL,
    TMirrorMapping,
    TSingleMapping,
    TStripeMapping);

#undef Y_CATBOOST_CUDA_IMPL
