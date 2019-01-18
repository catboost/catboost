#pragma once

#include "reduce_scatter.h"
#include "buffer_resharding.h"
#include <catboost/cuda/cuda_lib/cuda_buffer.h>

namespace NCudaLib {
    //reduce-scatter followed by resharding
    template <class T>
    inline void AllReduce(TStripeBuffer<T>& tmp,
                          TMirrorBuffer<T>& dst,
                          ui32 stream = 0,
                          bool compress = false) {
        TReducer<TStripeBuffer<T>> reducer(stream);
        reducer(tmp);
        dst.Reset(NCudaLib::TMirrorMapping(tmp.GetObjectsSlice().Size(), tmp.GetMapping().SingleObjectSize()));
        Reshard(tmp, dst, stream, compress);
    }

    template <class T>
    inline TVector<std::remove_const_t<T>> ReadReduce(const TStripeBuffer<T>& tmp,
                                                      ui32 stream = 0) {
        using T_ = std::remove_const_t<T>;
        auto objectsSlice = tmp.GetMapping().DeviceSlice(0);
        TVector<T_> result;
        NCudaLib::TCudaBufferReader<TStripeBuffer<T>>(tmp)
            .SetFactorSlice(objectsSlice)
            .SetReadSlice(objectsSlice)
            .SetCustomReadingStream(stream)
            .ReadReduce(result);
        return result;
    }

    template <class T>
    inline TVector<T> ReadReduce(const TMirrorBuffer<T>& tmp,
                                 ui32 stream = 0) {
        TVector<T> result;
        tmp.DeviceView(0).Read(result, stream);
        return result;
    }

    template <class T>
    inline TVector<T> ReadReduce(const TSingleBuffer<T>& tmp,
                                 ui32 stream = 0) {
        TVector<T> result;
        tmp.Read(result, stream);
        return result;
    }

    //reduce-scatter followed by resharding
    template <class T, NCudaLib::EPtrType PtrType = NCudaLib::EPtrType::CudaDevice>
    inline void AllReduceThroughMaster(const TStripeBuffer<T>& tmp,
                                       TCudaBuffer<std::remove_const_t<T>, NCudaLib::TMirrorMapping, PtrType>& dst,
                                       ui32 stream = 0,
                                       bool compress = false) {
        Y_UNUSED(compress);
        auto objectsSlice = tmp.GetMapping().DeviceSlice(0);
        dst.Reset(NCudaLib::TMirrorMapping(objectsSlice.Size(), tmp.GetMapping().SingleObjectSize()));
        if (NCudaLib::GetCudaManager().GetDeviceCount() == 1) {
            //just copy it
            Reshard(tmp, dst, stream);
            //ensure implicit syncing consistency
            NCudaLib::GetCudaManager().WaitComplete();
        } else {
            TVector<std::remove_const_t<T>> result = ReadReduce(tmp, stream);
            dst.Write(result);
        }
    }

}
