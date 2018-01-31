#pragma once

#include "mpi_promise_future.h"
#include "local_promise_future.h"
#include <catboost/cuda/cuda_lib/device_id.h>

namespace NCudaLib {
    template <bool IsRemote>
    class TPromiseFactory;

#if defined(USE_MPI)
    template <>
    class TPromiseFactory<true> {
    public:
        template <class T>
        using TPromise = TMpiPromise<T>;

        template <class T>
        static inline TPromise<T> CreateDevicePromise(const TDeviceId& deviceId) {
            CB_ENSURE(deviceId.HostId != 0, "Error: can't create remote promise on master host");
            return TPromise<T>(deviceId.HostId, TMpiManager::GetMasterId());
        }
    };
#endif

    template <>
    class TPromiseFactory<false> {
    public:
        template <class T>
        using TPromise = TLocalHostPromise<T>;

        template <class T>
        static inline TPromise<T> CreateDevicePromise(const TDeviceId& deviceId) {
            CB_ENSURE(deviceId.HostId == 0, "Error: can't create local promise on remote host");
            return TPromise<T>();
        }
    };

}
