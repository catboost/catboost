#include "single_device.h"

namespace NCudaLib {
    void TCudaSingleDevice::RequestHandlesImpl() {
        if (FreeHandles.size() == 0) {
            auto request = LaunchFunc(TRequestHandlesTask(OBJECT_HANDLE_REQUEST_SIZE));
            request->Wait();
            FreeHandles = request->Get();
            TotalHandles += FreeHandles.size();
        }
    }

    ui64 TCudaSingleDevice::GetFreeHandle() {
        RequestHandlesImpl();
        Y_ASSERT(FreeHandles.size());
        auto handle = FreeHandles.back();
        FreeHandles.pop_back();
        return handle;
    }

    ui32 TCudaSingleDevice::RequestStream() {
        if (UserFreeStreams.size() == 0) {
            THolder<IDeviceFuture<ui32>> streamFuture;
            if (IsLocalDevice()) {
                streamFuture = RequestStreamImpl<false>();
            } else {
#if defined(USE_MPI)
                streamFuture = RequestStreamImpl<true>();
#else
                CB_ENSURE(false, "Remote device support is not enabled");
#endif
            }
            streamFuture->Wait();
            UserFreeStreams.push_back(streamFuture->Get());
        }
        ui32 id = UserFreeStreams.back();
        UserFreeStreams.pop_back();
        CB_ENSURE(id != 0);
        return id;
    }

    TMemoryState TCudaSingleDevice::GetMemoryState() {
        using TFunc = TMemoryStateFunc;
        TFunc func;
        return LaunchFunc<TFunc>(std::move(func))->Get();
    }
}
