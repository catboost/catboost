#pragma once

#include <library/cpp/cuda/wrappers/base.h>
#include <library/cpp/cuda/wrappers/cuda_event.h>
#include <util/system/types.h>
#include <util/generic/vector.h>

namespace NCudaNN {

    class TStreamPool {
    public:

        explicit TStreamPool(TCudaStream defaultStream, ui64 initSize = 1);

        void RequestSize(ui64 requestedSize);


        TCudaStream DefaultStream() {
            return DefaultStream_;
        }

        ui64 StreamCount() const {
            return 1 + HelperStreams_.size();
        }


        operator TCudaStream() const {
            return DefaultStream_;
        }

        TCudaStream Stream(ui64 idx) const {
            return idx == 0 ? DefaultStream_ : HelperStreams_[idx - 1];
        }

        void DefaultWaitForPrefix(ui64 size);
        void PrefixWaitForDefault(ui64 size);
    private:
        TCudaStream DefaultStream_;
        TCudaEvent DefaultEvent_;
        TVector<TCudaEvent> SyncEvents_;
        TVector<TCudaStream> HelperStreams_;
    };
}
