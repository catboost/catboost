#pragma once

#include <catboost/libs/cuda_wrappers/base.h>
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
    private:
        TCudaStream DefaultStream_;
        TVector<TCudaStream> HelperStreams_;
    };
}
