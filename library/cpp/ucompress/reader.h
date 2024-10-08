#pragma once

#include <util/generic/buffer.h>
#include <util/stream/walk.h>


namespace NBlockCodecs {
    struct ICodec;
}

namespace NUCompress {
    class TDecodedInput: public IWalkInput {
    public:
        TDecodedInput(IInputStream* in);
        ~TDecodedInput() override;

    private:
        size_t DoUnboundedNext(const void** ptr) override;

    private:
        IInputStream* const S_;
        const NBlockCodecs::ICodec* C_ = nullptr;
        TBuffer D_;
    };
}
