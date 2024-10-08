#pragma once

#include <util/generic/buffer.h>
#include <util/stream/output.h>


namespace NBlockCodecs {
    struct ICodec;
}

namespace NUCompress {
    class TCodedOutput: public IOutputStream {
    public:
        TCodedOutput(IOutputStream* out, const NBlockCodecs::ICodec* c, size_t bufLen = 16 << 20);
        ~TCodedOutput() override;

    private:
        void DoWrite(const void* buf, size_t len) override;
        void DoFlush() override;
        void DoFinish() override;

        void FlushImpl();

    private:
        const NBlockCodecs::ICodec* const C_;
        TBuffer D_;
        TBuffer O_;
        IOutputStream* S_;
        bool HdrWritten = false;
    };
}
