#pragma once

#include "codecs.h"

namespace NBlockCodecs{

    struct TStreamingBlockEncoderFactory {
        THolder<IOutputStream> (*ForString)(const ICodec&, TString&);
        THolder<IOutputStream> (*ForKnownSize)(const ICodec&, IOutputStream&, ui64);
    };

    void RegisterCodec(TCodecPtr codec);
    // The factory must have static lifetime and provide both creation methods.
    void RegisterCodec(TCodecPtr codec, const TStreamingBlockEncoderFactory& factory);
    void RegisterAlias(TStringBuf from, TStringBuf to);

}
