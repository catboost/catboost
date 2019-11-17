#pragma once

#include "stream.h"

#include <util/generic/hash.h>

class TCompressionCodecFactory {
public:
    using TDecoderConstructor = std::function<THolder<IInputStream>(IInputStream*)>;
    using TEncoderConstructor = std::function<THolder<IOutputStream>(IOutputStream*)>;

    TCompressionCodecFactory();

    static inline TCompressionCodecFactory& Instance() noexcept {
        return *SingletonWithPriority<TCompressionCodecFactory, 0>();
    }

    inline const TDecoderConstructor* FindDecoder(TStringBuf name) const {
        if (auto codec = Codecs_.FindPtr(name)) {
            return &codec->Decoder;
        }

        return nullptr;
    }

    inline const TEncoderConstructor* FindEncoder(TStringBuf name) const {
        if (auto codec = Codecs_.FindPtr(name)) {
            return &codec->Encoder;
        }

        return nullptr;
    }

    inline TArrayRef<const TStringBuf> GetBestCodecs() const {
        return BestCodecs_;
    }

private:
    void Add(TStringBuf name, TDecoderConstructor d, TEncoderConstructor e);

    struct TCodec {
        TDecoderConstructor Decoder;
        TEncoderConstructor Encoder;
    };

    TVector<TString> Strings_;
    THashMap<TStringBuf, TCodec> Codecs_;
    TVector<TStringBuf> BestCodecs_;
};
