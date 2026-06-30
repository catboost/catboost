#pragma once

#include "stream.h"

#include <util/generic/deque.h>
#include <util/generic/hash.h>
#include <util/generic/vector.h>

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

    TDeque<TString> Strings_;
    THashMap<TStringBuf, TCodec> Codecs_;
    TVector<TStringBuf> BestCodecs_;
};

namespace NHttp {
    template <typename F>
    TString ChooseBestCompressionScheme(F accepted, TArrayRef<const TStringBuf> available) {
        if (available.empty()) {
            return "identity";
        }

        if (accepted("*")) {
            return TString(available[0]);
        }

        for (const auto& coding : available) {
            TString s(coding);
            if (accepted(s)) {
                return s;
            }
        }

        return "identity";
    }
}
