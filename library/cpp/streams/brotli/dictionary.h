#pragma once

#include "const.h"

#include <contrib/libs/brotli/c/include/brotli/encode.h>

#include <util/generic/strbuf.h>

class TBrotliDictionary: public TMoveOnly {
public:
    TBrotliDictionary(const ui8* dict, size_t size, int quality = NBrotli::BEST_BROTLI_QUALITY)
        : Data_(dict)
        , Size_(size)
        , PreparedDictionary_(
              BrotliEncoderPrepareDictionary(
                  /*type*/ BROTLI_SHARED_DICTIONARY_RAW,
                  /*size*/ Size_,
                  /*data*/ Data_,
                  /*quality*/ quality,
                  /*alloc_func*/ nullptr,
                  /*free_func*/ nullptr,
                  /*opaque*/ nullptr))
    {
    }

    explicit TBrotliDictionary(TStringBuf dict, int quality = NBrotli::BEST_BROTLI_QUALITY)
        : TBrotliDictionary(
              reinterpret_cast<const ui8*>(dict.data()),
              dict.size(),
              quality)
    {
    }

    ~TBrotliDictionary() {
        BrotliEncoderDestroyPreparedDictionary(PreparedDictionary_);
    }

    BrotliEncoderPreparedDictionary* GetPreparedDictionary() const {
        return PreparedDictionary_;
    }

    const ui8* GetData() const {
        return Data_;
    }

    size_t GetSize() const {
        return Size_;
    }

private:
    const ui8* Data_ = nullptr;
    const size_t Size_ = 0;
    BrotliEncoderPreparedDictionary* PreparedDictionary_ = nullptr;
};
