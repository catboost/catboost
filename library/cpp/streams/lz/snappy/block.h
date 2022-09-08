#pragma once

#include <library/cpp/streams/lz/common/compressor.h>

#include <contrib/libs/snappy/snappy.h>

/*
 * Snappy
 */
class TSnappy {
public:
    static constexpr char signature[] = "Snap";

    static inline size_t Hint(size_t len) noexcept {
        return Max<size_t>(snappy::MaxCompressedLength(len), 100);
    }

    inline size_t Compress(const char* data, size_t len, char* ptr, size_t /*dstMaxSize*/) {
        size_t reslen = 0;
        snappy::RawCompress(data, len, ptr, &reslen);
        return reslen;
    }

    inline size_t Decompress(const char* data, size_t len, char* ptr, size_t) {
        size_t srclen = 0;
        if (!snappy::GetUncompressedLength(data, len, &srclen) || !snappy::RawUncompress(data, len, ptr))
            ythrow TDecompressorError();
        return srclen;
    }

    inline void InitFromStream(IInputStream*) const noexcept {
    }

    static inline bool SaveIncompressibleChunks() noexcept {
        return false;
    }
};
