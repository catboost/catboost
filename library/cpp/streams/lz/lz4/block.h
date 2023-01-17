#pragma once

#include <library/cpp/streams/lz/common/compressor.h>

#include <contrib/libs/lz4/lz4.h>

/*
 * LZ4
 */
class TLZ4 {
public:
    static constexpr char signature[]= "LZ.4";

    static inline size_t Hint(size_t len) noexcept {
        return Max<size_t>((size_t)(len * 1.06), 100);
    }

    inline size_t Compress(const char* data, size_t len, char* ptr, size_t dstMaxSize) {
        return LZ4_compress_default(data, ptr, len, dstMaxSize);
    }

    inline size_t Decompress(const char* data, size_t len, char* ptr, size_t max) {
        int res = LZ4_decompress_safe(data, ptr, len, max);
        if (res < 0)
            ythrow TDecompressorError();
        return res;
    }

    inline void InitFromStream(IInputStream*) const noexcept {
    }

    static inline bool SaveIncompressibleChunks() noexcept {
        return false;
    }
};
