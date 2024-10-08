#pragma once


namespace NUCompress {
    // These limitations come from original implementation - library/python/compress
    using TBlockLen = ui32;
    constexpr TBlockLen MaxCompressedLen = 100000000;
}
