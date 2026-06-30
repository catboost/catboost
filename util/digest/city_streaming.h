#pragma once
#include <util/digest/city.h>

/**
 *   (partially) streaming version of CityHash64 for large data.
 *   You need to know length and first/last 64 bytes.
 *   Those bytes should be passed twice: in constructor and thru process().
 *   Length must be STRICTLY larger than 64 bytes.
 *   XXX: Dont use CityHash64 if you can use something else and need streaming
 */
class TStreamingCityHash64 {
    ui64 x, y, z;
    std::pair<ui64, ui64> v, w;
    char UnalignBuf_[64];
    size_t UnalignBufSz_, Rest64_;

public:
    TStreamingCityHash64(size_t len, const char* head64, const char* tail64);
    void Process(const char* s, size_t avail);
    ui64 operator()();
};
