#pragma once

#if defined(__cplusplus)
extern "C" {
#endif

struct TLZ4Methods {
    int (*LZ4CompressLimited)(const char* source, char* dest, int isize, int maxOut);
};

struct TLZ4Methods* LZ4Methods(int memory);

#if defined(__cplusplus)
}
#endif
