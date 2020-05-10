#pragma once

#include "bin_saver.h"
#include "buffered_io.h"

#include <util/memory/blob.h>

class TYaBlobStream: public IBinaryStream {
    TBlob Blob;
    i64 Pos;

    int WriteImpl(const void*, int) override {
        Y_ASSERT(0);
        return 0;
    }
    int ReadImpl(void* userBuffer, int size) override {
        if (size == 0)
            return 0;
        i64 res = Min<i64>(Blob.Length() - Pos, size);
        if (res)
            memcpy(userBuffer, ((const char*)Blob.Data()) + Pos, res);
        Pos += res;
        return res;
    }
    bool IsValid() const override {
        return true;
    }
    bool IsFailed() const override {
        return false;
    }

public:
    TYaBlobStream(const TBlob& blob)
        : Blob(blob)
        , Pos(0)
    {
    }
};

template <class T>
inline void SerializeBlob(const TBlob& data, T& c) {
    TYaBlobStream f(data);
    {
        IBinSaver bs(f, true);
        bs.Add(1, &c);
    }
}
