#pragma once

#include <util/generic/fwd.h>
#include <util/generic/vector.h>
#include <util/stream/output.h>
#include <util/system/types.h>
#include <cstring>


/* This class is needed because we need some resizable buffer transfer between JVM and C++
 *  and TString can't play this role because of string encoding issues
 */
class TVectorOutput: public IOutputStream {
public:
    TVectorOutput(TVector<i8>* buf)
        : Buf(buf)
    {}
private:
    void DoWrite(const void* buf, size_t len) override {
        size_t oldSize = Buf->size();
        Buf->yresize(oldSize + len);
        std::memcpy(Buf->data() + oldSize, buf, len);
    }
private:
    TVector<i8>* Buf;
};
