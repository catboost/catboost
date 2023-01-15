#include "ysaveload.h"

#include <util/generic/buffer.h>

void TSerializer<TBuffer>::Save(IOutputStream* rh, const TBuffer& buf) {
    ::SaveSize(rh, buf.Size());
    ::SavePodArray(rh, buf.Data(), buf.Size());
}

void TSerializer<TBuffer>::Load(IInputStream* rh, TBuffer& buf) {
    const size_t s = ::LoadSize(rh);
    buf.Resize(s);
    ::LoadPodArray(rh, buf.Data(), buf.Size());
}

[[noreturn]] void NPrivate::ThrowLoadEOFException(size_t typeSize, size_t realSize, TStringBuf structName) {
    ythrow TLoadEOF() << "can not load " << structName << "(" << typeSize << ", " << realSize << " bytes)";
}

[[noreturn]] void NPrivate::ThrowUnexpectedVariantTagException(ui8 tagIndex) {
    ythrow TLoadEOF() << "Unexpected tag value " << tagIndex << " while loading TVariant";
}
