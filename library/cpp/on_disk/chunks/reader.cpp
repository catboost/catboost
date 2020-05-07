#include <util/generic/cast.h>
#include <util/memory/blob.h>
#include <util/system/unaligned_mem.h>

#include "reader.h"

template <typename T>
static inline void ReadAux(const char* data, T* aux, T count, TVector<const char*>* result) {
    result->resize(count);
    for (size_t i = 0; i < count; ++i) {
        (*result)[i] = data + ReadUnaligned<T>(aux + i);
    }
}

TChunkedDataReader::TChunkedDataReader(const TBlob& blob) {
    const char* cdata = blob.AsCharPtr();
    const size_t size = blob.Size();
    Y_ENSURE(size >= sizeof(ui32), "Empty file with chunks. ");

    ui32 last = ReadUnaligned<ui32>((ui32*)(cdata + size) - 1);

    if (last != 0) { // old version file
        ui32* aux = (ui32*)(cdata + size);
        ui32 count = last;
        Size = size - (count + 1) * sizeof(ui32);

        aux -= (count + 1);
        ReadAux<ui32>(cdata, aux, count, &Offsets);
        return;
    }

    Y_ENSURE(size >= 3 * sizeof(ui64), "Blob size must be >= 3 * sizeof(ui64). ");

    ui64* aux = (ui64*)(cdata + size);
    Version = ReadUnaligned<ui64>(aux - 2);
    Y_ENSURE(Version > 0, "Invalid chunked array version. ");

    ui64 count = ReadUnaligned<ui64>(aux - 3);

    aux -= (count + 3);
    ReadAux<ui64>(cdata, aux, count, &Offsets);

    aux -= count;
    Lengths.resize(count);
    for (size_t i = 0; i < count; ++i) {
        Lengths[i] = IntegerCast<size_t>(ReadUnaligned<ui64>(aux + i));
    }
}

TBlob TChunkedDataReader::GetBlob(size_t index) const {
    return TBlob::NoCopy(GetBlock(index), GetBlockLen(index));
}
