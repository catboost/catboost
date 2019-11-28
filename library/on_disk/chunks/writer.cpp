#include <util/ysaveload.h>

#include "writer.h"

static inline void WriteAux(IOutputStream* out, const TVector<ui64>& data) {
    ::SavePodArray(out, data.data(), data.size());
}

/*************************** TBuffersWriter ***************************/

TChunkedDataWriter::TChunkedDataWriter(IOutputStream& slave)
    : Slave(slave)
    , Offset(0)
{
}

TChunkedDataWriter::~TChunkedDataWriter() {
}

void TChunkedDataWriter::NewBlock() {
    if (Offsets.size()) {
        Lengths.push_back(Offset - Offsets.back());
    }

    Pad(16);
    Offsets.push_back(Offset);
}

void TChunkedDataWriter::WriteFooter() {
    Lengths.push_back(Offset - Offsets.back());
    WriteAux(this, Lengths);
    WriteAux(this, Offsets);
    WriteBinary<ui64>(Offsets.size());
    WriteBinary<ui64>(Version);
    WriteBinary<ui64>(0);
}

size_t TChunkedDataWriter::GetCurrentBlockOffset() const {
    Y_ASSERT(!Offsets.empty());
    Y_ASSERT(Offset >= Offsets.back());
    return Offset - Offsets.back();
}

size_t TChunkedDataWriter::GetBlockCount() const {
    return Offsets.size();
}
