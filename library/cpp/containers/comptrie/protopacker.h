#pragma once

#include <util/stream/mem.h>
#include <util/ysaveload.h>

template <class Proto>
class TProtoPacker {
public:
    TProtoPacker() = default;

    void UnpackLeaf(const char* p, Proto& entry) const {
        TMemoryInput in(p + sizeof(ui32), SkipLeaf(p) - sizeof(ui32));
        entry.ParseFromArcadiaStream(&in);
    }
    void PackLeaf(char* p, const Proto& entry, size_t size) const {
        TMemoryOutput out(p, size + sizeof(ui32));
        Save<ui32>(&out, size);
        entry.SerializeToArcadiaStream(&out);
    }
    size_t MeasureLeaf(const Proto& entry) const {
        return entry.ByteSize() + sizeof(ui32);
    }
    size_t SkipLeaf(const char* p) const {
        TMemoryInput in(p, sizeof(ui32));
        ui32 size;
        Load<ui32>(&in, size);
        return size;
    }
};
