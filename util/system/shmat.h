#pragma once

#include "fhandle.h"

#include <util/generic/ptr.h>
#include <util/generic/guid.h>

class TSharedMemory: public TThrRefBase {
    TGUID Id;
    FHANDLE Handle;
    void* Data;
    int Size;

public:
    TSharedMemory();
    ~TSharedMemory() override;

    bool Create(int Size);
    bool Open(const TGUID& id, int size);

    const TGUID& GetId() {
        return Id;
    }

    void* GetPtr() {
        return Data;
    }

    int GetSize() const {
        return Size;
    }
};
