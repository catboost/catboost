#pragma once

#include "ctr_value_table.h"
#include <util/system/mutex.h>
#include <util/system/guard.h>

struct TCtrData {
    THashMap<TModelCtrBase, TCtrValueTable> LearnCtrs;

    bool operator==(const TCtrData& other) const {
        return LearnCtrs == other.LearnCtrs;
    }

    bool operator!=(const TCtrData& other) const {
        return !(*this == other);
    }

    void Save(IOutputStream* s) const;

    void Load(IInputStream* s);
};

struct TCtrDataStreamWriter {
    TCtrDataStreamWriter(IOutputStream* out, size_t expectedCtrTablesCount)
        : StreamPtr(out)
        , ExpectedWritesCount(expectedCtrTablesCount)
    {
        ::SaveSize(StreamPtr, ExpectedWritesCount);
    }
    void SaveOneCtr(const TCtrValueTable& valTable) {
        with_lock (StreamLock) {
            Y_VERIFY(WritesCount < ExpectedWritesCount);
            ++WritesCount;
            ::SaveMany(StreamPtr, valTable);
        }
    }
    ~TCtrDataStreamWriter() {
        Y_VERIFY(WritesCount == ExpectedWritesCount);
    }
private:
    IOutputStream* StreamPtr = nullptr;
    TMutex StreamLock;
    size_t WritesCount = 0;
    size_t ExpectedWritesCount = 0;
};
