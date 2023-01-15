#pragma once

#include "ctr_value_table.h"

#include <util/generic/hash.h>
#include <util/stream/fwd.h>
#include <util/system/mutex.h>
#include <util/system/guard.h>
#include <util/system/yassert.h>
#include <util/ysaveload.h>

#include <exception>


struct TCtrData {
    THashMap<TModelCtrBase, TCtrValueTable> LearnCtrs;

public:
    bool operator==(const TCtrData& other) const {
        return LearnCtrs == other.LearnCtrs;
    }

    bool operator!=(const TCtrData& other) const {
        return !(*this == other);
    }

    void Save(IOutputStream* s) const;

    void Load(IInputStream* s);
};

class TCtrDataStreamWriter {
public:
    TCtrDataStreamWriter(IOutputStream* out, size_t expectedCtrTablesCount)
        : StreamPtr(out)
        , ExpectedWritesCount(expectedCtrTablesCount)
    {
        ::SaveSize(StreamPtr, ExpectedWritesCount);
    }
    ~TCtrDataStreamWriter() {
        if (!std::uncaught_exception()) {
            Y_VERIFY(WritesCount == ExpectedWritesCount);
        }
    }
    void SaveOneCtr(const TCtrValueTable& valTable) {
        with_lock (StreamLock) {
            Y_VERIFY(WritesCount < ExpectedWritesCount);
            ++WritesCount;
            ::SaveMany(StreamPtr, valTable);
        }
    }

private:
    IOutputStream* StreamPtr = nullptr;
    TMutex StreamLock;
    size_t WritesCount = 0;
    size_t ExpectedWritesCount = 0;
};
