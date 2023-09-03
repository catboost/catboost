#pragma once


#include "ctr_value_table.h"
#include "online_ctr.h"

#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/logging/logging.h>

#include <util/generic/hash.h>
#include <util/stream/fwd.h>
#include <util/system/mutex.h>
#include <util/system/guard.h>
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
    void LoadNonOwning(TMemoryInput* in);
};

class TCtrDataStreamWriter {
public:
    TCtrDataStreamWriter(IOutputStream* out, size_t expectedCtrTablesCount)
        : StreamPtr(out)
        , ExpectedWritesCount(expectedCtrTablesCount)
    {
        ::SaveSize(StreamPtr, ExpectedWritesCount);
    }
    ~TCtrDataStreamWriter() noexcept(false) {
        if (WritesCount != ExpectedWritesCount) {
            CATBOOST_ERROR_LOG << "Some CTR data are lost" << Endl;
            if (!std::uncaught_exceptions()) {
                CB_ENSURE(WritesCount == ExpectedWritesCount);
            }
        }
    }
    void SaveOneCtr(const TCtrValueTable& valTable) {
        with_lock (StreamLock) {
            CB_ENSURE(WritesCount < ExpectedWritesCount, "Too many calls to SaveOneCtr");
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
