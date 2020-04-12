#pragma once

#include "contrib/libs/tensorboard/event.pb.h"

#include <util/generic/string.h>
#include <util/generic/ptr.h>
#include <util/stream/file.h>
#include <util/string/cast.h>
#include <util/folder/dirut.h>
#include <util/folder/path.h>

#include <library/cpp/digest/crc32c/crc32c.h>

#include <ctime>

class TTensorBoardLogger {
private:
    THolder<TOFStream> OutputStream;

    int AddEvent(int64_t step, THolder<tensorboard::Summary>* summary);
    int Write(tensorboard::Event& event);

public:
    TTensorBoardLogger() = default;
    TTensorBoardLogger(const TString& logDir);
    int AddScalar(const TString& tag, int step, float value);
};
