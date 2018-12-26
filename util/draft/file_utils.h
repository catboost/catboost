#pragma once

#include <util/system/compat.h>
#include <util/system/error.h>
#include <util/system/fs.h>
#include <util/folder/dirut.h>
#include <util/generic/noncopyable.h>
#include <util/generic/string.h>
#include <util/stream/file.h>
#include <util/string/cast.h>
#include <util/generic/yexception.h>

TString GetDir(const TString& filePath);
TString GetFilename(const TString& filePath);

// returns -1 if not found
i64 GetFileSize(const TString& srcLocation);

TString ResolveLocation(TString fileLocation, const TString& homeDir);
