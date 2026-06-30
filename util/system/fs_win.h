#pragma once

#include "winint.h"
#include "defaults.h"

#include <util/generic/strbuf.h>
#include <util/generic/string.h>

namespace NFsPrivate {
    bool WinRename(const TString& oldPath, const TString& newPath);

    bool WinSymLink(const TString& targetName, const TString& linkName);

    bool WinHardLink(const TString& existingPath, const TString& newPath);

    TString WinReadLink(const TString& path);

    ULONG WinReadReparseTag(HANDLE h);

    HANDLE CreateFileWithUtf8Name(const TStringBuf fName, ui32 accessMode, ui32 shareMode, ui32 createMode, ui32 attributes, bool inheritHandle);

    bool WinRemove(const TString& path);

    bool WinExists(const TString& path);

    TString WinCurrentWorkingDirectory();

    bool WinSetCurrentWorkingDirectory(const TString& path);

    bool WinMakeDirectory(const TString& path);
} // namespace NFsPrivate
