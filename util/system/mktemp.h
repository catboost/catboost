#pragma once

#include <util/generic/fwd.h>

/*
 * Creates a unique temporary filename in specified directory.
 * If specified directory is NULL or empty, then system temporary directory is used.
 *
 * Note, that the function is not race-free, the file is guaranteed to exist at the time the function returns, but not at the time the returned name is first used.
 * Throws TSystemError on error.
 * 
 * Returned filepath has such format: dir/prefixXXXXXX.extension or dir/prefixXXXXXX
 * But win32: dir/preXXXX.tmp (prefix is up to 3 characters, extension is always tmp).
 */
TString MakeTempName(const char* wrkDir = nullptr, const char* prefix = "yandex", const char* extension = "tmp");
