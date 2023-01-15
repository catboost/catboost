#pragma once

#include <util/generic/fwd.h>

/*
 * Create temp file name in the specified directory. If there is no file with this name, it will be created.
 * Note, that the function is not race-free, the file is guaranteed to exist at the time the function returns, but not at the time the returned name is first used.
 * If wrkDir is NULL or empty, GetSystemTempDir is used.
 * throw exception on error
 */
TString MakeTempName(const char* wrkDir = nullptr, const char* prefix = "yandex");
