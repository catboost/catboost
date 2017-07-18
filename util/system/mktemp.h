#pragma once

class TString;

/*
 * Create temp file name in the specified directory.
 * If wrkDir is NULL or empty, GetSystemTempDir is used.
 * throw exception on error
 */
TString MakeTempName(const char* wrkDir = nullptr, const char* prefix = "yandex");
