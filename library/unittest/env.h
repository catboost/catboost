#pragma once

#include <util/generic/string.h>
#include <util/generic/strbuf.h>
#include <util/system/src_location.h>

TString ArcadiaSourceRoot();
TString ArcadiaFromCurrentLocation(TStringBuf where, TStringBuf path);
TString BuildRoot();
TString BinaryPath(TStringBuf path);
bool FromYaTest();

#define SRC_(path) ArcadiaFromCurrentLocation(__SOURCE_FILE__, path)
