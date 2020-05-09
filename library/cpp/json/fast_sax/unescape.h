#pragma once

#include <util/generic/strbuf.h>

TStringBuf UnescapeJsonUnicode(TStringBuf data, char* scratch);
