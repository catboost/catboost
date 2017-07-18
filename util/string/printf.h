#pragma once

#include <util/system/compiler.h>

#include <cstdarg>

class TString;

/// formatted print. return printed length:
int Y_PRINTF_FORMAT(2, 0) vsprintf(TString& s, const char* c, va_list params);
/// formatted print. return printed length:
int Y_PRINTF_FORMAT(2, 3) sprintf(TString& s, const char* c, ...);
TString Y_PRINTF_FORMAT(1, 2) Sprintf(const char* c, ...);
int Y_PRINTF_FORMAT(2, 3) fcat(TString& s, const char* c, ...);
