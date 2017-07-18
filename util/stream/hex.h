#pragma once

#include <util/system/types.h>

class TOutputStream;

void HexEncode(const void* in, size_t len, TOutputStream& out);
void HexDecode(const void* in, size_t len, TOutputStream& out);
