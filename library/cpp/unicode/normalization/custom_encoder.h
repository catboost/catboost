#pragma once

#include <library/charset/codepage.h>

struct TCustomEncoder: public Encoder {
    void Create(const CodePage* target, bool extended = false);
    ~TCustomEncoder();

private:
    void addToTable(wchar32 ucode, unsigned char code, const CodePage* target);
};
