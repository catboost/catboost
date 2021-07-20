#include "util.h"

#include <util/generic/utility.h>

#include <cstdio>
#include <cstdarg>
#include <cstdlib>

int a2i(const TString& s) {
    return atoi(s.c_str());
}

//============================== span =====================================

void str_spn::init(const char* charset, bool extended) {
    // chars_table_1 is necessary to avoid some unexpected
    // multi-threading issues
    ui8 chars_table_1[256];
    memset(chars_table_1, 0, sizeof(chars_table_1));
    if (extended) {
        for (const char* cs = charset; *cs; cs++) {
            if (cs[1] == '-' && cs[2] != 0) {
                for (int c = (ui8)*cs; c <= (ui8)cs[2]; c++) {
                    chars_table_1[c] = 1;
                }
                cs += 2;
                continue;
            }
            chars_table_1[(ui8)*cs] = 1;
        }
    } else {
        for (; *charset; charset++) {
            chars_table_1[(ui8)*charset] = 1;
        }
    }
    memcpy(chars_table, chars_table_1, 256);
    chars_table_1[0] = 1;
    for (int n = 0; n < 256; n++) {
        c_chars_table[n] = !chars_table_1[n];
    }
}

Tr::Tr(const char* from, const char* to) {
    for (size_t n = 0; n < 256; n++) {
        Map[n] = (char)n;
    }
    for (; *from && *to; from++, to++) {
        Map[(ui8)*from] = *to;
    }
}

size_t Tr::FindFirstChangePosition(const TString& str) const {
    for (auto it = str.begin(); it != str.end(); ++it) {
        if (ConvertChar(*it) != *it) {
            return it - str.begin();
        }
    }

    return TString::npos;
}

void Tr::Do(TString& str) const {
    const size_t changePosition = FindFirstChangePosition(str);

    if (changePosition == TString::npos) {
        return;
    }

    for (auto it = str.begin() + changePosition; it != str.end(); ++it) {
        *it = ConvertChar(*it);
    }
}
