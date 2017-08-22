#pragma once

#include "defaults.h"

class IOutputStream;
class TString;

size_t BackTrace(void** p, size_t len);

struct TResolvedSymbol {
    const char* Name;
    void* NearestSymbol;
};

TResolvedSymbol ResolveSymbol(void* sym, char* buf, size_t len);

void FormatBackTrace(IOutputStream* out, void* const* backtrace, size_t backtraceSize);
void FormatBackTrace(IOutputStream* out);
void PrintBackTrace();

using TFormatBackTraceFn = void (*)(IOutputStream*);

TFormatBackTraceFn SetFormatBackTraceFn(TFormatBackTraceFn f);

class TBackTrace {
private:
    static const size_t CAPACITY = 300;
    void* Data[CAPACITY];
    size_t Size;

public:
    TBackTrace();
    void Capture();
    void PrintTo(IOutputStream&) const;
    TString PrintToString() const;
};
