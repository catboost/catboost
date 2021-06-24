#pragma once

#include <util/generic/fwd.h>
#include <util/system/defaults.h>

class IOutputStream;

size_t BackTrace(void** p, size_t len);

struct TResolvedSymbol {
    const char* Name;
    void* NearestSymbol;
};

TResolvedSymbol ResolveSymbol(void* sym, char* buf, size_t len);

void FormatBackTrace(IOutputStream* out, void* const* backtrace, size_t backtraceSize);
void FormatBackTrace(IOutputStream* out);
void PrintBackTrace();

using TFormatBackTraceFn = void (*)(IOutputStream*, void* const* backtrace, size_t backtraceSize);

TFormatBackTraceFn SetFormatBackTraceFn(TFormatBackTraceFn f);
TFormatBackTraceFn GetFormatBackTraceFn();

using TBackTraceView = TArrayRef<void* const>;

class TBackTrace {
private:
    static constexpr size_t CAPACITY = 300;
    void* Data[CAPACITY];
    size_t Size;

public:
    TBackTrace();
    void Capture();
    void PrintTo(IOutputStream&) const;
    TString PrintToString() const;
    size_t size() const;
    const void* const* data() const;
    operator TBackTraceView() const;

    static TBackTrace FromCurrentException();
};
