#include "input.h"
#include "output.h"
#include "str.h"

#include <util/charset/wide.h>
#include <util/memory/tempbuf.h>
#include <util/generic/string.h>
#include <util/generic/yexception.h>
#include <util/generic/singleton.h>
#include <util/string/cast.h>
#include <util/system/compat.h>
#include <util/system/spinlock.h>

#include <cstdlib>

IInputStream::IInputStream() noexcept = default;

IInputStream::~IInputStream() = default;

size_t IInputStream::DoReadTo(TString& st, char to) {
    char ch;

    if (!Read(&ch, 1)) {
        return 0;
    }

    st.clear();

    size_t result = 0;
    do {
        ++result;

        if (ch == to) {
            break;
        }

        st += ch;
    } while (Read(&ch, 1));

    return result;
}

ui64 IInputStream::DoReadAll(IOutputStream& out) {
    TTempBuf buffer;
    void* ptr = buffer.Data();
    size_t size = buffer.Size();

    ui64 result = 0;
    while (size_t read = Read(ptr, size)) {
        out.Write(ptr, read);
        result += read;
    }

    return result;
}

size_t IInputStream::Load(void* buf_in, size_t len) {
    char* buf = (char*)buf_in;

    while (len) {
        const size_t ret = Read(buf, len);

        buf += ret;
        len -= ret;

        if (ret == 0) {
            break;
        }
    }

    return buf - (char*)buf_in;
}

void IInputStream::LoadOrFail(void* buf, size_t len) {
    const size_t realLen = Load(buf, len);
    if (Y_UNLIKELY(realLen != len)) {
        ythrow yexception() << "Failed to read required number of bytes from stream! Expected: " << len << ", gained: " << realLen << "!";
    }
}

size_t IInputStream::ReadLine(TString& st) {
    const size_t ret = ReadTo(st, '\n');

    if (ret && !st.empty() && st.back() == '\r') {
        st.pop_back();
    }

    return ret;
}

size_t IInputStream::ReadLine(TUtf16String& w) {
    TString s;
    size_t result = ReadLine(s);

    if (result) {
        UTF8ToWide(s, w);
    }

    return result;
}

TString IInputStream::ReadLine() {
    TString ret;

    if (!ReadLine(ret)) {
        ythrow yexception() << "can not read line from stream";
    }

    return ret;
}

TString IInputStream::ReadTo(char ch) {
    TString ret;

    if (!ReadTo(ret, ch)) {
        ythrow yexception() << "can not read from stream";
    }

    return ret;
}

size_t IInputStream::Skip(size_t sz) {
    return DoSkip(sz);
}

size_t IInputStream::DoSkip(size_t sz) {
    if (sz < 128) {
        return Load(alloca(sz), sz);
    }

    TTempBuf buf;
    size_t total = 0;

    while (sz) {
        const size_t lresult = Read(buf.Data(), Min<size_t>(sz, buf.Size()));

        if (lresult == 0) {
            return total;
        }

        total += lresult;
        sz -= lresult;
    }

    return total;
}

TString IInputStream::ReadAll() {
    TString result;
    TStringOutput stream(result);

    DoReadAll(stream);

    return result;
}

ui64 IInputStream::ReadAll(IOutputStream& out) {
    return DoReadAll(out);
}

ui64 TransferData(IInputStream* in, IOutputStream* out) {
    return in->ReadAll(*out);
}

namespace {
    struct TStdIn: public IInputStream {
        ~TStdIn() override = default;

        size_t DoRead(void* buf, size_t len) override {
            const size_t ret = fread(buf, 1, len, F_);

            if (ret < len && ferror(F_)) {
                ythrow TSystemError() << "can not read from stdin";
            }

            return ret;
        }

        FILE* F_ = stdin;
    };

#if defined(_win_)
    using TGetLine = TStdIn;
#else
    #if defined(_bionic_)
    using TGetLineBase = TStdIn;
    #else
    struct TGetLineBase: public TStdIn {
        ~TGetLineBase() override {
            free(B_);
        }

        size_t DoReadTo(TString& st, char ch) override {
            auto&& guard = Guard(M_);

            (void)guard;

            const auto r = getdelim(&B_, &L_, ch, F_);

            if (r < 0) {
                if (ferror(F_)) {
                    ythrow TSystemError() << "can not read from stdin";
                }

                st.clear();

                return 0;
            }

            st.AssignNoAlias(B_, r);

            if (st && st.back() == ch) {
                st.pop_back();
            }

            return r;
        }

        TAdaptiveLock M_;
        char* B_ = nullptr;
        size_t L_ = 0;
    };
    #endif

    #if defined(_glibc_) || defined(_cygwin_)
    // glibc does not have fgetln
    using TGetLine = TGetLineBase;
    #else
    struct TGetLine: public TGetLineBase {
        size_t DoReadTo(TString& st, char ch) override {
            if (ch == '\n') {
                size_t len = 0;
                auto r = fgetln(F_, &len);

                if (r) {
                    st.AssignNoAlias(r, len);

                    if (st && st.back() == '\n') {
                        st.pop_back();
                    }

                    return len;
                }
            }

            return TGetLineBase::DoReadTo(st, ch);
        }
    };
    #endif
#endif
}

IInputStream& NPrivate::StdInStream() noexcept {
    return *SingletonWithPriority<TGetLine, 4>();
}

// implementation of >> operator

// helper functions

static inline bool IsStdDelimiter(char c) {
    return (c == '\0') || (c == ' ') || (c == '\r') || (c == '\n') || (c == '\t');
}

static void ReadUpToDelimiter(IInputStream& i, TString& s) {
    char c;
    while (i.ReadChar(c)) { // skip delimiters
        if (!IsStdDelimiter(c)) {
            s += c;
            break;
        }
    }
    while (i.ReadChar(c) && !IsStdDelimiter(c)) { // read data (with trailing delimiter)
        s += c;
    }
}

// specialization for string-related stuff

template <>
void In<TString>(IInputStream& i, TString& s) {
    s.resize(0);
    ReadUpToDelimiter(i, s);
}

template <>
void In<TUtf16String>(IInputStream& i, TUtf16String& w) {
    TString s;
    ReadUpToDelimiter(i, s);

    if (s.empty()) {
        w.erase();
    } else {
        w = UTF8ToWide(s);
    }
}

// specialization for char types

#define SPEC_FOR_CHAR(T)                  \
    template <>                           \
    void In<T>(IInputStream & i, T & t) { \
        i.ReadChar((char&)t);             \
    }

SPEC_FOR_CHAR(char)
SPEC_FOR_CHAR(unsigned char)
SPEC_FOR_CHAR(signed char)

#undef SPEC_FOR_CHAR

// specialization for number types

#define SPEC_FOR_NUMBER(T)                                                       \
    template <>                                                                  \
    void In<T>(IInputStream & i, T & t) {                                        \
        char buf[128];                                                           \
        size_t pos = 0;                                                          \
        while (i.ReadChar(buf[0])) {                                             \
            if (!IsStdDelimiter(buf[0])) {                                       \
                ++pos;                                                           \
                break;                                                           \
            }                                                                    \
        }                                                                        \
        while (i.ReadChar(buf[pos]) && !IsStdDelimiter(buf[pos]) && pos < 127) { \
            ++pos;                                                               \
        }                                                                        \
        t = FromString<T, char>(buf, pos);                                       \
    }

SPEC_FOR_NUMBER(signed short)
SPEC_FOR_NUMBER(signed int)
SPEC_FOR_NUMBER(signed long int)
SPEC_FOR_NUMBER(signed long long int)
SPEC_FOR_NUMBER(unsigned short)
SPEC_FOR_NUMBER(unsigned int)
SPEC_FOR_NUMBER(unsigned long int)
SPEC_FOR_NUMBER(unsigned long long int)

SPEC_FOR_NUMBER(float)
SPEC_FOR_NUMBER(double)
SPEC_FOR_NUMBER(long double)

#undef SPEC_FOR_NUMBER
