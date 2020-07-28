#include "output.h"

#include <util/string/cast.h>
#include "format.h"
#include <util/memory/tempbuf.h>
#include <util/generic/singleton.h>
#include <util/generic/yexception.h>
#include <util/charset/utf8.h>
#include <util/charset/wide.h>

#if defined(_android_)
#include <util/system/dynlib.h>
#include <util/system/guard.h>
#include <util/system/mutex.h>
#include <android/log.h>
#endif

#include <cerrno>
#include <string>
#include <string_view>
#include <cstdio>

#if defined(_win_)
#include <io.h>
#endif

constexpr size_t MAX_UTF8_BYTES = 4; // UTF-8-encoded code point takes between 1 and 4 bytes

IOutputStream::IOutputStream() noexcept = default;

IOutputStream::~IOutputStream() = default;

void IOutputStream::DoFlush() {
    /*
     * do nothing
     */
}

void IOutputStream::DoFinish() {
    Flush();
}

void IOutputStream::DoWriteV(const TPart* parts, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        const TPart& part = parts[i];

        DoWrite(part.buf, part.len);
    }
}

void IOutputStream::DoWriteC(char ch) {
    DoWrite(&ch, 1);
}

template <>
void Out<wchar16>(IOutputStream& o, wchar16 ch) {
    const wchar32 w32ch = ReadSymbol(&ch, &ch + 1);
    size_t length;
    unsigned char buffer[MAX_UTF8_BYTES];
    WriteUTF8Char(w32ch, length, buffer);
    o.Write(buffer, length);
}

template <>
void Out<wchar32>(IOutputStream& o, wchar32 ch) {
    size_t length;
    unsigned char buffer[MAX_UTF8_BYTES];
    WriteUTF8Char(ch, length, buffer);
    o.Write(buffer, length);
}

static void WriteString(IOutputStream& o, const wchar16* w, size_t n) {
    const size_t buflen = (n * MAX_UTF8_BYTES); // * 4 because the conversion functions can convert unicode character into maximum 4 bytes of UTF8
    TTempBuf buffer(buflen + 1);
    char* const data = buffer.Data();
    size_t written = 0;
    WideToUTF8(w, n, data, written);
    data[written] = 0;
    o.Write(data, written);
}

static void WriteString(IOutputStream& o, const wchar32* w, size_t n) {
    const size_t buflen = (n * MAX_UTF8_BYTES); // * 4 because the conversion functions can convert unicode character into maximum 4 bytes of UTF8
    TTempBuf buffer(buflen + 1);
    char* const data = buffer.Data();
    size_t written = 0;
    WideToUTF8(w, n, data, written);
    data[written] = 0;
    o.Write(data, written);
}

template <>
void Out<TString>(IOutputStream& o, const TString& p) {
    o.Write(p.data(), p.size());
}

template <>
void Out<std::string>(IOutputStream& o, const std::string& p) {
    o.Write(p.data(), p.length());
}

template <>
void Out<std::string_view>(IOutputStream& o, const std::string_view& p) {
    o.Write(p.data(), p.length());
}

template <>
void Out<std::u16string_view>(IOutputStream& o, const std::u16string_view& p) {
    WriteString(o, p.data(), p.length());
}

template <>
void Out<std::u32string_view>(IOutputStream& o, const std::u32string_view& p) {
    WriteString(o, p.data(), p.length());
}

template <>
void Out<TStringBuf>(IOutputStream& o, const TStringBuf& p) {
    o.Write(p.data(), p.length());
}

template <>
void Out<TWtringBuf>(IOutputStream& o, const TWtringBuf& p) {
    WriteString(o, p.data(), p.length());
}

template <>
void Out<TUtf32StringBuf>(IOutputStream& o, const TUtf32StringBuf& p) {
    WriteString(o, p.data(), p.length());
}

template <>
void Out<const wchar16*>(IOutputStream& o, const wchar16* w) {
    if (w) {
        WriteString(o, w, TCharTraits<wchar16>::GetLength(w));
    } else {
        o.Write("(null)");
    }
}

template <>
void Out<const wchar32*>(IOutputStream& o, const wchar32* w) {
    if (w) {
        WriteString(o, w, TCharTraits<wchar32>::GetLength(w));
    } else {
        o.Write("(null)");
    }
}

template <>
void Out<TUtf16String>(IOutputStream& o, const TUtf16String& w) {
    WriteString(o, w.c_str(), w.size());
}

template <>
void Out<TUtf32String>(IOutputStream& o, const TUtf32String& w) {
    WriteString(o, w.c_str(), w.size());
}

#define DEF_CONV_DEFAULT(type)                  \
    template <>                                 \
    void Out<type>(IOutputStream & o, type p) { \
        o << ToString(p);                       \
    }

#define DEF_CONV_CHR(type)                      \
    template <>                                 \
    void Out<type>(IOutputStream & o, type p) { \
        o.Write((char)p);                       \
    }

#define DEF_CONV_NUM(type, len)                                   \
    template <>                                                   \
    void Out<type>(IOutputStream & o, type p) {                   \
        char buf[len];                                            \
        o.Write(buf, ToString(p, buf, sizeof(buf)));              \
    }                                                             \
                                                                  \
    template <>                                                   \
    void Out<volatile type>(IOutputStream & o, volatile type p) { \
        Out<type>(o, p);                                          \
    }

DEF_CONV_NUM(bool, 64)

DEF_CONV_CHR(char)
DEF_CONV_CHR(signed char)
DEF_CONV_CHR(unsigned char)

DEF_CONV_NUM(signed short, 64)
DEF_CONV_NUM(signed int, 64)
DEF_CONV_NUM(signed long int, 64)
DEF_CONV_NUM(signed long long int, 64)

DEF_CONV_NUM(unsigned short, 64)
DEF_CONV_NUM(unsigned int, 64)
DEF_CONV_NUM(unsigned long int, 64)
DEF_CONV_NUM(unsigned long long int, 64)

DEF_CONV_NUM(float, 512)
DEF_CONV_NUM(double, 512)
DEF_CONV_NUM(long double, 512)

#ifndef TSTRING_IS_STD_STRING
template <>
void Out<TBasicCharRef<TString>>(IOutputStream& o, const TBasicCharRef<TString>& c) {
    o << static_cast<char>(c);
}

template <>
void Out<TBasicCharRef<TUtf16String>>(IOutputStream& o, const TBasicCharRef<TUtf16String>& c) {
    o << static_cast<wchar16>(c);
}

template <>
void Out<TBasicCharRef<TUtf32String>>(IOutputStream& o, const TBasicCharRef<TUtf32String>& c) {
    o << static_cast<wchar32>(c);
}
#endif

template <>
void Out<const void*>(IOutputStream& o, const void* t) {
    o << Hex(size_t(t));
}

template <>
void Out<void*>(IOutputStream& o, void* t) {
    Out<const void*>(o, t);
}

using TNullPtr = decltype(nullptr);

template <>
void Out<TNullPtr>(IOutputStream& o, TTypeTraits<TNullPtr>::TFuncParam) {
    o << AsStringBuf("nullptr");
}

#if defined(_android_)
namespace {
    class TAndroidStdIOStreams {
    public:
        TAndroidStdIOStreams()
            : LogLibrary("liblog.so")
            , LogFuncPtr((TLogFuncPtr)LogLibrary.Sym("__android_log_write"))
            , Out(LogFuncPtr)
            , Err(LogFuncPtr)
        {
        }

    public:
        using TLogFuncPtr = void (*)(int, const char*, const char*);

        class TAndroidStdOutput: public IOutputStream {
        public:
            inline TAndroidStdOutput(TLogFuncPtr logFuncPtr) noexcept
                : Buffer()
                , LogFuncPtr(logFuncPtr)
            {
            }

            virtual ~TAndroidStdOutput() {
            }

        private:
            virtual void DoWrite(const void* buf, size_t len) override {
                with_lock (BufferMutex) {
                    Buffer.Write(buf, len);
                }
            }

            virtual void DoFlush() override {
                with_lock (BufferMutex) {
                    LogFuncPtr(ANDROID_LOG_DEBUG, GetTag(), Buffer.Data());
                    Buffer.Clear();
                }
            }

            virtual const char* GetTag() const = 0;

        private:
            TMutex BufferMutex;
            TStringStream Buffer;
            TLogFuncPtr LogFuncPtr;
        };

        class TStdErr: public TAndroidStdOutput {
        public:
            TStdErr(TLogFuncPtr logFuncPtr)
                : TAndroidStdOutput(logFuncPtr)
            {
            }

            virtual ~TStdErr() {
            }

        private:
            virtual const char* GetTag() const override {
                return "stderr";
            }
        };

        class TStdOut: public TAndroidStdOutput {
        public:
            TStdOut(TLogFuncPtr logFuncPtr)
                : TAndroidStdOutput(logFuncPtr)
            {
            }

            virtual ~TStdOut() {
            }

        private:
            virtual const char* GetTag() const override {
                return "stdout";
            }
        };

        static bool Enabled;
        TDynamicLibrary LogLibrary; // field order is important, see constructor
        TLogFuncPtr LogFuncPtr;
        TStdOut Out;
        TStdErr Err;

        static inline TAndroidStdIOStreams& Instance() {
            return *SingletonWithPriority<TAndroidStdIOStreams, 4>();
        }
    };

    bool TAndroidStdIOStreams::Enabled = false;
}
#endif // _android_

namespace {
    class TStdOutput: public IOutputStream {
    public:
        inline TStdOutput(FILE* f) noexcept
            : F_(f)
        {
        }

        ~TStdOutput() override = default;

    private:
        void DoWrite(const void* buf, size_t len) override {
            if (len != fwrite(buf, 1, len, F_)) {
#if defined(_win_)
                // On Windows, if 'F_' is console -- 'fwrite' returns count of written characters.
                // If, for example, console output codepage is UTF-8, then returned value is
                // not equal to 'len'. So, we ignore some 'errno' values...
                if ((errno == 0 || errno == EINVAL || errno == EILSEQ) && _isatty(fileno(F_))) {
                    return;
                }
#endif
                ythrow TSystemError() << "write failed";
            }
        }

        void DoFlush() override {
            if (fflush(F_) != 0) {
                ythrow TSystemError() << "fflush failed";
            }
        }

    private:
        FILE* F_;
    };

    struct TStdIOStreams {
        struct TStdErr: public TStdOutput {
            inline TStdErr()
                : TStdOutput(stderr)
            {
            }

            ~TStdErr() override = default;
        };

        struct TStdOut: public TStdOutput {
            inline TStdOut()
                : TStdOutput(stdout)
            {
            }

            ~TStdOut() override = default;
        };

        TStdOut Out;
        TStdErr Err;

        static inline TStdIOStreams& Instance() {
            return *SingletonWithPriority<TStdIOStreams, 4>();
        }
    };
}

IOutputStream& NPrivate::StdErrStream() noexcept {
#if defined(_android_)
    if (TAndroidStdIOStreams::Enabled) {
        return TAndroidStdIOStreams::Instance().Err;
    }
#endif
    return TStdIOStreams::Instance().Err;
}

IOutputStream& NPrivate::StdOutStream() noexcept {
#if defined(_android_)
    if (TAndroidStdIOStreams::Enabled) {
        return TAndroidStdIOStreams::Instance().Out;
    }
#endif
    return TStdIOStreams::Instance().Out;
}

void RedirectStdioToAndroidLog(bool redirect) {
#if defined(_android_)
    TAndroidStdIOStreams::Enabled = redirect;
#else
    Y_UNUSED(redirect);
#endif
}
