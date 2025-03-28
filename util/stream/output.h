#pragma once

#include "fwd.h"
#include "labeled.h"

#include <util/generic/noncopyable.h>
#include <util/generic/string.h>
#include <util/generic/strbuf.h>
#include <util/generic/typetraits.h>
#include <util/system/compiler.h>

#include <type_traits>

/**
 * @addtogroup Streams_Base
 * @{
 */

/**
 * Abstract output stream.
 */
class IOutputStream: public TNonCopyable {
public:
    /**
     * Data block for output.
     */
    struct TPart {
        inline TPart(const void* Buf, size_t Len) noexcept
            : buf(Buf)
            , len(Len)
        {
        }

        inline TPart(const TStringBuf s) noexcept
            : buf(s.data())
            , len(s.size())
        {
        }

        inline TPart() noexcept
            : buf(nullptr)
            , len(0)
        {
        }

        inline ~TPart() = default;

        static inline TPart CrLf() noexcept {
            return TPart("\r\n", 2);
        }

        const void* buf;
        size_t len;
    };

    IOutputStream() noexcept;
    virtual ~IOutputStream();

    IOutputStream(IOutputStream&&) noexcept {
    }

    IOutputStream& operator=(IOutputStream&&) noexcept {
        return *this;
    }

    /**
     * Writes into this stream.
     *
     * @param buf                       Data to write.
     * @param len                       Number of bytes to write.
     */
    inline void Write(const void* buf, size_t len) {
        if (len) {
            DoWrite(buf, len);
        }
    }

    /**
     * Writes a string into this stream.
     *
     * @param st                        String to write.
     */
    inline void Write(const TStringBuf st) {
        Write(st.data(), st.size());
    }

    /**
     * Writes several data blocks into this stream.
     *
     * @param parts                     Pointer to the start of the data blocks
     *                                  array.
     * @param count                     Number of data blocks to write.
     */
    inline void Write(const TPart* parts, size_t count) {
        if (count > 1) {
            DoWriteV(parts, count);
        } else if (count) {
            DoWrite(parts->buf, parts->len);
        }
    }

    /**
     * Writes a single character into this stream.
     *
     * @param ch                        Character to write.
     */
    inline void Write(char ch) {
        DoWriteC(ch);
    }

    /**
     * Flushes this stream's buffer, if any.
     *
     * Note that this can also be done with a `Flush` manipulator:
     * @code
     * stream << "some string" << Flush;
     * @endcode
     */
    inline void Flush() {
        DoFlush();
    }

    /**
     * Flushes and closes this stream. No more data can be written into a stream
     * once it's closed.
     */
    inline void Finish() {
        DoFinish();
    }

protected:
    /**
     * Writes into this stream.
     *
     * @param buf                       Data to write.
     * @param len                       Number of bytes to write.
     * @throws yexception               If IO error occurs.
     */
    virtual void DoWrite(const void* buf, size_t len) = 0;

    /**
     * Writes several data blocks into this stream.
     *
     * @param parts                     Pointer to the start of the data blocks
     *                                  array.
     * @param count                     Number of data blocks to write.
     * @throws yexception               If IO error occurs.
     */
    virtual void DoWriteV(const TPart* parts, size_t count);

    /**
     * Writes a single character into this stream. Can be overridden with a faster implementation.
     *
     * @param ch                        Character to write.
     */
    virtual void DoWriteC(char ch);

    /**
     * Flushes this stream's buffer, if any.
     *
     * @throws yexception               If IO error occurs.
     */
    virtual void DoFlush();

    /**
     * Flushes and closes this stream. No more data can be written into a stream
     * once it's closed.
     *
     * @throws yexception               If IO error occurs.
     */
    virtual void DoFinish();
};

/**
 * `operator<<` for `IOutputStream` by default delegates to this function.
 *
 * Note that while `operator<<` uses overloading (and thus argument-dependent
 * lookup), `Out` uses template specializations. This makes it possible to
 * have a single `Out` declaration, and then just provide specializations in
 * cpp files, letting the linker figure everything else out. This approach
 * reduces compilation times.
 *
 * However, if the flexibility of overload resolution is needed, then one should
 * just overload `operator<<`.
 *
 * @param out                           Output stream to write into.
 * @param value                         Value to write.
 */
template <class T>
void Out(IOutputStream& out, typename TTypeTraits<T>::TFuncParam value);

#define Y_DECLARE_OUT_SPEC(MODIF, T, stream, value) \
    template <>                                     \
    MODIF void Out<T>(IOutputStream & stream, TTypeTraits<T>::TFuncParam value)

template <>
inline void Out<const char*>(IOutputStream& o, const char* t) {
    if (t) {
        o.Write(t);
    } else {
        o.Write("(null)");
    }
}

template <>
void Out<const wchar16*>(IOutputStream& o, const wchar16* w);

template <>
void Out<const wchar32*>(IOutputStream& o, const wchar32* w);

static inline IOutputStream& operator<<(IOutputStream& o Y_LIFETIME_BOUND, TStreamManipulator m) {
    m(o);

    return o;
}

static inline IOutputStream& operator<<(IOutputStream& o Y_LIFETIME_BOUND, const char* t) {
    Out<const char*>(o, t);

    return o;
}

static inline IOutputStream& operator<<(IOutputStream& o Y_LIFETIME_BOUND, char* t) {
    Out<const char*>(o, t);

    return o;
}

template <class T>
static inline std::enable_if_t<std::is_scalar<T>::value, IOutputStream&> operator<<(IOutputStream& o Y_LIFETIME_BOUND, T t) {
    Out<T>(o, t);

    return o;
}

template <class T>
static inline std::enable_if_t<!std::is_scalar<T>::value, IOutputStream&> operator<<(IOutputStream& o Y_LIFETIME_BOUND, const T& t) {
    Out<T>(o, t);

    return o;
}

static inline IOutputStream& operator<<(IOutputStream& o Y_LIFETIME_BOUND, const wchar16* t) {
    Out<const wchar16*>(o, t);
    return o;
}

static inline IOutputStream& operator<<(IOutputStream& o Y_LIFETIME_BOUND, wchar16* t) {
    Out<const wchar16*>(o, t);
    return o;
}

static inline IOutputStream& operator<<(IOutputStream& o Y_LIFETIME_BOUND, const wchar32* t) {
    Out<const wchar32*>(o, t);
    return o;
}

static inline IOutputStream& operator<<(IOutputStream& o Y_LIFETIME_BOUND, wchar32* t) {
    Out<const wchar32*>(o, t);
    return o;
}

namespace NPrivate {
    IOutputStream& StdOutStream() noexcept;
    IOutputStream& StdErrStream() noexcept;
    IOutputStream& StdDbgStream() noexcept;
} // namespace NPrivate

/**
 * Standard output stream.
 */
#define Cout (::NPrivate::StdOutStream())

/**
 * Standard error stream.
 */
#define Cerr (::NPrivate::StdErrStream())

/**
 * Standard log stream.
 */
#define Clog Cerr

/**
 * Standard debug stream.
 *
 * Behavior of this stream is controlled via `DBGOUT` environment variable.
 * If this variable is set, then this stream is redirected into `stderr`,
 * otherwise whatever is written into it is simply ignored.
 */
#define Cdbg (::NPrivate::StdDbgStream())

/**
 * End-of-line output manipulator, basically the same as `std::endl`.
 */
static inline void Endl(IOutputStream& o) {
    (o << '\n').Flush();
}

/**
 * Flushing stream manipulator, basically the same as `std::flush`.
 */
static inline void Flush(IOutputStream& o) {
    o.Flush();
}

/*
 * Also see format.h for additional manipulators.
 */

void RedirectStdioToAndroidLog(bool redirect);

/**
 * Debug output stream. Writes into `stderr`.
 */
class TDebugOutput: public IOutputStream {
public:
    inline TDebugOutput() noexcept = default;
    ~TDebugOutput() override = default;

    TDebugOutput(TDebugOutput&&) noexcept = default;
    TDebugOutput& operator=(TDebugOutput&&) noexcept = default;

private:
    void DoWrite(const void* buf, size_t len) override;
};

/**
 * This function returns the current debug level as set via `DBGOUT` environment
 * variable.
 *
 * Note that the proper way to use this function is via `Y_DBGTRACE` macro.
 * There are very few cases when there is a need to use it directly.
 *
 * @returns                             Debug level.
 * @see ETraceLevel
 * @see DBGTRACE
 */
int StdDbgLevel() noexcept;
