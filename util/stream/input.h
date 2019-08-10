#pragma once

#include <util/generic/fwd.h>
#include <util/generic/noncopyable.h>
#include <util/system/defaults.h>

class IOutputStream;

/**
 * @addtogroup Streams_Base
 * @{
 */

/**
 * Abstract input stream.
 */
class IInputStream: public TNonCopyable {
public:
    IInputStream() noexcept;
    virtual ~IInputStream();

    IInputStream(IInputStream&&) noexcept {
    }

    IInputStream& operator=(IInputStream&&) noexcept {
        return *this;
    }

    /**
     * Reads some data from the stream. Note that this function might read less
     * data than what was requested. Use `Load` function if you want to read as
     * much data as possible.
     *
     * @param buf                       Buffer to read into.
     * @param len                       Number of bytes to read.
     * @returns                         Number of bytes that were actually read.
     *                                  A return value of zero signals end of stream.
     */
    inline size_t Read(void* buf, size_t len) {
        if (len == 0) {
            return 0;
        }

        return DoRead(buf, len);
    }

    /**
     * Reads one character from the stream.
     *
     * @param[out] c                    Character to read.
     * @returns                         Whether the character was read.
     *                                  A return value of false signals the end
     *                                  of stream.
     */
    inline bool ReadChar(char& c) {
        return DoRead(&c, 1) > 0;
    }

    /**
     * Reads all characters from the stream until the given character is
     * encountered, and stores them into the given string. The character itself
     * is read from the stream, but not stored in the string.
     *
     * @param[out] st                   String to read into.
     * @param ch                        Character to stop at.
     * @returns                         Total number of characters read from the stream.
     *                                  A return value of zero signals end of stream.
     */
    inline size_t ReadTo(TString& st, char ch) {
        return DoReadTo(st, ch);
    }

    /**
     * Reads the requested amount of data from the stream. Unlike `Read`, this
     * function stops only when the requested amount of data is read, or when
     * end of stream is reached.
     *
     * @param buf                       Buffer to read into.
     * @param len                       Number of bytes to read.
     * @returns                         Number of bytes that were actually read.
     *                                  A return value different from `len`
     *                                  signals end of stream.
     */
    size_t Load(void* buf, size_t len);

    /**
     * Reads the requested amount of data from the stream, or fails with an
     * exception if unable to do so.
     *
     * @param buf                       Buffer to read into.
     * @param len                       Number of bytes to read.
     * @see Load
     */
    void LoadOrFail(void* buf, size_t len);

    /**
     * Reads all data from this stream and returns it as a string.
     *
     * @returns                         Contents of this stream as a string.
     */
    TString ReadAll();

    /**
     * Reads all data from this stream and writes it into a provided output
     * stream.
     *
     * @param out                       Output stream to use.
     * @returns                         Total number of characters read from the stream.
     */
    ui64 ReadAll(IOutputStream& out);

    /**
     * Reads all data from the stream until the first occurrence of '\n'. Also
     * handles Windows line breaks correctly.
     *
     * @returns                         Next line read from this stream,
     *                                  excluding the line terminator.
     * @throws yexception               If no data could be read from a stream
     *                                  because end of stream has already been
     *                                  reached.
     */
    TString ReadLine();

    /**
     * Reads all characters from the stream until the given character is
     * encountered and returns them as a string. The character itself is read
     * from the stream, but not stored in the string.
     *
     * @param ch                        Character to stop at.
     * @returns                         String containing all the characters read.
     * @throws yexception               If no data could be read from a stream
     *                                  because end of stream has already been
     *                                  reached.
     */
    TString ReadTo(char ch);

    /**
     * Reads all data from the stream until the first occurrence of '\n' and
     * stores it into provided string. Also handles Windows line breaks correctly.
     *
     * @param[out] st                   String to store read characters into,
     *                                  excluding the line terminator.
     * @returns                         Total number of characters read from the stream.
     *                                  A return value of zero signals end of stream.
     */
    size_t ReadLine(TString& st);

    /**
     * Reads UTF8 encoded characters from the stream the first occurrence of '\n',
     * converts them into wide ones, and stores into provided string. Also handles
     * Windows line breaks correctly.
     *
     * @param[out] w                    Wide string to store read characters into,
     *                                  excluding the line terminator.
     * @returns                         Total number of characters read from the stream.
     *                                  A return value of zero signals end of stream.
     */
    size_t ReadLine(TUtf16String& w);

    /**
     * Skips some data from the stream without reading / copying it. Note that
     * this function might skip less data than what was requested.
     *
     * @param len                       Number of bytes to skip.
     * @returns                         Number of bytes that were actually skipped.
     *                                  A return value of zero signals end of stream.
     */
    size_t Skip(size_t len);

protected:
    /**
     * Reads some data from the stream. Might read less data than what was
     * requested.
     *
     * @param buf                       Buffer to read into.
     * @param len                       Number of bytes to read.
     * @returns                         Number of bytes that were actually read.
     *                                  A return value of zero signals end of stream.
     * @throws yexception               If IO error occurs.
     */
    virtual size_t DoRead(void* buf, size_t len) = 0;

    /**
     * Skips some data from the stream. Might skip less data than what was
     * requested.
     *
     * @param len                       Number of bytes to skip.
     * @returns                         Number of bytes that were actually skipped.
     *                                  A return value of zero signals end of stream.
     * @throws yexception               If IO error occurs.
     */
    virtual size_t DoSkip(size_t len);

    /**
     * Reads all characters from the stream until the given character is
     * encountered, and stores them into the given string. The character itself
     * is read from the stream, but not stored in the string.
     *
     * Provided string is cleared only if there is data in the stream.
     *
     * @param[out] st                   String to read into.
     * @param ch                        Character to stop at.
     * @returns                         Total number of characters read from the stream.
     *                                  A return value of zero signals end of stream.
     * @throws yexception               If IO error occurs.
     */
    virtual size_t DoReadTo(TString& st, char ch);

    /**
     * Reads all data from this stream and writes it into a provided output
     * stream.
     *
     * @param out                       Output stream to use.
     * @returns                         Total number of characters read from
     *                                  this stream.
     * @throws yexception               If IO error occurs.
     */
    virtual ui64 DoReadAll(IOutputStream& out);
};

/**
 * Transfers all data from the given input stream into the given output stream.
 *
 * @param in                            Input stream.
 * @param out                           Output stream.
 */
ui64 TransferData(IInputStream* in, IOutputStream* out);

/**
 * `operator>>` for `IInputStream` by default delegates to this function.
 *
 * Note that while `operator>>` uses overloading (and thus argument-dependent
 * lookup), `In` uses template specializations. This makes it possible to
 * have a single `In` declaration, and then just provide specializations in
 * cpp files, letting the linker figure everything else out. This approach
 * reduces compilation times.
 *
 * However, if the flexibility of overload resolution is needed, then one should
 * just overload `operator>>`.
 *
 * @param in                            Input stream to read from.
 * @param[out] value                    Value to read.
 * @throws                              `yexception` on invalid input or end of stream.
 * @see Out(IOutputStream&, T&)
 */
template <typename T>
void In(IInputStream& in, T& value);

/**
 * Reads a value from the stream.
 *
 * @param in                            Input stream to read from.
 * @param[out] value                    Value to read.
 * @returns                             Input stream.
 * @throws                              `yexception` on invalid input or end of stream.
 * @see operator<<(IOutputStream&, T&)
 */
template <typename T>
inline IInputStream& operator>>(IInputStream& in, T& value) {
    In<T>(in, value);
    return in;
}

namespace NPrivate {
    IInputStream& StdInStream() noexcept;
}

/**
 * Standard input stream.
 */
#define Cin (::NPrivate::StdInStream())

/** @} */
