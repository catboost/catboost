#pragma once

#include "fwd.h"
#include "input.h"
#include "output.h"
#include "buffered.h"
#include "mem.h"

#include <util/system/file.h>
#include <utility>

/**
 * @addtogroup Streams_Files
 * @{
 */

/**
 * Unbuffered file input stream.
 *
 * Note that the input is not buffered, which means that `ReadLine` calls will
 * be _very_ slow.
 */
class TUnbufferedFileInput: public IInputStream {
public:
    TUnbufferedFileInput(const char* path);
    TUnbufferedFileInput(const TString& path);
    TUnbufferedFileInput(const std::filesystem::path& path);
    TUnbufferedFileInput(const TFile& file);

private:
    static constexpr EOpenMode OPEN_MODE = OpenExisting | RdOnly | Seq;

    size_t DoRead(void* buf, size_t len) override;
    size_t DoSkip(size_t len) override;

private:
    TFile File_;
};

/**
 * Memory-mapped file input stream.
 */
class TMappedFileInput: public TMemoryInput {
public:
    TMappedFileInput(const TFile& file);
    TMappedFileInput(const TString& path);
    ~TMappedFileInput() override;

private:
    class TImpl;
    THolder<TImpl> Impl_;
};

/**
 * File output stream.
 *
 * Note that the output is unbuffered, thus writing in many small chunks is
 * likely to be quite slow.
 */
class TUnbufferedFileOutput: public IOutputStream {
public:
    TUnbufferedFileOutput(const char* path);
    TUnbufferedFileOutput(const TString& path);
    TUnbufferedFileOutput(const std::filesystem::path& path);
    TUnbufferedFileOutput(const TFile& file);
    ~TUnbufferedFileOutput() override;

    TUnbufferedFileOutput(TUnbufferedFileOutput&&) noexcept = default;
    TUnbufferedFileOutput& operator=(TUnbufferedFileOutput&&) noexcept = default;

private:
    static constexpr EOpenMode OPEN_MODE = CreateAlways | WrOnly | Seq;

    void DoWrite(const void* buf, size_t len) override;
    void DoFlush() override;

private:
    TFile File_;
};

/**
 * Buffered file input stream.
 *
 * @see TBuffered
 */
class TFileInput: public TBuffered<TUnbufferedFileInput> {
public:
    template <class T>
    inline TFileInput(T&& t, size_t buf = 1 << 13)
        : TBuffered<TUnbufferedFileInput>(buf, std::forward<T>(t))
    {
    }

    ~TFileInput() override = default;
};

/**
 * Buffered file output stream.
 *
 * Currently deprecated, please use TFileOutput in new code.
 *
 * @deprecated
 * @see TBuffered
 */
class TFixedBufferFileOutput: public TBuffered<TUnbufferedFileOutput> {
public:
    template <class T>
    inline TFixedBufferFileOutput(T&& t, size_t buf = 1 << 13)
        : TBuffered<TUnbufferedFileOutput>(buf, std::forward<T>(t))
    {
    }

    ~TFixedBufferFileOutput() override = default;
};

/** @} */
