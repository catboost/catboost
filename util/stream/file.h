#pragma once

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
 * File input stream.
 *
 * Note that the input is not buffered, which means that `ReadLine` calls will
 * be _very_ slow.
 */
class TFileInput: public IInputStream {
public:
    TFileInput(const TFile& file);
    TFileInput(const TString& path);

private:
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
class TFileOutput: public IOutputStream {
public:
    TFileOutput(const TString& path);
    TFileOutput(const TFile& file);
    ~TFileOutput() override;

    TFileOutput(TFileOutput&&) noexcept = default;
    TFileOutput& operator=(TFileOutput&&) noexcept = default;

private:
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
class TBufferedFileInput: public TBuffered<TFileInput> {
public:
    template <class T>
    inline TBufferedFileInput(T&& t, size_t buf = 1 << 13)
        : TBuffered<TFileInput>(buf, std::forward<T>(t))
    {
    }

    ~TBufferedFileInput() override = default;
};

using TIFStream = TBufferedFileInput;

/**
 * Buffered file output stream.
 *
 * Currently deprecated, please use TAdaptiveFileOutput in new code.
 *
 * @deprecated
 * @see TBuffered
 */
class TBufferedFileOutput: public TBuffered<TFileOutput> {
public:
    template <class T>
    inline TBufferedFileOutput(T&& t, size_t buf = 1 << 13)
        : TBuffered<TFileOutput>(buf, std::forward<T>(t))
    {
    }

    ~TBufferedFileOutput() override = default;
};

using TOFStream = TBufferedFileOutput;

/**
 * Adaptively buffered file output stream.
 *
 * @see TAdaptivelyBuffered
 */
using TAdaptiveFileOutput = TAdaptivelyBuffered<TFileOutput>;

/** @} */
