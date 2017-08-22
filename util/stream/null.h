#pragma once

#include "zerocopy.h"
#include "output.h"

/**
 * @addtogroup Streams
 * @{
 */

/**
 * Null input stream. Does nothing, contains no data.
 */
class TNullInput: public IZeroCopyInput {
public:
    TNullInput() noexcept;
    ~TNullInput() override;

private:
    size_t DoRead(void* buf, size_t len) override;
    size_t DoSkip(size_t len) override;
    size_t DoNext(const void** ptr, size_t len) override;
};

/**
 * Null output stream. Just ignores whatever is written into it.
 */
class TNullOutput: public IOutputStream {
public:
    TNullOutput() noexcept;
    ~TNullOutput() override;

    TNullOutput(TNullOutput&&) noexcept = default;
    TNullOutput& operator=(TNullOutput&&) noexcept = default;

private:
    void DoWrite(const void* buf, size_t len) override;
};

/**
 * Null input-output stream.
 *
 * @see TNullInput
 * @see TNullOutput
 */
class TNullIO: public TNullInput, public TNullOutput {
public:
    TNullIO() noexcept;
    ~TNullIO() override;
};

namespace NPrivate {
    TNullIO& StdNullStream() noexcept;
}

/**
 * Standard null stream.
 */
#define Cnull (::NPrivate::StdNullStream())

/** @} */
