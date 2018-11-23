#pragma once

#include "input.h"

/**
 * @addtogroup Streams_Multi
 * @{
 */

/**
 * A proxy input stream that concatenates two slave streams into one.
 */
class TMultiInput: public IInputStream {
public:
    TMultiInput(IInputStream* f, IInputStream* s) noexcept;
    ~TMultiInput() override;

private:
    size_t DoRead(void* buf, size_t len) override;
    size_t DoSkip(size_t len) override;
    size_t DoReadTo(TString& st, char ch) override;

private:
    IInputStream* C_;
    IInputStream* N_;
};

/**
 * See also "util/stream/tee.h" for multi output.
 */

/** @} */
