#pragma once

#include "input.h"

/**
 * @addtogroup Streams_Multi
 * @{
 */

/**
 * A proxy input stream that concatenates two slave streams into one.
 */
class TMultiInput: public TInputStream {
public:
    TMultiInput(TInputStream* f, TInputStream* s) noexcept;
    ~TMultiInput() override;

private:
    size_t DoRead(void* buf, size_t len) override;
    size_t DoSkip(size_t len) override;
    size_t DoReadTo(TString& st, char ch) override;

private:
    TInputStream* C_;
    TInputStream* N_;
};

/** @} */
