#pragma once

#include "output.h"

/**
 * @addtogroup Streams_Multi
 * @{
 */

/**
 * A proxy output stream that writes into two slave streams simultaneously.
 */
class TTeeOutput: public IOutputStream {
public:
    TTeeOutput(IOutputStream* l Y_LIFETIME_BOUND, IOutputStream* r Y_LIFETIME_BOUND) noexcept;
    ~TTeeOutput() override;

private:
    void DoWrite(const void* buf, size_t len) override;
    void DoFlush() override;
    void DoFinish() override;

private:
    IOutputStream* L_;
    IOutputStream* R_;
};

/** @} */
