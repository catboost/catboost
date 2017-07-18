#pragma once

#include "output.h"

/**
 * @addtogroup Streams_Multi
 * @{
 */

/**
 * A proxy output stream that writes into two slave streams simultaneously.
 */
class TTeeOutput: public TOutputStream {
public:
    TTeeOutput(TOutputStream* l, TOutputStream* r) noexcept;
    ~TTeeOutput() override;

private:
    void DoWrite(const void* buf, size_t len) override;
    void DoFlush() override;
    void DoFinish() override;

private:
    TOutputStream* L_;
    TOutputStream* R_;
};

/** @} */
