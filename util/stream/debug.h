#pragma once

#include "output.h"

/**
 * @addtogroup Streams
 * @{
 */

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
 * @returns                             Standard debug stream.
 * @see Cdbg
 */
IOutputStream& StdDbgStream() noexcept;

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

/**
 * Standard debug stream.
 *
 * Behavior of this stream is controlled via `DBGOUT` environment variable.
 * If this variable is set, then this stream is redirected into `stderr`,
 * otherwise whatever is written into it is simply ignored.
 */
#define Cdbg (StdDbgStream())

/** @} */
