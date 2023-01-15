#pragma once

#include "debug.h"

/**
 * Debug level, as set via `DBGOUT` environment variable.
 */
enum ETraceLevel : ui8 {
    TRACE_ERR = 1,
    TRACE_WARN = 2,
    TRACE_NOTICE = 3,
    TRACE_INFO = 4,
    TRACE_DEBUG = 5,
    TRACE_DETAIL = 6,
    TRACE_VERBOSE = 7
};

#if !defined(NDEBUG) && !defined(Y_ENABLE_TRACE)
#define Y_ENABLE_TRACE
#endif

#ifdef Y_ENABLE_TRACE

/**
 * Writes the given data into standard debug stream if current debug level set
 * via `DBGOUT` environment variable permits it.
 *
 * Does nothing in release builds unless `Y_ENABLE_TRACE` is defined.
 *
 * Example usage:
 * @code
 * Y_DBGTRACE(DEBUG, "Advance from " << node1 << " to " << node2);
 * @endcode
 *
 * @param elevel                        Debug level of this trace command, e.g.
 *                                      `WARN` or `DEBUG`. Basically a suffix of
 *                                      one of the values of `ETraceLevel` enum.
 * @param args                          Argument chain to be written out into
 *                                      standard debug stream, joined with `<<`
 *                                      operator.
 * @see ETraceLevel
 */
#define Y_DBGTRACE(elevel, args) Y_DBGTRACE0(int(TRACE_##elevel), args)
#define Y_DBGTRACE0(level, args)            \
    do                                      \
        if ((level) <= StdDbgLevel()) {     \
            StdDbgStream() << args << Endl; \
        }                                   \
    while (false)

#else

#define Y_DBGTRACE(elevel, args) \
    do {                         \
    } while (false)
#define Y_DBGTRACE0(level, args) \
    do {                         \
    } while (false)

#endif
