#pragma once

#include <util/system/compat.h>

class IOutputStream;

/**
 * Stream-based `printf` function. Prints formatted data into the provided stream.
 * Works the same way as a standard C `printf`.
 *
 * @param out                           Stream to write into.
 * @param fmt                           Format string.
 * @param ...                           Additional arguments.
 */
size_t Y_PRINTF_FORMAT(2, 3) Printf(IOutputStream& out, const char* fmt, ...);

/**
 * Stream-based `vprintf` function. Prints formatted data from variable argument
 * list into the provided stream. Works the same way as a standard C `vprintf`.
 *
 * @param out                           Stream to write into.
 * @param fmt                           Format string.
 * @param params                        Additional arguments as a variable argument list.
 */
size_t Y_PRINTF_FORMAT(2, 0) Printf(IOutputStream& out, const char* fmt, va_list params);
