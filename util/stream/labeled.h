#pragma once

#include <util/generic/va_args.h>

/**
 * Generates an output sequence for the provided expressions that is formatted
 * as a labeled comma-separated list.
 *
 * Example usage:
 * @code
 * int a = 1, b = 2, c = 3;
 * stream << LabeledOutput(a, b, c, a + b + c);
 * // Outputs "a = 1, b = 2, c = 3, a + b + c = 6"
 * @endcode
 */
#define LabeledOutput(...) "" Y_PASS_VA_ARGS(Y_MAP_ARGS_WITH_LAST(__LABELED_OUTPUT_NONLAST__, __LABELED_OUTPUT_IMPL__, __VA_ARGS__))

#define __LABELED_OUTPUT_IMPL__(x) << #x " = " << (x)
#define __LABELED_OUTPUT_NONLAST__(x) __LABELED_OUTPUT_IMPL__(x) << ", "
