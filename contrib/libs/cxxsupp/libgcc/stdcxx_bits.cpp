#include "stdcxx_bits.h"

#include <stdexcept>

namespace std {

void __throw_out_of_range_fmt(const char* __fmt, ...) {
    (void)__fmt;
    throw std::out_of_range("__throw_out_of_range_fmt");
}

}
