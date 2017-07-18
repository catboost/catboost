#pragma once

namespace std {

void __throw_out_of_range_fmt(const char*, ...)
    __attribute__((__noreturn__))
    __attribute__((__format__(__printf__, 1, 2)))
    __attribute__((__weak__))
;

}
