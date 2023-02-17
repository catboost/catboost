#pragma once

extern "C" {
    void __cxa_throw_bad_array_length() __attribute__((weak, noreturn));
    void __cxa_throw_bad_array_new_length() __attribute__((weak, noreturn));
}
