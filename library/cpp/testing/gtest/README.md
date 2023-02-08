# Gtest support in Arcadia

Gtest wrapper that reports results exactly how Arcadia CI wants it.

How to use:

- use `GTEST` in your `ya.make`;
- include `gtest.h` from this library. Don't include `<gtest/gtest.h>` and `<gmock/gmock.h>` directly because then you'll not get our extensions, including pretty printers for util types;
- write tests and enjoy.
