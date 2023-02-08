# Extensions for Gtest and Gmock

Extensions that enable better support of util types in gtest and gmock: pretty printers, matchers, some convenience macros.

If you're using `GTEST`, include `library/cpp/testing/gtest/gtest.h` and it will automatically enable these extensions. This is the preferred way to include gtest and gmock as opposed to including gtest, gmock and extensions directly. It eliminates chances of forgetting to include extensions.
