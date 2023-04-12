#pragma once

#include <util/generic/string.h>

#include <gtest/gtest.h>
#include <gmock/gmock.h>

/**
 * Check that the given statement throws an exception of the given type,
 * and that the thrown exception message contains the given substring.
 */
#define EXPECT_THROW_MESSAGE_HAS_SUBSTR(statement, expectedException, substring) \
    _Y_GTEST_EXPECT_THROW_MESSAGE_HAS_SUBSTR_IMPL_(statement, expectedException, substring, GTEST_NONFATAL_FAILURE_)

/**
 * Check that the given statement throws an exception of the given type,
 * and that the thrown exception message contains the given substring.
 */
#define ASSERT_THROW_MESSAGE_HAS_SUBSTR(statement, expectedException, substring) \
    _Y_GTEST_EXPECT_THROW_MESSAGE_HAS_SUBSTR_IMPL_(statement, expectedException, substring, GTEST_FATAL_FAILURE_)


// Improve default macros. New implementation shows better exception messages.
// See https://github.com/google/googletest/issues/2878

#undef EXPECT_THROW
#define EXPECT_THROW(statement, expectedException) \
    _Y_GTEST_EXPECT_THROW_IMPL_(statement, expectedException, GTEST_NONFATAL_FAILURE_)

#undef ASSERT_THROW
#define ASSERT_THROW(statement, expectedException) \
    _Y_GTEST_EXPECT_THROW_IMPL_(statement, expectedException, GTEST_FATAL_FAILURE_)

#undef EXPECT_NO_THROW
#define EXPECT_NO_THROW(statement) \
    _Y_GTEST_EXPECT_NO_THROW_IMPL_(statement, GTEST_NONFATAL_FAILURE_)

#undef ASSERT_NO_THROW
#define ASSERT_NO_THROW(statement) \
    _Y_GTEST_EXPECT_NO_THROW_IMPL_(statement, GTEST_FATAL_FAILURE_)


// Implementation details

namespace NGTest::NInternal {
    TString FormatErrorWrongException(const char* statement, const char* type);
    TString FormatErrorWrongException(const char* statement, const char* type, TString contains);
    TString FormatErrorUnexpectedException(const char* statement);
    bool ExceptionMessageContains(const std::exception& err, TString contains);
}

#define _Y_GTEST_EXPECT_THROW_IMPL_(statement, expectedException, fail)                                             \
    GTEST_AMBIGUOUS_ELSE_BLOCKER_                                                                                   \
    if (::TString gtestMsg = ""; ::testing::internal::AlwaysTrue()) {                                               \
        bool gtestCaughtExpected = false;                                                                           \
        try {                                                                                                       \
            GTEST_SUPPRESS_UNREACHABLE_CODE_WARNING_BELOW_(statement);                                              \
        } catch (expectedException const&) {                                                                        \
            gtestCaughtExpected = true;                                                                             \
        } catch (...) {                                                                                             \
            gtestMsg = ::NGTest::NInternal::FormatErrorWrongException(                                              \
                #statement, #expectedException);                                                                    \
            goto GTEST_CONCAT_TOKEN_(gtest_label_testthrow_, __LINE__);                                             \
        } if (!gtestCaughtExpected) {                                                                               \
            gtestMsg = ::NGTest::NInternal::FormatErrorWrongException(                                              \
                #statement, #expectedException);                                                                    \
            goto GTEST_CONCAT_TOKEN_(gtest_label_testthrow_, __LINE__);                                             \
        }                                                                                                           \
    } else                                                                                                          \
        GTEST_CONCAT_TOKEN_(gtest_label_testthrow_, __LINE__):                                                      \
            fail(gtestMsg.c_str())

#define _Y_GTEST_EXPECT_THROW_MESSAGE_HAS_SUBSTR_IMPL_(statement, expectedException, substring, fail)               \
    GTEST_AMBIGUOUS_ELSE_BLOCKER_                                                                                   \
    if (::TString gtestMsg = ""; ::testing::internal::AlwaysTrue()) {                                               \
        bool gtestCaughtExpected = false;                                                                           \
        ::TString gtestSubstring{substring};                                                                        \
        try {                                                                                                       \
            GTEST_SUPPRESS_UNREACHABLE_CODE_WARNING_BELOW_(statement);                                              \
        } catch (expectedException const& gtestError) {                                                             \
            if (!::NGTest::NInternal::ExceptionMessageContains(gtestError, gtestSubstring)) {                       \
                gtestMsg = ::NGTest::NInternal::FormatErrorWrongException(                                          \
                    #statement, #expectedException, gtestSubstring);                                                \
                goto GTEST_CONCAT_TOKEN_(gtest_label_testthrowsubstr_, __LINE__);                                   \
            }                                                                                                       \
            gtestCaughtExpected = true;                                                                             \
        } catch (...) {                                                                                             \
            gtestMsg = ::NGTest::NInternal::FormatErrorWrongException(                                              \
                #statement, #expectedException, gtestSubstring);                                                    \
            goto GTEST_CONCAT_TOKEN_(gtest_label_testthrowsubstr_, __LINE__);                                       \
        } if (!gtestCaughtExpected) {                                                                               \
            gtestMsg = ::NGTest::NInternal::FormatErrorWrongException(                                              \
                #statement, #expectedException, gtestSubstring);                                                    \
            goto GTEST_CONCAT_TOKEN_(gtest_label_testthrowsubstr_, __LINE__);                                       \
        }                                                                                                           \
    } else                                                                                                          \
        GTEST_CONCAT_TOKEN_(gtest_label_testthrowsubstr_, __LINE__):                                                \
            fail(gtestMsg.c_str())

#define _Y_GTEST_EXPECT_NO_THROW_IMPL_(statement, fail)                                                             \
    GTEST_AMBIGUOUS_ELSE_BLOCKER_                                                                                   \
    if (::TString gtestMsg = ""; ::testing::internal::AlwaysTrue()) {                                               \
        try {                                                                                                       \
            GTEST_SUPPRESS_UNREACHABLE_CODE_WARNING_BELOW_(statement);                                              \
        } catch (...) {                                                                                             \
            gtestMsg = ::NGTest::NInternal::FormatErrorUnexpectedException(#statement);                             \
            goto GTEST_CONCAT_TOKEN_(gtest_label_testnothrow_, __LINE__);                                           \
        }                                                                                                           \
    } else                                                                                                          \
        GTEST_CONCAT_TOKEN_(gtest_label_testnothrow_, __LINE__):                                                    \
            fail(gtestMsg.c_str())
