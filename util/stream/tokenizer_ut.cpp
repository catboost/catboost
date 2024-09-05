#include <library/cpp/testing/unittest/registar.h>

#include <util/generic/array_size.h>
#include <util/generic/strbuf.h>

#include "mem.h"
#include "null.h"
#include "tokenizer.h"

static inline void CheckIfNullTerminated(const TStringBuf str) {
    UNIT_ASSERT_VALUES_EQUAL('\0', *(str.data() + str.size()));
}

Y_UNIT_TEST_SUITE(TStreamTokenizerTests) {
    Y_UNIT_TEST(EmptyStreamTest) {
        auto&& input = TNullInput{};
        auto&& tokenizer = TStreamTokenizer<TEol>{&input};
        auto tokensCount = size_t{};
        for (auto it = tokenizer.begin(); tokenizer.end() != it; ++it) {
            CheckIfNullTerminated(TStringBuf{it->Data(), it->Length()});
            ++tokensCount;
        }
        UNIT_ASSERT_VALUES_EQUAL(0, tokensCount);
    }

    Y_UNIT_TEST(EmptyTokensTest) {
        const char data[] = "\n\n";
        const auto dataSize = Y_ARRAY_SIZE(data) - 1;
        auto&& input = TMemoryInput{data, dataSize};
        auto&& tokenizer = TStreamTokenizer<TEol>{&input};
        auto tokensCount = size_t{};
        for (auto it = tokenizer.begin(); tokenizer.end() != it; ++it) {
            CheckIfNullTerminated(TStringBuf{it->Data(), it->Length()});
            UNIT_ASSERT_VALUES_EQUAL(0, it->Length());
            ++tokensCount;
        }
        UNIT_ASSERT_VALUES_EQUAL(2, tokensCount);
    }

    Y_UNIT_TEST(LastTokenendDoesntSatisfyPredicateTest) {
        const char data[] = "abc\ndef\nxxxxxx";
        const auto dataSize = Y_ARRAY_SIZE(data) - 1;
        const TStringBuf tokens[] = {TStringBuf("abc"), TStringBuf("def"), TStringBuf("xxxxxx")};
        const auto tokensSize = Y_ARRAY_SIZE(tokens);
        auto&& input = TMemoryInput{data, dataSize};
        auto&& tokenizer = TStreamTokenizer<TEol>{&input};
        auto tokensCount = size_t{};
        for (auto it = tokenizer.begin(); tokenizer.end() != it; ++it) {
            UNIT_ASSERT(tokensCount < tokensSize);
            const auto token = TStringBuf{it->Data(), it->Length()};
            CheckIfNullTerminated(token);
            UNIT_ASSERT_VALUES_EQUAL(tokens[tokensCount], token);
            ++tokensCount;
        }
        UNIT_ASSERT_VALUES_EQUAL(tokensSize, tokensCount);
    }

    Y_UNIT_TEST(FirstTokenIsEmptyTest) {
        const char data[] = "\ndef\nxxxxxx";
        const auto dataSize = Y_ARRAY_SIZE(data) - 1;
        const TStringBuf tokens[] = {TStringBuf(), TStringBuf("def"), TStringBuf("xxxxxx")};
        const auto tokensSize = Y_ARRAY_SIZE(tokens);
        auto&& input = TMemoryInput{data, dataSize};
        auto&& tokenizer = TStreamTokenizer<TEol>{&input};
        auto tokensCount = size_t{};
        for (auto it = tokenizer.begin(); tokenizer.end() != it; ++it) {
            UNIT_ASSERT(tokensCount < tokensSize);
            const auto token = TStringBuf{it->Data(), it->Length()};
            CheckIfNullTerminated(token);
            UNIT_ASSERT_VALUES_EQUAL(tokens[tokensCount], token);
            ++tokensCount;
        }
        UNIT_ASSERT_VALUES_EQUAL(tokensSize, tokensCount);
    }

    Y_UNIT_TEST(PredicateDoesntMatch) {
        const char data[] = "1234567890-=!@#$%^&*()_+QWERTYUIOP{}qwertyuiop[]ASDFGHJKL:";
        const auto dataSize = Y_ARRAY_SIZE(data) - 1;
        auto&& input = TMemoryInput{data, dataSize};
        auto&& tokenizer = TStreamTokenizer<TEol>{&input};
        auto tokensCount = size_t{};
        for (auto it = tokenizer.begin(); tokenizer.end() != it; ++it) {
            const auto token = TStringBuf{it->Data(), it->Length()};
            CheckIfNullTerminated(token);
            UNIT_ASSERT_VALUES_EQUAL(data, token);
            ++tokensCount;
        }
        UNIT_ASSERT_VALUES_EQUAL(1, tokensCount);
    }

    Y_UNIT_TEST(SimpleTest) {
        const char data[] = "qwerty\n1234567890\n";
        const auto dataSize = Y_ARRAY_SIZE(data) - 1;
        const TStringBuf tokens[] = {TStringBuf("qwerty"), TStringBuf("1234567890")};
        const auto tokensSize = Y_ARRAY_SIZE(tokens);
        auto&& input = TMemoryInput{data, dataSize};
        auto&& tokenizer = TStreamTokenizer<TEol>{&input};
        auto tokensCount = size_t{};
        for (auto it = tokenizer.begin(); tokenizer.end() != it; ++it) {
            UNIT_ASSERT(tokensCount < tokensSize);
            const auto token = TStringBuf{it->Data(), it->Length()};
            CheckIfNullTerminated(token);
            UNIT_ASSERT_VALUES_EQUAL(tokens[tokensCount], token);
            ++tokensCount;
        }
        UNIT_ASSERT_VALUES_EQUAL(tokensSize, tokensCount);
    }

    Y_UNIT_TEST(CustomPredicateTest) {
        struct TIsVerticalBar {
            inline bool operator()(const char ch) const noexcept {
                return '|' == ch;
            }
        };

        const char data[] = "abc|def|xxxxxx";
        const auto dataSize = Y_ARRAY_SIZE(data) - 1;
        const TStringBuf tokens[] = {TStringBuf("abc"), TStringBuf("def"), TStringBuf("xxxxxx")};
        const auto tokensSize = Y_ARRAY_SIZE(tokens);
        auto&& input = TMemoryInput{data, dataSize};
        auto&& tokenizer = TStreamTokenizer<TIsVerticalBar>{&input};
        auto tokensCount = size_t{};
        for (auto it = tokenizer.begin(); tokenizer.end() != it; ++it) {
            UNIT_ASSERT(tokensCount < tokensSize);
            const auto token = TStringBuf{it->Data(), it->Length()};
            CheckIfNullTerminated(token);
            UNIT_ASSERT_VALUES_EQUAL(tokens[tokensCount], token);
            ++tokensCount;
        }
        UNIT_ASSERT_VALUES_EQUAL(tokensSize, tokensCount);
    }

    Y_UNIT_TEST(CustomPredicateSecondTest) {
        struct TIsVerticalBar {
            inline bool operator()(const char ch) const noexcept {
                return '|' == ch || ',' == ch;
            }
        };

        const char data[] = "abc|def|xxxxxx,abc|def|xxxxxx";
        const auto dataSize = Y_ARRAY_SIZE(data) - 1;
        const TStringBuf tokens[] = {TStringBuf("abc"), TStringBuf("def"), TStringBuf("xxxxxx"),
                                     TStringBuf("abc"), TStringBuf("def"), TStringBuf("xxxxxx")};
        const auto tokensSize = Y_ARRAY_SIZE(tokens);
        auto&& input = TMemoryInput{data, dataSize};
        auto&& tokenizer = TStreamTokenizer<TIsVerticalBar>{&input};
        auto tokensCount = size_t{};
        for (auto it = tokenizer.begin(); tokenizer.end() != it; ++it) {
            UNIT_ASSERT(tokensCount < tokensSize);
            const auto token = TStringBuf{it->Data(), it->Length()};
            CheckIfNullTerminated(token);
            UNIT_ASSERT_VALUES_EQUAL(tokens[tokensCount], token);
            ++tokensCount;
        }
        UNIT_ASSERT_VALUES_EQUAL(tokensSize, tokensCount);
    }

    Y_UNIT_TEST(FalsePredicateTest) {
        struct TAlwaysFalse {
            inline bool operator()(const char) const noexcept {
                return false;
            }
        };

        const char data[] = "1234567890-=!@#$%^&*()_+QWERTYUIOP{}qwertyuiop[]ASDFGHJKL:";
        const auto dataSize = Y_ARRAY_SIZE(data) - 1;
        auto&& input = TMemoryInput{data, dataSize};
        auto&& tokenizer = TStreamTokenizer<TAlwaysFalse>{&input};
        auto tokensCount = size_t{};
        for (auto it = tokenizer.begin(); tokenizer.end() != it; ++it) {
            const auto token = TStringBuf{it->Data(), it->Length()};
            CheckIfNullTerminated(token);
            UNIT_ASSERT_VALUES_EQUAL(data, token);
            ++tokensCount;
        }
        UNIT_ASSERT_VALUES_EQUAL(1, tokensCount);
    }

    Y_UNIT_TEST(TruePredicateTest) {
        struct TAlwaysTrue {
            inline bool operator()(const char) const noexcept {
                return true;
            }
        };

        const char data[] = "1234567890-=!@#$%^&*()_+QWERTYUIOP{}qwertyuiop[]ASDFGHJKL:";
        const auto dataSize = Y_ARRAY_SIZE(data) - 1;
        auto&& input = TMemoryInput{data, dataSize};
        auto&& tokenizer = TStreamTokenizer<TAlwaysTrue>{&input};
        auto tokensCount = size_t{};
        for (auto it = tokenizer.begin(); tokenizer.end() != it; ++it) {
            CheckIfNullTerminated(TStringBuf{it->Data(), it->Length()});
            UNIT_ASSERT_VALUES_EQUAL(0, it->Length());
            ++tokensCount;
        }
        UNIT_ASSERT_VALUES_EQUAL(dataSize, tokensCount);
    }

    Y_UNIT_TEST(FirstTokenHasSizeOfTheBufferTest) {
        const char data[] = "xxxxx\nxx";
        const auto dataSize = Y_ARRAY_SIZE(data) - 1;
        const TStringBuf tokens[] = {TStringBuf("xxxxx"), TStringBuf("xx")};
        const auto tokensSize = Y_ARRAY_SIZE(tokens);
        auto&& input = TMemoryInput{data, dataSize};
        auto&& tokenizer = TStreamTokenizer<TEol>{&input, TEol{}, tokens[0].size()};
        auto tokensCount = size_t{};
        for (auto it = tokenizer.begin(); tokenizer.end() != it; ++it) {
            const auto token = TStringBuf{it->Data(), it->Length()};
            CheckIfNullTerminated(token);
            UNIT_ASSERT_VALUES_EQUAL(tokens[tokensCount], token);
            ++tokensCount;
        }
        UNIT_ASSERT_VALUES_EQUAL(tokensSize, tokensCount);
    }

    Y_UNIT_TEST(OnlyTokenHasSizeOfTheBufferTest) {
        const char data[] = "xxxxx";
        const auto dataSize = Y_ARRAY_SIZE(data) - 1;
        auto&& input = TMemoryInput{data, dataSize};
        auto&& tokenizer = TStreamTokenizer<TEol>{&input, TEol{}, dataSize};
        auto tokensCount = size_t{};
        for (auto it = tokenizer.begin(); tokenizer.end() != it; ++it) {
            const auto token = TStringBuf{it->Data(), it->Length()};
            CheckIfNullTerminated(token);
            UNIT_ASSERT_VALUES_EQUAL(data, token);
            ++tokensCount;
        }
        UNIT_ASSERT_VALUES_EQUAL(1, tokensCount);
    }

    Y_UNIT_TEST(BufferSizeInitialSizeSmallerThanTokenTest) {
        const char data[] = "xxxxx\nxx";
        const auto dataSize = Y_ARRAY_SIZE(data) - 1;
        const TStringBuf tokens[] = {TStringBuf("xxxxx"), TStringBuf("xx")};
        const auto tokensSize = Y_ARRAY_SIZE(tokens);
        auto&& input = TMemoryInput{data, dataSize};
        auto&& tokenizer = TStreamTokenizer<TEol>{&input, TEol{}, 1};
        auto tokensCount = size_t{};
        for (auto it = tokenizer.begin(); tokenizer.end() != it; ++it) {
            const auto token = TStringBuf{it->Data(), it->Length()};
            CheckIfNullTerminated(token);
            UNIT_ASSERT_VALUES_EQUAL(tokens[tokensCount], token);
            ++tokensCount;
        }
        UNIT_ASSERT_VALUES_EQUAL(tokensSize, tokensCount);
    }

    Y_UNIT_TEST(RangeBasedForTest) {
        const char data[] = "abc\ndef\nxxxxxx";
        const auto dataSize = Y_ARRAY_SIZE(data) - 1;
        const TStringBuf tokens[] = {TStringBuf("abc"), TStringBuf("def"), TStringBuf("xxxxxx")};
        const auto tokensSize = Y_ARRAY_SIZE(tokens);
        auto&& input = TMemoryInput{data, dataSize};
        auto&& tokenizer = TStreamTokenizer<TEol>{&input};
        auto tokensCount = size_t{};
        for (const auto& token : tokenizer) {
            UNIT_ASSERT(tokensCount < tokensSize);
            CheckIfNullTerminated(token);
            UNIT_ASSERT_VALUES_EQUAL(tokens[tokensCount], token);
            ++tokensCount;
        }
        UNIT_ASSERT_VALUES_EQUAL(tokensSize, tokensCount);
    }
} // Y_UNIT_TEST_SUITE(TStreamTokenizerTests)
