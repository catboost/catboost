#include <library/cpp/case_insensitive_string/case_insensitive_string.h>

#include <library/cpp/digest/murmur/murmur.h>
#include <library/cpp/testing/gtest/gtest.h>

#include <util/generic/hash_table.h>
#include <util/random/random.h>

TEST(CaseInsensitiveString, Hash) {
    size_t h1 = ComputeHash(TCaseInsensitiveString("some long string..."));
    size_t h2 = ComputeHash(TCaseInsensitiveString("Some Long String..."));
    EXPECT_EQ(h1, h2);
    size_t otherHash = ComputeHash(TCaseInsensitiveString("other long string..."));
    EXPECT_NE(h1, otherHash);
}

namespace {
    size_t NaiveCaseInsensitiveHash(TCaseInsensitiveStringBuf str) noexcept {
        TMurmurHash2A<size_t> hash;
        for (size_t i = 0; i < str.size(); ++i) {
            char lower = std::tolower(str[i]);
            hash.Update(&lower, 1);
        }
        return hash.Value();
    }
}

TEST(CaseInsensitiveString, HashValues) {
    SetRandomSeed(123);
    for (size_t n = 0; n < 64; ++n) {
        TCaseInsensitiveString s;
        for (size_t i = 0; i < n; ++i) {
            s.push_back(RandomNumber<unsigned char>());
        }
        EXPECT_EQ(ComputeHash(s), NaiveCaseInsensitiveHash(s)) << "Hashes for \"" << s << "\" differ";
    }
}
