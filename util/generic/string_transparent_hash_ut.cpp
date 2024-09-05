#include "string.h"
#include "vector.h"
#include "strbuf.h"

#include <library/cpp/testing/unittest/registar.h>

#include <util/str_stl.h>

#ifdef __cpp_lib_generic_unordered_lookup
    #include <unordered_set>

template <class T, class THasher, class TPred>
using THashSetType = std::unordered_set<T, THasher, TPred>;
#else
    // Using Abseil hash set because `std::unordered_set` is transparent only from libstdc++11.
    // Meanwhile clang-linux-x86_64-release-stl-system autocheck sets OS_SDK=ubuntu-20,
    // that support libstdc++10 by default.
    #include <library/cpp/containers/absl_flat_hash/flat_hash_set.h>

template <class T, class THasher, class TPred>
using THashSetType = absl::flat_hash_set<T, THasher, TPred>;
#endif

Y_UNIT_TEST_SUITE(StringHashFunctorTests) {
    Y_UNIT_TEST(TestTransparencyWithUnorderedSet) {
        THashSetType<TString, THash<TString>, TEqualTo<TString>> s = {"foo"};
        // If either `THash` or `TEqualTo` is not transparent compilation will fail.
        UNIT_ASSERT_UNEQUAL(s.find(TStringBuf("foo")), s.end());
        UNIT_ASSERT_EQUAL(s.find(TStringBuf("bar")), s.end());
    }
} // Y_UNIT_TEST_SUITE(StringHashFunctorTests)
