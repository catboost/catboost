#include "string.h"
#include "vector.h"
#include "strbuf.h"

#include <library/cpp/unittest/registar.h>
#include <library/cpp/containers/absl_flat_hash/flat_hash_set.h>

#include <util/str_stl.h>

Y_UNIT_TEST_SUITE(StringHashFunctorTests) {
    Y_UNIT_TEST(TestTransparencyWithUnorderedSet) {
        // Using Abseil hash set because `std::unordered_set` is transparent only from C++20 (while
        // we stuck with C++17 right now).
        absl::flat_hash_set<TString, THash<TString>, TEqualTo<TString>> s = {"foo"};
        // If either `THash` or `TEqualTo` is not transparent compilation will fail.
        UNIT_ASSERT_UNEQUAL(s.find(TStringBuf("foo")), s.end());
        UNIT_ASSERT_EQUAL(s.find(TStringBuf("bar")), s.end());
    }
}
