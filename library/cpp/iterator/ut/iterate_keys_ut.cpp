#include <library/cpp/iterator/iterate_keys.h>

#include <library/cpp/testing/gtest/gtest.h>

#include <map>

using namespace testing;

TEST(IterateKeys, ConstMappingIteration) {
    const std::map<int, int> squares{
        {1, 1},
        {2, 4},
        {3, 9},
    };
    EXPECT_THAT(
        IterateKeys(squares),
        ElementsAre(1, 2, 3)
    );
}

TEST(IterateKeys, ConstMultiMappingIteration) {
    const std::multimap<int, int> primesBelow{
        {2, 2},
        {5, 3},
        {5, 5},
        {11, 7},
        {11, 11},
        {23, 13},
        {23, 17},
        {23, 23},
    };

    EXPECT_THAT(
        IterateKeys(primesBelow),
        ElementsAre(2, 5, 5, 11, 11, 23, 23, 23)
    );
}
