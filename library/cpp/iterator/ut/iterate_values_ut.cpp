#include <library/cpp/iterator/iterate_values.h>

#include <library/cpp/testing/gtest/gtest.h>

#include <util/generic/algorithm.h>

#include <map>
#include <unordered_map>

using namespace testing;

TEST(IterateValues, ConstMappingIteration) {
    const std::map<int, int> squares{
        {1, 1},
        {2, 4},
        {3, 9},
    };
    EXPECT_THAT(
        IterateValues(squares),
        ElementsAre(1, 4, 9)
    );

    const std::unordered_map<int, int> roots{
        {49, 7},
        {36, 6},
        {25, 5},
    };
    EXPECT_THAT(
        IterateValues(roots),
        UnorderedElementsAre(5, 6, 7)
    );

    const std::map<int, std::string> translations{
        {1, "one"},
        {2, "two"},
        {3, "three"},
    };
    EXPECT_EQ(
        Accumulate(IterateValues(translations), std::string{}),
        "onetwothree"
    );
}

TEST(IterateValues, NonConstMappingIteration) {
    std::map<int, int> squares{
        {1, 1},
        {2, 4},
        {3, 9},
    };
    for (auto& value: IterateValues(squares)) {
        value *= value;
    }
    EXPECT_THAT(
        IterateValues(squares),
        ElementsAre(1, 16, 81)
    );
}

TEST(IterateValues, ConstMultiMappingIteration) {
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
        IterateValues(primesBelow),
        ElementsAre(2, 3, 5, 7, 11, 13, 17, 23)
    );
    auto [begin, end] = primesBelow.equal_range(11);
    EXPECT_EQ(std::distance(begin, end), 2);
    EXPECT_THAT(
        IterateValues(std::vector(begin, end)),
        ElementsAre(7, 11)
    );
}

TEST(IterateValues, ConstUnorderedMultiMappingIteration) {
    const std::unordered_multimap<int, int> primesBelow{
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
        IterateValues(primesBelow),
        UnorderedElementsAre(2, 3, 5, 7, 11, 13, 17, 23)
    );

    auto [begin, end] = primesBelow.equal_range(11);
    EXPECT_EQ(std::distance(begin, end), 2);
    EXPECT_THAT(
        IterateValues(std::vector(begin, end)),
        UnorderedElementsAre(7, 11)
    );
}
