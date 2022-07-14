#include <library/cpp/iterator/filtering.h>

#include <library/cpp/testing/gtest/gtest.h>

#include <util/generic/vector.h>

using namespace testing;

TEST(Filtering, TFilteringRangeTest) {
    const TVector<int> x = {1, 2, 3, 4, 5};

    EXPECT_THAT(
        MakeFilteringRange(
            x,
            [](int x) { return x % 2 == 0; }
        ),
        ElementsAre(2, 4)
    );
}

TEST(Filtering, TEmptyFilteringRangeTest) {
    TVector<int> x = {1, 2, 3, 4, 5};
    EXPECT_THAT(
        MakeFilteringRange(
            x,
            [](int x) { return x > 100; }
        ),
        ElementsAre()
    );
}

TEST(Filtering, TMutableFilteringRangeTest) {
    TVector<int> x = {1, 2, 3, 4, 5};
    for (auto& y : MakeFilteringRange(x, [](int x) { return x % 2 == 0; })) {
        y = 7;
    }
    EXPECT_THAT(
        x,
        ElementsAre(1, 7, 3, 7, 5)
    );
}
