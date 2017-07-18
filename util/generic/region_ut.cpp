#include <library/unittest/registar.h>

#include "region.h"
#include "vector.h"

#include <iterator>

SIMPLE_UNIT_TEST_SUITE(TRegion) {
    SIMPLE_UNIT_TEST(Iterator) {
        int array[] = {17, 19, 21};
        TRegion<int> r(array, 3);

        TRegion<int>::const_iterator iterator = r.begin();
        for (auto& i : array) {
            UNIT_ASSERT(iterator != r.end());
            UNIT_ASSERT_VALUES_EQUAL(i, *iterator);
            ++iterator;
        }
        UNIT_ASSERT(iterator == r.end());
    }

    SIMPLE_UNIT_TEST(OperatorAt) {
        int array[] = {17, 19, 21};
        TRegion<int> r(array, 3);

        UNIT_ASSERT_VALUES_EQUAL(21, r[2]);
        r[1] = 23;
        UNIT_ASSERT_VALUES_EQUAL(23, array[1]);
    }

    SIMPLE_UNIT_TEST(ConstructorFromValue) {
        int x = 10;
        TRegion<int> r(x);
        UNIT_ASSERT_VALUES_EQUAL(1u, r.Size());
        UNIT_ASSERT_VALUES_EQUAL(10, r[0]);
        r[0] = 11;
        UNIT_ASSERT_VALUES_EQUAL(11, x);
    }

    SIMPLE_UNIT_TEST(ConstructorFromValueConstFromNonConst) {
        int x = 10;
        TRegion<const int> r(x);
        UNIT_ASSERT_VALUES_EQUAL(10, r[0]);
    }

    SIMPLE_UNIT_TEST(ConstructorFromArray) {
        int x[] = {10, 20, 30};
        TRegion<int> r(x);
        UNIT_ASSERT_VALUES_EQUAL(3u, r.size());
        UNIT_ASSERT_VALUES_EQUAL(30, r[2]);
        r[2] = 50;
        UNIT_ASSERT_VALUES_EQUAL(50, x[2]);
    }

    SIMPLE_UNIT_TEST(ConstructorFromArrayConstFromNonConst) {
        int x[] = {100, 200};
        TRegion<const int> r(x);
        UNIT_ASSERT_VALUES_EQUAL(100, r[0]);
    }

    SIMPLE_UNIT_TEST(ConstructorFromRegionConstFromNonConst) {
        TRegion<int> ints;
        TRegion<const int> constInts = ints;
        UNIT_ASSERT_VALUES_EQUAL(0u, constInts.size());
    }

    SIMPLE_UNIT_TEST(ToRegionFromVector) {
        yvector<int> vec;
        vec.push_back(17);
        vec.push_back(19);
        vec.push_back(21);
        TRegion<int> r = ToRegion(vec);
        UNIT_ASSERT_VALUES_EQUAL(21, r[2]);
        r[1] = 23;
        UNIT_ASSERT_VALUES_EQUAL(23, vec[1]);
    }

    SIMPLE_UNIT_TEST(ToRegionFromConstVector) {
        yvector<int> vec;
        vec.push_back(17);
        vec.push_back(19);
        vec.push_back(21);
        TRegion<const int> r = ToRegion(static_cast<const yvector<int>&>(vec));
        UNIT_ASSERT_VALUES_EQUAL(21, r[2]);
    }

    SIMPLE_UNIT_TEST(ConstAndNonConstBeginEndEqualityTest) {
        int x[] = {1, 2, 3};
        TRegion<int> rx{x};
        UNIT_ASSERT_EQUAL(rx.begin(), rx.cbegin());
        UNIT_ASSERT_EQUAL(rx.end(), rx.cend());
        UNIT_ASSERT_EQUAL(rx.rbegin(), rx.crbegin());
        UNIT_ASSERT_EQUAL(rx.rend(), rx.crend());

        int w[] = {1, 2, 3};
        const TRegion<int> rw{w};
        UNIT_ASSERT_EQUAL(rw.begin(), rw.cbegin());
        UNIT_ASSERT_EQUAL(rw.end(), rw.cend());
        UNIT_ASSERT_EQUAL(rw.rbegin(), rw.crbegin());
        UNIT_ASSERT_EQUAL(rw.rend(), rw.crend());

        int y[] = {1, 2, 3};
        TRegion<const int> ry{y};
        UNIT_ASSERT_EQUAL(ry.begin(), ry.cbegin());
        UNIT_ASSERT_EQUAL(ry.end(), ry.cend());
        UNIT_ASSERT_EQUAL(ry.rbegin(), ry.crbegin());
        UNIT_ASSERT_EQUAL(ry.rend(), ry.crend());

        const int z[] = {1, 2, 3};
        TRegion<const int> rz{z};
        UNIT_ASSERT_EQUAL(rz.begin(), rz.cbegin());
        UNIT_ASSERT_EQUAL(rz.end(), rz.cend());
        UNIT_ASSERT_EQUAL(rz.rbegin(), rz.crbegin());
        UNIT_ASSERT_EQUAL(rz.rend(), rz.crend());

        const int q[] = {1, 2, 3};
        const TRegion<const int> rq{q};
        UNIT_ASSERT_EQUAL(rq.begin(), rq.cbegin());
        UNIT_ASSERT_EQUAL(rq.end(), rq.cend());
        UNIT_ASSERT_EQUAL(rq.rbegin(), rq.crbegin());
        UNIT_ASSERT_EQUAL(rq.rend(), rq.crend());
    }

    SIMPLE_UNIT_TEST(ReverseIteratorsTest) {
        const int x[] = {1, 2, 3};
        const TRegion<const int> rx{x};
        auto i = rx.crbegin();
        UNIT_ASSERT_VALUES_EQUAL(*i, 3);
        ++i;
        UNIT_ASSERT_VALUES_EQUAL(*i, 2);
        ++i;
        UNIT_ASSERT_VALUES_EQUAL(*i, 1);
        ++i;
        UNIT_ASSERT_EQUAL(i, rx.crend());
    }

    SIMPLE_UNIT_TEST(FrontBackTest) {
        const int x[] = {1, 2, 3};
        const TRegion<const int> rx{x};
        UNIT_ASSERT_VALUES_EQUAL(rx.front(), 1);
        UNIT_ASSERT_VALUES_EQUAL(rx.back(), 3);

        int y[] = {1, 2, 3};
        TRegion<int> ry{y};
        UNIT_ASSERT_VALUES_EQUAL(ry.front(), 1);
        UNIT_ASSERT_VALUES_EQUAL(ry.back(), 3);

        ry.front() = 100;
        ry.back() = 500;
        UNIT_ASSERT_VALUES_EQUAL(ry.front(), 100);
        UNIT_ASSERT_VALUES_EQUAL(ry.back(), 500);
        UNIT_ASSERT_VALUES_EQUAL(y[0], 100);
        UNIT_ASSERT_VALUES_EQUAL(y[2], 500);
    }

    void CheckRegion(const yvector<char>& expected, const TDataRegion& region) {
        UNIT_ASSERT_VALUES_EQUAL(expected.size(), region.size());
        for (size_t i = 0; i < expected.size(); ++i) {
            UNIT_ASSERT_VALUES_EQUAL(expected[i], region[i]);
        }
    }

    SIMPLE_UNIT_TEST(SubRegion) {
        yvector<char> x;
        for (size_t i = 0; i < 42; ++i) {
            x.push_back('a' + (i * 42424243) % 13);
        }
        TDataRegion region(x.data(), 42);
        for (size_t i = 0; i <= 50; ++i) {
            yvector<char> expected;
            for (size_t j = 0; j <= 100; ++j) {
                CheckRegion(expected, region.SubRegion(i, j));
                if (i + j < 42) {
                    expected.push_back(x[i + j]);
                }
            }
        }
    }
}
