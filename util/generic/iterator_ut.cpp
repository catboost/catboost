#include "iterator.h"

#include <library/unittest/registar.h>

Y_UNIT_TEST_SUITE(TIterator) {
    Y_UNIT_TEST(ToForwardIteratorTest) {
        TVector<int> x = {1, 2};
        UNIT_ASSERT_VALUES_EQUAL(*std::prev(x.end()), *ToForwardIterator(x.rbegin()));
        UNIT_ASSERT_VALUES_EQUAL(*ToForwardIterator(std::prev(x.rend())), *x.begin());
    }
}

Y_UNIT_TEST_SUITE(TInputRangeAdaptor) {
    class TSquaresGenerator : public TInputRangeAdaptor<TSquaresGenerator> {
    public:
        using TRetVal = const i64*;
        TRetVal Next() {
            Current_ = State_ * State_;
            ++State_;
            // Never return nullptr => we have infinite range!
            return &Current_;
        }

    private:
        i64 State_ = 0.0;
        i64 Current_ = 0.0;
    };

    Y_UNIT_TEST(TSquaresGenerator) {
        i64 cur = 0;
        for (i64 sqr : TSquaresGenerator{}) {
            UNIT_ASSERT_VALUES_EQUAL(cur * cur, sqr);

            if (++cur > 10) {
                break;
            }
        }
    }
}
