#include "xrange.h"

#include "algorithm.h"
#include "maybe.h"
#include "vector.h"
#include <library/cpp/testing/unittest/registar.h>
#include <util/string/builder.h>

Y_UNIT_TEST_SUITE(XRange) {
    void TestXRangeImpl(size_t begin, size_t end) {
        size_t count = 0;
        size_t sum = 0;
        size_t first = 42;
        bool firstInited = false;
        size_t last = 0;

        for (auto i : xrange(begin, end)) {
            ++count;
            sum += i;
            last = i;
            if (!firstInited) {
                first = i;
                firstInited = true;
            }
        }

        UNIT_ASSERT_VALUES_EQUAL(count, end - begin);
        UNIT_ASSERT_VALUES_EQUAL(first, begin);
        UNIT_ASSERT_VALUES_EQUAL(last, end - 1);
        UNIT_ASSERT_VALUES_EQUAL(sum, count * (first + last) / 2);
    }

    void TestSteppedXRangeImpl(int begin, int end, int step, const TVector<int>& expected) {
        size_t expInd = 0;
        for (auto i : xrange(begin, end, step)) {
            UNIT_ASSERT(expInd < expected.size());
            UNIT_ASSERT_VALUES_EQUAL(i, expected[expInd]);
            ++expInd;
        }
        UNIT_ASSERT_VALUES_EQUAL(expInd, expected.size());
    }

    Y_UNIT_TEST(IncrementWorks) {
        TestXRangeImpl(0, 10);
        TestXRangeImpl(10, 20);
    }

    Y_UNIT_TEST(DecrementWorks) {
        TestSteppedXRangeImpl(10, 0, -1, {10, 9, 8, 7, 6, 5, 4, 3, 2, 1});
        TestSteppedXRangeImpl(10, -1, -1, {10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0});
        TestSteppedXRangeImpl(20, 9, -1, {20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10});
    }

    Y_UNIT_TEST(StepWorks) {
        TestSteppedXRangeImpl(0, 0, 1, {});
        TestSteppedXRangeImpl(0, 9, 3, {0, 3, 6});
        TestSteppedXRangeImpl(0, 10, 3, {0, 3, 6, 9});
        TestSteppedXRangeImpl(0, 11, 3, {0, 3, 6, 9});
        TestSteppedXRangeImpl(0, 12, 3, {0, 3, 6, 9});
        TestSteppedXRangeImpl(0, 13, 3, {0, 3, 6, 9, 12});
        TestSteppedXRangeImpl(0, 10, 2, {0, 2, 4, 6, 8});
        TestSteppedXRangeImpl(15, 0, -4, {15, 11, 7, 3});
        TestSteppedXRangeImpl(15, -1, -4, {15, 11, 7, 3});
        TestSteppedXRangeImpl(15, -2, -4, {15, 11, 7, 3, -1});
    }

    Y_UNIT_TEST(PointersWorks) {
        TVector<size_t> data = {3, 1, 4, 1, 5, 9, 2, 6};
        const size_t digSumExpected = Accumulate(data.begin(), data.end(), static_cast<size_t>(0));
        size_t digSumByIt = 0;
        for (auto ptr : xrange(data.begin(), data.end())) {
            digSumByIt += *ptr;
        }
        UNIT_ASSERT_VALUES_EQUAL(digSumByIt, digSumExpected);
        size_t digSumByPtr = 0;
        for (auto ptr : xrange(&data[0], &data[0] + data.size())) {
            digSumByPtr += *ptr;
        }
        UNIT_ASSERT_VALUES_EQUAL(digSumByPtr, digSumExpected);
    }

    Y_UNIT_TEST(SizeMethodCheck) {
        UNIT_ASSERT_VALUES_EQUAL(xrange(5).size(), 5);
        UNIT_ASSERT_VALUES_EQUAL(xrange(0, 5, 2).size(), 3);
        UNIT_ASSERT_VALUES_EQUAL(xrange(0, 6, 2).size(), 3);
    }

    class TVectorChild: public TVector<size_t> {
    public:
        template <typename TIterator>
        TVectorChild(TIterator a, TIterator b)
            : TVector<size_t>(a, b)
        {
        }
    };

    Y_UNIT_TEST(ConvertionWorks) {
        TVector<size_t> data = {0, 1, 2, 3, 4, 5, 6, 7, 8};

        TVector<size_t> convertionResults[] = {xrange<size_t>(9),
                                               xrange<ui32>(0, 9),
                                               xrange(0, 9, 1)};

        for (const auto& arr : convertionResults) {
            UNIT_ASSERT(arr == data);
        }

        TVectorChild sons[] = {xrange(0, 9),
                               xrange(0, 9, 1)};

        for (const auto& arr : sons) {
            UNIT_ASSERT(arr == data);
        }
    }

    template <class XRangeContainer>
    void TestEmptyRanges(const XRangeContainer& c) {
        for (const auto& emptyRange : c) {
            UNIT_ASSERT_VALUES_EQUAL(emptyRange.size(), 0);

            for (auto i : emptyRange) {
                Y_UNUSED(i);
                UNIT_ASSERT(false);
            }

            using TValueType = decltype(*emptyRange.begin());
            const TVector<TValueType> asVector = emptyRange;
            UNIT_ASSERT(asVector.empty());
        }
    }

    Y_UNIT_TEST(EmptySimpleRange) {
        using TSimpleRange = decltype(xrange(1));

        const TSimpleRange emptySimpleRanges[] = {
            xrange(-1),
            xrange(-10),
            xrange(0, -5),
            xrange(10, 10),
            xrange(10, 9),
        };

        TestEmptyRanges(emptySimpleRanges);
    }

    Y_UNIT_TEST(EmptySteppedRange) {
        using TSteppedRange = decltype(xrange(1, 10, 1));

        const TSteppedRange emptySteppedRanges[] = {
            xrange(5, 5, 1),
            xrange(5, 0, 5),
            xrange(0, -1, 5),
            xrange(0, 1, -1),
            xrange(0, -10, 10),
        };

        TestEmptyRanges(emptySteppedRanges);
    }

    template <class TRange>
    static void TestIteratorDifferenceImpl(TRange range, int a, int b, TMaybe<int> step) {
        auto fmtCase = [&]() -> TString { return TStringBuilder() << "xrange(" << a << ", " << b << (step ? ", " + ToString(*step) : TString{}) << ")"; };
        auto begin = std::begin(range);
        auto end = std::end(range);
        auto distance = end - begin;
        UNIT_ASSERT_VALUES_EQUAL_C(range.size(), distance, fmtCase());
        UNIT_ASSERT_EQUAL_C(end, begin + distance, fmtCase());
    }

    Y_UNIT_TEST(IteratorDifference) {
        for (int a = -20; a <= 20; ++a) {
            for (int b = -20; b <= 20; ++b) {
                for (int step = -25; step <= 25; ++step) {
                    if (step != 0) {
                        TestIteratorDifferenceImpl(xrange(a, b, step), a, b, step);
                    }
                }
                TestIteratorDifferenceImpl(xrange(a, b), a, b, Nothing());
            }
        }
    }

    Y_UNIT_TEST(Advance) {
        {
            auto range = xrange(30, 160, 7);
            auto it = std::begin(range);
            it += 5;
            UNIT_ASSERT_VALUES_EQUAL(*(std::begin(range) + 5), *it);
            UNIT_ASSERT_VALUES_EQUAL(65, *it);
            it -= 2;
            UNIT_ASSERT_VALUES_EQUAL(*(std::begin(range) + 3), *it);
            UNIT_ASSERT_VALUES_EQUAL(51, *it);
            std::advance(it, 10);
            UNIT_ASSERT_VALUES_EQUAL(*(std::begin(range) + 13), *it);
            UNIT_ASSERT_VALUES_EQUAL(121, *it);
            std::advance(it, -5);
            UNIT_ASSERT_VALUES_EQUAL(*(std::begin(range) + 8), *it);
            UNIT_ASSERT_VALUES_EQUAL(86, *it);
        }
        {
            auto range = xrange(-20, 100);
            auto it = std::begin(range);
            it += 5;
            UNIT_ASSERT_VALUES_EQUAL(*(std::begin(range) + 5), *it);
            UNIT_ASSERT_VALUES_EQUAL(-15, *it);
            it -= 2;
            UNIT_ASSERT_VALUES_EQUAL(*(std::begin(range) + 3), *it);
            UNIT_ASSERT_VALUES_EQUAL(-17, *it);
            std::advance(it, 30);
            UNIT_ASSERT_VALUES_EQUAL(*(std::begin(range) + 33), *it);
            UNIT_ASSERT_VALUES_EQUAL(13, *it);
            std::advance(it, -8);
            UNIT_ASSERT_VALUES_EQUAL(*(std::begin(range) + 25), *it);
            UNIT_ASSERT_VALUES_EQUAL(5, *it);
        }
    }
} // Y_UNIT_TEST_SUITE(XRange)
