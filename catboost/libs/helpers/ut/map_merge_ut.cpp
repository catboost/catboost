#include <catboost/libs/helpers/map_merge.h>

#include <util/generic/algorithm.h>
#include <util/generic/vector.h>

#include <library/cpp/testing/unittest/registar.h>


Y_UNIT_TEST_SUITE(TMapMergeTest) {
    Y_UNIT_TEST(TestSumSingleThread) {
        TVector<int> v{0,1,2,3,4,5,6,7,8,9,10};

        NPar::TLocalExecutor localExecutor;
        localExecutor.RunAdditionalThreads(1);

        int res = 0;
        NCB::MapMerge(
            &localExecutor,
            NCB::TSimpleIndexRangesGenerator<int>(NCB::TIndexRange<int>((int)v.size()), 1),
            [&v](NCB::TIndexRange<int> range, int* res) {
                *res = Accumulate(v.begin() + range.Begin, v.begin() + range.End, 0);
            },
            [](int* res, TVector<int>&& mapOutputs) {
                *res += Accumulate(mapOutputs.begin(), mapOutputs.end(), 0);
            },
            &res
        );
        UNIT_ASSERT_EQUAL(res, 10*11 / 2);
    }
    Y_UNIT_TEST(TestSumMultiThread) {
        TVector<int> v{0,1,2,3,4,5,6,7,8,9,10};

        NPar::TLocalExecutor localExecutor;
        localExecutor.RunAdditionalThreads(5);

        int res = 0;
        NCB::MapMerge(
            &localExecutor,
            NCB::TSimpleIndexRangesGenerator<int>(NCB::TIndexRange<int>((int)v.size()), 1),
            [&v](NCB::TIndexRange<int> range, int* res) {
                *res = Accumulate(v.begin() + range.Begin, v.begin() + range.End, 0);
            },
            [](int* res, TVector<int>&& mapOutputs) {
                *res += Accumulate(mapOutputs.begin(), mapOutputs.end(), 0);
            },
            &res
        );
        UNIT_ASSERT_EQUAL(res, 10*11 / 2);
    }
    Y_UNIT_TEST(TestMaxLenMultiThread) {
        TVector<TString> v{"ask", "mail.ru", "baidu", "", "duckduckgo", "yahoo", "bing", "google", "yandex"};

        NPar::TLocalExecutor localExecutor;
        localExecutor.RunAdditionalThreads(5);

        {
            size_t maxLen = 0;
            NCB::MapMerge(
                &localExecutor,
                NCB::TSimpleIndexRangesGenerator<int>(NCB::TIndexRange<int>((int)v.size()), 4),
                [&v](NCB::TIndexRange<int> range, size_t* maxLen) {
                    *maxLen = 0;
                    for (int i : range.Iter()) {
                        *maxLen = Max(*maxLen, v[i].size());
                    }
                },
                [](size_t* maxLen, TVector<size_t>&& mapOutputs) {
                    for (auto len : mapOutputs) {
                        *maxLen = Max(*maxLen, len);
                    }
                },
                &maxLen
            );
            UNIT_ASSERT_EQUAL(maxLen, 10); // maxLen = 10 ("duckduckgo")
        }

        // try with subrange
        {
            size_t maxLen = 0;
            NCB::MapMerge(
                &localExecutor,
                NCB::TSimpleIndexRangesGenerator<int>(NCB::TIndexRange<int>(5, 8), 4), // "yahoo", "bing", "google"
                [&v](NCB::TIndexRange<int> range, size_t* maxLen) {
                    *maxLen = 0;
                    for (int i : range.Iter()) {
                        *maxLen = Max(*maxLen, v[i].size());
                    }
                },
                [](size_t* maxLen, TVector<size_t>&& mapOutputs) {
                    for (auto len : mapOutputs) {
                        *maxLen = Max(*maxLen, len);
                    }
                },
                &maxLen
            );
            UNIT_ASSERT_EQUAL(maxLen, 6); // maxLen = 6 ("google")
        }
    }
}
