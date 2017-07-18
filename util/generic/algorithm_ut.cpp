#include <library/unittest/registar.h>

#include "algorithm.h"
#include "strbuf.h"
#include "string.h"

static auto isOne = [](char c) { return c == '1'; };

SIMPLE_UNIT_TEST_SUITE(TAlgorithm) {
    SIMPLE_UNIT_TEST(AnyTest) {
        UNIT_ASSERT(0 == AnyOf(STRINGBUF("00"), isOne));
        UNIT_ASSERT(1 == AnyOf(STRINGBUF("01"), isOne));
        UNIT_ASSERT(1 == AnyOf(STRINGBUF("10"), isOne));
        UNIT_ASSERT(1 == AnyOf(STRINGBUF("11"), isOne));
        UNIT_ASSERT(0 == AnyOf(TStringBuf(), isOne));
    }

    SIMPLE_UNIT_TEST(AllOfTest) {
        UNIT_ASSERT(0 == AllOf(STRINGBUF("00"), isOne));
        UNIT_ASSERT(0 == AllOf(STRINGBUF("01"), isOne));
        UNIT_ASSERT(0 == AllOf(STRINGBUF("10"), isOne));
        UNIT_ASSERT(1 == AllOf(STRINGBUF("11"), isOne));
        UNIT_ASSERT(1 == AllOf(TStringBuf(), isOne));
    }

    SIMPLE_UNIT_TEST(CountIfTest) {
        UNIT_ASSERT(3 == CountIf(STRINGBUF("____1________1____1_______"), isOne));
        UNIT_ASSERT(5 == CountIf(STRINGBUF("1____1________1____1_______1"), isOne));
        UNIT_ASSERT(0 == CountIf(STRINGBUF("___________"), isOne));
        UNIT_ASSERT(0 == CountIf(TStringBuf(), isOne));
        UNIT_ASSERT(1 == CountIf(STRINGBUF("1"), isOne));
    }

    struct TStrokaNoCopy : TString {
    public:
        TStrokaNoCopy(const char* p)
            : TString(p)
        {
        }

    private:
        TStrokaNoCopy(const TStrokaNoCopy&);
        void operator=(const TStrokaNoCopy&);
    };

    SIMPLE_UNIT_TEST(CountOfTest) {
        UNIT_ASSERT_VALUES_EQUAL(CountOf(1, 2), 0);
        UNIT_ASSERT_VALUES_EQUAL(CountOf(1, 1), 1);
        UNIT_ASSERT_VALUES_EQUAL(CountOf(2, 4, 5), 0);
        UNIT_ASSERT_VALUES_EQUAL(CountOf(2, 4, 2), 1);
        UNIT_ASSERT_VALUES_EQUAL(CountOf(3, 3, 3), 2);

        // Checking comparison of different types.
        UNIT_ASSERT_VALUES_EQUAL(CountOf(0x61, 'x', 'y', 'z'), 0);
        UNIT_ASSERT_VALUES_EQUAL(CountOf(0x61, 'a', 'b', 'c', 0x61), 2);
        UNIT_ASSERT_VALUES_EQUAL(CountOf(0x61, 'a', 'b', 'c', 0x61ll), 2);

        // TString and const char *
        UNIT_ASSERT_VALUES_EQUAL(CountOf(TString("xyz"), "123", "poi"), 0);
        UNIT_ASSERT_VALUES_EQUAL(CountOf(TString("xyz"), "123", "poi", "xyz"), 1);

        // TString and TStringBuf
        UNIT_ASSERT_VALUES_EQUAL(CountOf(TString("xyz"), STRINGBUF("123"), STRINGBUF("poi")), 0);
        UNIT_ASSERT_VALUES_EQUAL(CountOf(TString("xyz"), STRINGBUF("123"), STRINGBUF("poi"),
                                         STRINGBUF("xyz")),
                                 1);

        // TStringBuf and const char *
        UNIT_ASSERT_VALUES_EQUAL(CountOf(STRINGBUF("xyz"), "123", "poi"), 0);
        UNIT_ASSERT_VALUES_EQUAL(CountOf(STRINGBUF("xyz"), "123", "poi", "xyz"), 1);

        // TStringBuf and TString
        UNIT_ASSERT_VALUES_EQUAL(CountOf(STRINGBUF("xyz"), TString("123"), TString("poi")), 0);
        UNIT_ASSERT_VALUES_EQUAL(CountOf(STRINGBUF("xyz"), TString("123"), TString("poi"),
                                         TString("xyz")),
                                 1);
    }

    SIMPLE_UNIT_TEST(EqualToOneOfTest) {
        UNIT_ASSERT(1 == EqualToOneOf(1, 1, 2));
        UNIT_ASSERT(1 == EqualToOneOf(2, 1, 2));
        UNIT_ASSERT(0 == EqualToOneOf(3, 1, 2));
        UNIT_ASSERT(1 == EqualToOneOf(1, 1));
        UNIT_ASSERT(0 == EqualToOneOf(1, 2));
        UNIT_ASSERT(0 == EqualToOneOf(3));

        //test, that EqualToOneOf can compare different types, and don't copy objects:
        TStrokaNoCopy x("x");
        TStrokaNoCopy y("y");
        TStrokaNoCopy z("z");
        const char* px = "x";
        const char* py = "y";
        const char* pz = "z";

        UNIT_ASSERT(1 == EqualToOneOf(x, px, py));
        UNIT_ASSERT(1 == EqualToOneOf(y, px, py));
        UNIT_ASSERT(1 == EqualToOneOf(y, px, y));
        UNIT_ASSERT(1 == EqualToOneOf(y, x, py));
        UNIT_ASSERT(0 == EqualToOneOf(z, px, py));
        UNIT_ASSERT(1 == EqualToOneOf(px, x, y));
        UNIT_ASSERT(1 == EqualToOneOf(py, x, y));
        UNIT_ASSERT(0 == EqualToOneOf(pz, x, y));
    }

    template <class TTestConstPtr>
    void TestFindIfPtrFoundValue(int j, TTestConstPtr root) {
        if (j == 3) {
            UNIT_ASSERT(root == nullptr);
        } else if (j == 4) {
            UNIT_ASSERT(root && *root == 2);
        } else {
            ythrow yexception() << "invalid param " << j;
        }
    }

    struct TVectorNoCopy : std::vector<int> {
    public:
        TVectorNoCopy() {
        }

    private:
        TVectorNoCopy(const TVectorNoCopy&);
        void operator=(const TVectorNoCopy&);
    };

    SIMPLE_UNIT_TEST(FindIfPtrTest) {
        TVectorNoCopy v;
        v.push_back(1);
        v.push_back(2);
        v.push_back(3);

        int array[3] = {1, 2, 3};
        const int array_const[3] = {1, 2, 3};

        //test (const, non-const) * (iterator, vector, array) * (found, not found) variants.
        // search, that 2*2 == 4, but there is no value 'x' in array that (x*x == 3)
        for (int j = 3; j <= 4; ++j) {
            TestFindIfPtrFoundValue<int*>(j, FindIfPtr(v, [j](int i) { return i * i == j; }));
            TestFindIfPtrFoundValue<int*>(j, FindIfPtr(v.begin(), v.end(), [j](int i) { return i * i == j; }));
            const TVectorNoCopy& q = v;
            TestFindIfPtrFoundValue<const int*>(j, FindIfPtr(q, [j](int i) { return i * i == j; }));

            TestFindIfPtrFoundValue<const int*>(j, FindIfPtr(q.begin(), q.end(), [j](int i) { return i * i == j; }));
            TestFindIfPtrFoundValue<int*>(j, FindIfPtr(array, [j](int i) { return i * i == j; }));
            TestFindIfPtrFoundValue<const int*>(j, FindIfPtr(array_const, [j](int i) { return i * i == j; }));
        }
    }

    SIMPLE_UNIT_TEST(FindIndexTest) {
        TVectorNoCopy v;
        v.push_back(1);
        v.push_back(2);
        v.push_back(3);

        UNIT_ASSERT_EQUAL(0, FindIndex(v, 1));
        UNIT_ASSERT_EQUAL(1, FindIndex(v, 2));
        UNIT_ASSERT_EQUAL(2, FindIndex(v, 3));
        UNIT_ASSERT_EQUAL(NPOS, FindIndex(v, 42));

        int array[3] = {1, 2, 3};

        UNIT_ASSERT_EQUAL(0, FindIndex(array, 1));
        UNIT_ASSERT_EQUAL(1, FindIndex(array, 2));
        UNIT_ASSERT_EQUAL(2, FindIndex(array, 3));
        UNIT_ASSERT_EQUAL(NPOS, FindIndex(array, 42));

        yvector<int> empty;
        UNIT_ASSERT_EQUAL(NPOS, FindIndex(empty, 0));
    }

    SIMPLE_UNIT_TEST(SortUniqueTest) {
        {
            yvector<TString> v;
            SortUnique(v);
            UNIT_ASSERT_EQUAL(v, yvector<TString>());
        }

        {
            const char* ar[] = {"345", "3", "123", "2", "23", "3", "2"};
            yvector<TString> v(ar, ar + Y_ARRAY_SIZE(ar));
            SortUnique(v);

            const char* suAr[] = {"123", "2", "23", "3", "345"};
            yvector<TString> suV(suAr, suAr + Y_ARRAY_SIZE(suAr));

            UNIT_ASSERT_EQUAL(v, suV);
        }
    }

    SIMPLE_UNIT_TEST(EraseTest) {
        yvector<int> data = {5, 4, 3, 2, 1, 0};
        yvector<int> expected = {5, 4, 2, 1, 0};
        Erase(data, 3);
        UNIT_ASSERT_EQUAL(data, expected);
    }

    SIMPLE_UNIT_TEST(EraseIfTest) {
        yvector<int> data = {5, 4, 3, 2, 1, 0};
        yvector<int> expected = {2, 1, 0};
        EraseIf(data, [](int i) { return i >= 3; });
        UNIT_ASSERT_EQUAL(data, expected);
    }

    SIMPLE_UNIT_TEST(NthElementTest) {
        {
            yvector<TString> v;
            NthElement(v.begin(), v.begin(), v.end());
            UNIT_ASSERT_EQUAL(v, yvector<TString>());
        }

        {
            int data[] = {3, 2, 1, 4, 6, 5, 7, 9, 8};
            yvector<int> testVector(data, data + Y_ARRAY_SIZE(data));

            size_t medianInd = testVector.size() / 2;

            NthElement(testVector.begin(), testVector.begin() + medianInd, testVector.end());
            UNIT_ASSERT_EQUAL(testVector[medianInd], 5);

            NthElement(testVector.begin(), testVector.begin() + medianInd, testVector.end(), [](int lhs, int rhs) { return lhs > rhs; });
            UNIT_ASSERT_EQUAL(testVector[medianInd], 5);
        }

        {
            const char* data[] = {"3", "234", "1231", "333", "545345", "11", "111", "55", "66"};
            yvector<TString> testVector(data, data + Y_ARRAY_SIZE(data));

            size_t medianInd = testVector.size() / 2;
            NthElement(testVector.begin(), testVector.begin() + medianInd, testVector.end());

            auto median = testVector.begin() + medianInd;
            for (auto it0 = testVector.begin(); it0 != median; ++it0) {
                for (auto it1 = median; it1 != testVector.end(); ++it1) {
                    UNIT_ASSERT(*it0 <= *it1);
                }
            }
        }
    }

    SIMPLE_UNIT_TEST(BinarySearchTest) {
        {
            yvector<TString> v;
            bool test = BinarySearch(v.begin(), v.end(), "test");
            UNIT_ASSERT_EQUAL(test, false);
        }

        {
            int data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};

            bool test = BinarySearch(data, data + Y_ARRAY_SIZE(data), 2);
            UNIT_ASSERT_EQUAL(test, true);

            test = BinarySearch(data, data + Y_ARRAY_SIZE(data), 10);
            UNIT_ASSERT_EQUAL(test, false);
        }

        {
            yvector<size_t> data = {10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0};

            bool test = BinarySearch(data.begin(), data.end(), (size_t)9, TGreater<size_t>());
            UNIT_ASSERT_EQUAL(test, true);

            test = BinarySearch(data.begin(), data.end(), (size_t)11, TGreater<size_t>());
            UNIT_ASSERT_EQUAL(test, false);

            test = BinarySearch(data.rbegin(), data.rend(), (size_t)1);
            UNIT_ASSERT_EQUAL(test, true);
        }
    }

    SIMPLE_UNIT_TEST(EqualRangeTest) {
        {
            yvector<TString> v;
            using PairOfVector = std::pair<yvector<TString>::iterator, yvector<TString>::iterator>;
            PairOfVector tmp = EqualRange(v.begin(), v.end(), "tmp");

            UNIT_ASSERT_EQUAL(tmp.first, tmp.second);
            UNIT_ASSERT_EQUAL(tmp.first, v.end());
        }

        {
            int data[] = {1, 1, 1, 1, 1, 2, 2, 3, 3, 3, 3, 4, 5};
            using PairOfInt = std::pair<int*, int*>;
            PairOfInt tmp = EqualRange(data, data + Y_ARRAY_SIZE(data), 3);

            UNIT_ASSERT_EQUAL(tmp.second - tmp.first, 4);
            UNIT_ASSERT_EQUAL(tmp.first - data, 7);
            UNIT_ASSERT_EQUAL(data + Y_ARRAY_SIZE(data) - tmp.second, 2);
        }

        {
            yvector<size_t> data = {9, 9, 8, 8, 8, 5, 4, 3, 3, 0, 0};

            using PairOfVector = std::pair<yvector<size_t>::iterator, yvector<size_t>::iterator>;
            PairOfVector tmp = EqualRange(data.begin(), data.end(), 8, TGreater<size_t>());

            UNIT_ASSERT_EQUAL(tmp.first - data.begin(), 2);
            UNIT_ASSERT_EQUAL(tmp.second - tmp.first, 3);

            using PairOfVectorReverse = std::pair<yvector<size_t>::reverse_iterator, yvector<size_t>::reverse_iterator>;
            PairOfVectorReverse tmpR = EqualRange(data.rbegin(), data.rend(), (size_t)0);

            UNIT_ASSERT_EQUAL(tmpR.first, data.rbegin());
            UNIT_ASSERT_EQUAL(tmpR.second - tmpR.first, 2);
        }
    }

    SIMPLE_UNIT_TEST(IsSortedTest) {
        yvector<int> v0;
        UNIT_ASSERT_VALUES_EQUAL(IsSorted(v0.begin(), v0.end()), true);

        yvector<int> v1 = {1, 2, 3, 4, 5, 5, 5, 6, 6, 7, 8};
        UNIT_ASSERT_VALUES_EQUAL(IsSorted(v1.begin(), v1.end()), true);
        UNIT_ASSERT_VALUES_EQUAL(IsSorted(v1.begin(), v1.end(), TLess<int>()), true);
        UNIT_ASSERT_VALUES_EQUAL(IsSorted(v1.begin(), v1.end(), TGreater<int>()), false);

        yvector<int> v2 = {1, 2, 1};
        UNIT_ASSERT_VALUES_EQUAL(IsSorted(v2.begin(), v2.end()), false);
        UNIT_ASSERT_VALUES_EQUAL(IsSorted(v2.begin(), v2.end(), TLess<int>()), false);
        UNIT_ASSERT_VALUES_EQUAL(IsSorted(v2.begin(), v2.end(), TGreater<int>()), false);
    }

    SIMPLE_UNIT_TEST(SortByTest) {
        yvector<int> collection = {10, 2, 7};
        SortBy(collection, [](int x) { return -x; });
        yvector<int> expected = {10, 7, 2};
        UNIT_ASSERT_VALUES_EQUAL(collection, expected);
    }

    SIMPLE_UNIT_TEST(StableSortByTest) {
        yvector<int> collection = {404, 101, 106, 203, 102, 205, 401};
        StableSortBy(collection, [](int x) { return x / 100; });
        yvector<int> expected = {101, 106, 102, 203, 205, 404, 401};
        UNIT_ASSERT_VALUES_EQUAL(collection, expected);
    }

    SIMPLE_UNIT_TEST(SortUniqueByTest) {
        yvector<int> collection = {404, 101, 101, 203, 101, 203, 404};
        StableSortUniqueBy(collection, [](int x) { return x / 100; });
        yvector<int> expected = {101, 203, 404};
        UNIT_ASSERT_VALUES_EQUAL(collection, expected);
    }

    SIMPLE_UNIT_TEST(StableSortUniqueByTest) {
        yvector<int> collection = {404, 101, 106, 203, 102, 205, 401};
        StableSortUniqueBy(collection, [](int x) { return x / 100; });
        yvector<int> expected = {101, 203, 404};
        UNIT_ASSERT_VALUES_EQUAL(collection, expected);
    }

    SIMPLE_UNIT_TEST(IotaTest) {
        yvector<int> v(10);

        Iota(v.begin(), v.end(), 0);
        UNIT_ASSERT_VALUES_EQUAL(v[0], 0);
        UNIT_ASSERT_VALUES_EQUAL(v[5], 5);
        UNIT_ASSERT_VALUES_EQUAL(v[9], 9);

        Iota(v.begin() + 2, v.begin() + 5, 162);
        UNIT_ASSERT_VALUES_EQUAL(v[0], 0);
        UNIT_ASSERT_VALUES_EQUAL(v[3], 163);
        UNIT_ASSERT_VALUES_EQUAL(v[9], 9);
    }

    SIMPLE_UNIT_TEST(CopyNTest) {
        int data[] = {1, 2, 3, 4, 8, 7, 6, 5};
        const size_t vSize = 10;
        yvector<int> result(10, 0);
        size_t toCopy = 5;

        yvector<int>::iterator iter = CopyN(data, toCopy, result.begin());
        UNIT_ASSERT_VALUES_EQUAL(iter - result.begin(), toCopy);
        UNIT_ASSERT_VALUES_EQUAL(+result, 10);
        for (size_t idx = 0; idx < toCopy; ++idx) {
            UNIT_ASSERT_VALUES_EQUAL(data[idx], result[idx]);
        }
        for (size_t idx = toCopy; idx < vSize; ++idx) {
            UNIT_ASSERT_VALUES_EQUAL(result[idx], 0);
        }

        toCopy = 8;
        const size_t start = 1;
        result.assign(vSize, 0);
        iter = CopyN(data, toCopy, result.begin() + start);
        UNIT_ASSERT_VALUES_EQUAL(iter - result.begin(), start + toCopy);
        for (size_t idx = 0; idx < start; ++idx) {
            UNIT_ASSERT_VALUES_EQUAL(result[idx], 0);
        }
        for (size_t idx = 0; idx < toCopy; ++idx) {
            UNIT_ASSERT_VALUES_EQUAL(result[start + idx], data[idx]);
        }
        for (size_t idx = start + toCopy; idx < vSize; ++idx) {
            UNIT_ASSERT_VALUES_EQUAL(result[idx], 0);
        }
    }

    SIMPLE_UNIT_TEST(CopyIfTest) {
        const size_t count = 9;
        int data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
        const size_t vSize = 10;
        yvector<int> v(vSize, 0);

        yvector<int>::iterator iter = CopyIf(data, data + count, v.begin(), [](int x) { return !(x % 3); });
        UNIT_ASSERT_VALUES_EQUAL(+v, vSize);
        UNIT_ASSERT_VALUES_EQUAL(iter - v.begin(), 3);
        v.resize(iter - v.begin());
        for (size_t idx = 0; idx < v.size(); ++idx) {
            UNIT_ASSERT_VALUES_EQUAL(v[idx], 3 * (idx + 1));
        }
    }

    SIMPLE_UNIT_TEST(MinMaxElementTest) {
        yvector<int> v(10);
        Iota(v.begin(), v.end(), 0);
        UNIT_ASSERT_EQUAL(*MinMaxElement(v.begin(), v.end()).first, 0);
        UNIT_ASSERT_EQUAL(*MinMaxElement(v.begin(), v.end()).second, 9);

        v[3] = -2;
        v[7] = 11;
        UNIT_ASSERT_EQUAL(*MinMaxElement(v.begin(), v.end()).first, -2);
        UNIT_ASSERT_EQUAL(*MinMaxElement(v.begin(), v.end()).second, 11);
    }

    SIMPLE_UNIT_TEST(MinMaxTest) {
        std::pair<int, int> p1 = MinMax(5, 12);
        UNIT_ASSERT_EQUAL(p1.first, 5);
        UNIT_ASSERT_EQUAL(p1.second, 12);

        std::pair<TString, TString> p2 = MinMax(TString("test"), TString("data"));
        UNIT_ASSERT_EQUAL(p2.first, TString("data"));
        UNIT_ASSERT_EQUAL(p2.second, TString("test"));
    }

    SIMPLE_UNIT_TEST(TestMaxElementBy) {
        const int array[] = {1, 2, 5, 3, 4, 5};
        UNIT_ASSERT_VALUES_EQUAL(*MaxElementBy(array, [](int x) {
            return x * x;
        }),
                                 5);

        const yvector<int> vec(array, array + Y_ARRAY_SIZE(array));
        UNIT_ASSERT_VALUES_EQUAL(*MaxElementBy(vec, [](int x) {
            return -1.0 * x;
        }),
                                 1);

        auto identity = [](char x) {
            return x;
        };
        auto singleElementSequence = {'z'};
        UNIT_ASSERT_VALUES_EQUAL(*MaxElementBy(singleElementSequence, identity), 'z');

        const TString strings[] = {"one", "two", "three", "four"};
        UNIT_ASSERT_STRINGS_EQUAL(*MaxElementBy(strings, [](TString s) { return s.size(); }), "three");
    }

    SIMPLE_UNIT_TEST(TestMinElementBy) {
        const int array[] = {2, 3, 4, 1, 5};
        UNIT_ASSERT_VALUES_EQUAL(*MinElementBy(array, [](int x) -> char {
            return 'a' + x;
        }),
                                 1);

        const yvector<int> vec(std::begin(array), std::end(array));
        UNIT_ASSERT_VALUES_EQUAL(*MinElementBy(vec, [](int x) {
            return -x;
        }),
                                 5);

        auto identity = [](char x) {
            return x;
        };
        auto singleElementSequence = {'z'};
        UNIT_ASSERT_VALUES_EQUAL(*MinElementBy(singleElementSequence, identity), 'z');

        const yvector<TStringBuf> strings = {"one", "two", "three", "four"};
        auto stringLength = [](TStringBuf s) {
            return s.size();
        };
        UNIT_ASSERT_STRINGS_EQUAL(*MinElementBy(strings, stringLength), "one");
        UNIT_ASSERT_STRINGS_EQUAL(*MinElementBy(strings.rbegin(), strings.rend(), stringLength), "two");
    }

    SIMPLE_UNIT_TEST(MaxElementByReturnsEndForEmptyRange) {
        const yvector<int> empty;
        UNIT_ASSERT_EQUAL(MaxElementBy(empty, [](int) { return 0; }), empty.end());
    }

    SIMPLE_UNIT_TEST(MaxElementByDoesntCallFunctorForEmptyRange) {
        const yvector<int> empty;
        auto functor = [](int) {
            UNIT_ASSERT(false);
            return 0;
        };
        MaxElementBy(empty, functor);
    }

    SIMPLE_UNIT_TEST(MinElementByReturnsEndForEmptyRange) {
        const yvector<int> empty;
        UNIT_ASSERT_EQUAL(MinElementBy(empty, [](int) { return 0; }), empty.end());
    }

    SIMPLE_UNIT_TEST(MinElementByDoesntCallFunctorForEmptyRange) {
        const yvector<int> empty;
        auto functor = [](int) {
            UNIT_ASSERT(false);
            return 0;
        };
        MinElementBy(empty, functor);
    }

    SIMPLE_UNIT_TEST(FindIfForContainer) {
        using std::begin;
        using std::end;

        int array[] = {1, 2, 3, 4, 5};
        UNIT_ASSERT_EQUAL(FindIf(array, [](int x) { return x == 1; }), begin(array));
        UNIT_ASSERT_EQUAL(FindIf(array, [](int x) { return x > 5; }), end(array));

        yvector<int> vector = {1, 2, 3, 4, 5};
        UNIT_ASSERT_EQUAL(FindIf(vector, [](int x) { return x == 1; }), begin(vector));
        UNIT_ASSERT_EQUAL(FindIf(vector, [](int x) { return x > 5; }), end(vector));

        // Compilability test. Check if the returned iterator is non const
        auto iter = FindIf(vector, [](int x) { return x == 1; });
        *iter = 5;

        // Compilability test. Check if the returned iterator is const. Should not compile
        const yvector<int> constVector = {1, 2, 3, 4, 5};
        auto constIter = FindIf(constVector, [](int x) { return x == 1; });
        Y_UNUSED(constIter);
        // *constIter = 5;
    }

    struct TRange {
    };

    const TRange* begin(const TRange& r) {
        return &r;
    }

    const TRange* end(const TRange& r) {
        return &r + 1;
    }

    SIMPLE_UNIT_TEST(FindIfForUserType) {
        // Compileability test. Should work for user types with begin/end overloads
        TRange range;
        auto i = FindIf(range, [](auto) { return false; });
        Y_UNUSED(i);
    }

    SIMPLE_UNIT_TEST(TestLowerBoundBy) {
        using TIntPairs = yvector<std::pair<i32, i32>>;

        auto data = TIntPairs{{1, 5}, {3, 2}, {3, 4}, {8, 0}, {5, 4}};
        auto getKey = [](const auto& x) { return x.second; };

        StableSortBy(data, getKey);

        auto it = LowerBoundBy(data.begin(), data.end(), 4, getKey);
        UNIT_ASSERT(it != data.end());
        UNIT_ASSERT_EQUAL(it->second, 4);
        UNIT_ASSERT_EQUAL(it->first, 3);

        UNIT_ASSERT(it > data.begin());
        UNIT_ASSERT_EQUAL((it - 1)->second, 2);

        UNIT_ASSERT((it + 1) < data.end());
        UNIT_ASSERT_EQUAL((it + 1)->second, 4);
    }

    SIMPLE_UNIT_TEST(TestUpperBoundBy) {
        using TIntPairs = yvector<std::pair<i32, i32>>;

        auto data = TIntPairs{{1, 5}, {3, 2}, {3, 4}, {8, 0}, {5, 4}};
        auto getKey = [](const auto& x) { return x.second; };

        StableSortBy(data, getKey);

        auto it = UpperBoundBy(data.begin(), data.end(), 4, getKey);
        UNIT_ASSERT(it != data.end());
        UNIT_ASSERT_EQUAL(it->second, 5);
        UNIT_ASSERT_EQUAL(it->first, 1);

        UNIT_ASSERT(it > data.begin());
        UNIT_ASSERT_EQUAL((it - 1)->second, 4);

        UNIT_ASSERT((it + 1) == data.end());
    }

    SIMPLE_UNIT_TEST(TestFindInContainer) {
        std::vector<int> v = {1, 2, 1000, 15, 100};
        UNIT_ASSERT(Find(v, 5) == v.end());
        UNIT_ASSERT(Find(v, 1) == v.begin());
        UNIT_ASSERT(Find(v, 100) == v.end() - 1);
    }
};
