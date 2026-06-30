#include <library/cpp/iterator/functools.h>

#include <library/cpp/testing/gtest/gtest.h>

#include <util/generic/vector.h>
#include <util/generic/xrange.h>
#include <util/generic/adaptor.h>

#include <set>

// default-win-x86_64-release compiler can't decompose tuple to structure binding (02.03.2019)
#ifndef _WINDOWS
#   define FOR_DISPATCH_2(i, j, r) \
        for (auto [i, j] : r)
#   define FOR_DISPATCH_3(i, j, k, r) \
        for (auto [i, j, k] : r)
#else
#   define FOR_DISPATCH_2(i, j, r) \
        for (auto __t_##i##_##j : r) \
            if (auto& i = std::get<0>(__t_##i##_##j); true) \
                if (auto& j = std::get<1>(__t_##i##_##j); true)
#   define FOR_DISPATCH_3(i, j, k, r) \
        for (auto __t_##i##_##j##_##k : r) \
            if (auto& i = std::get<0>(__t_##i##_##j##_##k); true) \
                if (auto& j = std::get<1>(__t_##i##_##j##_##k); true) \
                    if (auto& k = std::get<2>(__t_##i##_##j##_##k); true)
#endif

using namespace NFuncTools;


    template <typename TContainer>
    auto ToVector(TContainer&& container) {
        return std::vector{container.begin(), container.end()};
    }

    template <typename TContainerObjOrRef>
    void TestViewCompileability(TContainerObjOrRef&& container) {
        using TContainer = std::decay_t<TContainerObjOrRef>;
        using TIterator = typename TContainer::iterator;

        static_assert(std::is_same_v<decltype(container.begin()), TIterator>);

        // iterator_traits must work!
        using difference_type = typename std::iterator_traits<TIterator>::difference_type;
        using value_type = typename std::iterator_traits<TIterator>::value_type;
        using reference = typename std::iterator_traits<TIterator>::reference;
        using pointer = typename std::iterator_traits<TIterator>::pointer;

        {
            // operator assignment
            auto it = container.begin();
            it = container.end();
            it = std::move(container.begin());
            // operator copying
            auto it2 = it;
            Y_UNUSED(it2);
            auto it3 = std::move(it);
            Y_UNUSED(it3);
            Y_UNUSED(*it3);
            EXPECT_TRUE(it3 == it3);
            EXPECT_FALSE(it3 != it3);
            // const TIterator
            const auto it4 = it3;
            Y_UNUSED(*it4);
            EXPECT_TRUE(it4 == it4);
            EXPECT_FALSE(it4 != it4);
            EXPECT_TRUE(it3 == it4);
            EXPECT_TRUE(it4 == it3);
            EXPECT_FALSE(it3 != it4);
            EXPECT_FALSE(it4 != it3);
        }

        auto it = container.begin();

        // sanity check for types
        using TConstReference = const std::remove_reference_t<reference>&;
        TConstReference ref = *it;
        Y_UNUSED(ref);
        (void) static_cast<value_type>(*it);
        (void) static_cast<difference_type>(1);
        if constexpr (std::is_reference_v<decltype(*it)>) {
            pointer ptr = &*it;
            Y_UNUSED(ptr);
        }

        // std compatibility
        ToVector(container);

        // const iterators
        [](const auto& cont) {
            auto constBeginIterator = cont.begin();
            auto constEndIterator = cont.end();
            static_assert(std::is_same_v<decltype(constBeginIterator), typename TContainer::const_iterator>);
            Y_UNUSED(constBeginIterator);
            Y_UNUSED(constEndIterator);
        }(container);
    }

    struct TTestSentinel {};
    struct TTestIterator {
        int operator*() const {
            return X;
        }
        void operator++() {
            ++X;
        }
        bool operator!=(const TTestSentinel&) const {
            return X < 3;
        }

        int X;
    };

    // container with minimal interface
    auto MakeMinimalisticContainer() {
        return MakeIteratorRange(TTestIterator{}, TTestSentinel{});
    }


    TEST(FuncTools, CompileRange) {
        TestViewCompileability(Range(19));
        TestViewCompileability(Range(10, 19));
        TestViewCompileability(Range(10, 19, 2));
    }


    TEST(FuncTools, Enumerate) {
        TVector<size_t> a = {1, 2, 4};
        TVector<size_t> b;
        TVector<size_t> c = {1};
        for (auto& v : {a, b, c}) {
            size_t j = 0;
            FOR_DISPATCH_2(i, x, Enumerate(v)) {
                EXPECT_EQ(v[i], x);
                EXPECT_EQ(i, j++);
                EXPECT_LT(i, v.size());
            }
            EXPECT_EQ(j, v.size());
        }

        // Test correctness of iterator traits.
        auto enumerated = Enumerate(a);
        static_assert(std::ranges::input_range<decltype(enumerated)>);
        static_assert(
            std::is_same_v<decltype(enumerated.begin())::pointer,
            std::iterator_traits<decltype(enumerated.begin())>::pointer>);

        // Post-increment test.
        auto it = enumerated.begin();
        EXPECT_EQ(*(it++), (std::tuple{0, 1}));
        EXPECT_EQ(*it, (std::tuple{1, 2}));

        TVector<size_t> d = {0, 0, 0};
        FOR_DISPATCH_2(i, x, Enumerate(d)) {
            x = i;
        }
        EXPECT_THAT(
            d,
            testing::ElementsAre(0u, 1u, 2u)
        );
    }

    TEST(FuncTools, EnumerateTemporary) {
        TVector<size_t> a = {1, 2, 4};
        TVector<size_t> b;
        TVector<size_t> c = {1};
        for (auto& v : {a, b, c}) {
            size_t j = 0;
            FOR_DISPATCH_2(i, x, Enumerate(TVector(v))) {
                EXPECT_EQ(v[i], x);
                EXPECT_EQ(i, j++);
                EXPECT_LT(i, v.size());
            }
            EXPECT_EQ(j, v.size());
        }

        FOR_DISPATCH_2(i, x, Enumerate(TVector<size_t>{1, 2, 3})) {
            EXPECT_EQ(i + 1, x);
        }
    }

    TEST(FuncTools, CompileEnumerate) {
        auto container = std::vector{1, 2, 3};
        TestViewCompileability(Enumerate(container));
        const auto constContainer = std::vector{1, 2, 3};
        TestViewCompileability(Enumerate(constContainer));
        const int arrayContainer[] = {1, 2, 3};
        TestViewCompileability(Enumerate(arrayContainer));

        std::vector<std::pair<int, int>> res;
        FOR_DISPATCH_2(i, x, Enumerate(MakeMinimalisticContainer())) {
            res.push_back({i, x});
        }
        EXPECT_EQ(res, (std::vector<std::pair<int, int>>{
            {0, 0}, {1, 1}, {2, 2},
        }));
    }

    TEST(FuncTools, Zip) {
        TVector<std::pair<TVector<size_t>, TVector<size_t>>> ts = {
            {{1, 2, 3}, {4, 5, 6}},
            {{1, 2, 3}, {4, 5, 6, 7}},
            {{1, 2, 3, 4}, {4, 5, 6}},
            {{1, 2, 3, 4}, {}},
        };

        FOR_DISPATCH_2(a, b, ts) {
            size_t k = 0;
            FOR_DISPATCH_2(i, j, Zip(a, b)) {
                EXPECT_EQ(++k, i);
                EXPECT_EQ(i + 3, j);
            }
            EXPECT_EQ(k, Min(a.size(), b.size()));
        }
    }

    TEST(FuncTools, ZipReference) {
        TVector a = {0, 1, 2};
        TVector b = {2, 1, 0, -1};
        FOR_DISPATCH_2(ai, bi, Zip(a, b)) {
            ai = bi;
        }
        EXPECT_THAT(
            a,
            testing::ElementsAre(2u, 1u, 0u)
        );
    }

    TEST(FuncTools, Zip3) {
        TVector<std::tuple<TVector<i32>, TVector<i32>, TVector<i32>>> ts = {
            {{1, 2, 3}, {4, 5, 6}, {11, 3}},
            {{1, 2, 3}, {4, 5, 6, 7}, {9, 0}},
            {{1, 2, 3, 4}, {9}, {4, 5, 6}},
            {{1, 2, 3, 4}, {1}, {}},
            {{}, {1}, {1, 2, 3, 4}},
        };

        FOR_DISPATCH_3(a, b, c, ts) {
            TVector<std::tuple<i32, i32, i32>> e;
            for (size_t j = 0; j < a.size() && j < b.size() && j < c.size(); ++j) {
                e.push_back({a[j], b[j], c[j]});
            }

            TVector<std::tuple<i32, i32, i32>> f;
            FOR_DISPATCH_3(ai, bi, ci, Zip(a, b, c)) {
                f.push_back({ai, bi, ci});
            }

            EXPECT_EQ(e, f);
        }
    }

    TEST(FuncTools, CompileZip) {
        auto container = std::vector{1, 2, 3};
        TestViewCompileability(Zip(container));
        TestViewCompileability(Zip(container, container, container));
        const auto constContainer = std::vector{1, 2, 3};
        TestViewCompileability(Zip(constContainer, constContainer));
        const int arrayContainer[] = {1, 2, 3};
        TestViewCompileability(Zip(arrayContainer, arrayContainer));

        std::vector<std::pair<int, int>> res;
        FOR_DISPATCH_2(a, b, Zip(MakeMinimalisticContainer(), container)) {
            res.push_back({a, b});
        }
        EXPECT_EQ(res, (std::vector<std::pair<int, int>>{
            {0, 1}, {1, 2}, {2, 3},
        }));
    }

    TEST(FuncTools, Filter) {
        TVector<TVector<i32>> ts = {
            {},
            {1},
            {2},
            {1, 2},
            {2, 1},
            {1, 2, 3, 4, 5, 6, 7},
        };

        auto pred = [](i32 x) -> bool { return x & 1; };

        for (auto& a : ts) {
            TVector<i32> b;
            for (i32 x : a) {
                if (pred(x)) {
                    b.push_back(x);
                }
            }

            TVector<i32> c;
            for (i32 x : Filter(pred, a)) {
                c.push_back(x);
            }

            EXPECT_EQ(b, c);
        }
    }

    TEST(FuncTools, CompileFilter) {
        auto container = std::vector{1, 2, 3};
        auto isOdd = [](int x) { return bool(x & 1); };
        TestViewCompileability(Filter(isOdd, container));
        const int arrayContainer[] = {1, 2, 3};
        TestViewCompileability(Filter(isOdd, arrayContainer));
    }

    TEST(FuncTools, Map) {
        TVector<TVector<i32>> ts = {
            {},
            {1},
            {1, 2},
            {1, 2, 3, 4, 5, 6, 7},
        };

        auto f = [](i32 x) { return x * x; };

        for (auto& a : ts) {
            TVector<i32> b;
            for (i32 x : a) {
                b.push_back(f(x));
            }

            TVector<i32> c;
            for (i32 x : Map(f, a)) {
                c.push_back(x);
            }

            EXPECT_EQ(b, c);
        }

        TVector floats = {1.4, 4.1, 13.9};
        TVector ints = {1, 4, 13};
        TVector<float> roundedFloats = {1, 4, 13};
        TVector<int> res;
        TVector<float> resFloat;
        for (auto i : Map<int>(floats)) {
            res.push_back(i);
        }
        for (auto i : Map<float>(Map<int>(floats))) {
            resFloat.push_back(i);
        }
        EXPECT_EQ(ints, res);
        EXPECT_EQ(roundedFloats, resFloat);
    }

    TEST(FuncTools, CompileMap) {
        auto container = std::vector{1, 2, 3};
        auto sqr = [](int x) { return x * x; };
        TestViewCompileability(Map(sqr, container));
        const int arrayContainer[] = {1, 2, 3};
        TestViewCompileability(Map(sqr, arrayContainer));
    }

    TEST(FuncTools, MapRandomAccess) {
        auto sqr = [](int x) { return x * x; };
        {
            auto container = std::vector{1, 2, 3};
            auto mapped = Map(sqr, container);
            static_assert(
                std::is_same_v<decltype(mapped)::iterator::iterator_category, std::random_access_iterator_tag>
            );
        }
        {
            auto container = std::set<int>{1, 2, 3};
            auto mapped = Map(sqr, container);
            static_assert(
                std::is_same_v<decltype(mapped)::iterator::iterator_category, std::input_iterator_tag>
            );
        }
    }

    TEST(FuncTools, CartesianProduct) {
        TVector<std::pair<TVector<i32>, TVector<i32>>> ts = {
            {{1, 2, 3}, {4, 5, 6}},
            {{1, 2, 3}, {4, 5, 6, 7}},
            {{1, 2, 3, 4}, {4, 5, 6}},
            {{1, 2, 3, 4}, {}},
            {{}, {1, 2, 3, 4}},
        };

        for (auto [a, b] : ts) {
            TVector<std::pair<i32, i32>> c;
            for (auto ai : a) {
                for (auto bi : b) {
                    c.push_back({ai, bi});
                }
            }

            TVector<std::pair<i32, i32>> d;
            FOR_DISPATCH_2(ai, bi, CartesianProduct(a, b)) {
                d.push_back({ai, bi});
            }

            EXPECT_EQ(c, d);
        }

        {
            TVector<TVector<int>> g = {{}, {}};
            TVector h = {10, 11, 12};
            FOR_DISPATCH_2(gi, i, CartesianProduct(g, h)) {
                gi.push_back(i);
            }
            EXPECT_EQ(g[0], h);
            EXPECT_EQ(g[1], h);
        }
    }

    TEST(FuncTools, CartesianProduct3) {
        TVector<std::tuple<TVector<i32>, TVector<i32>, TVector<i32>>> ts = {
            {{1, 2, 3}, {4, 5, 6}, {11, 3}},
            {{1, 2, 3}, {4, 5, 6, 7}, {9}},
            {{1, 2, 3, 4}, {9}, {4, 5, 6}},
            {{1, 2, 3, 4}, {1}, {}},
            {{}, {1}, {1, 2, 3, 4}},
        };

        FOR_DISPATCH_3(a, b, c, ts) {
            TVector<std::tuple<i32, i32, i32>> e;
            for (auto ai : a) {
                for (auto bi : b) {
                    for (auto ci : c) {
                        e.push_back({ai, bi, ci});
                    }
                }
            }

            TVector<std::tuple<i32, i32, i32>> f;
            FOR_DISPATCH_3(ai, bi, ci, CartesianProduct(a, b, c)) {
                f.push_back({ai, bi, ci});
            }

            EXPECT_EQ(e, f);
        }
    }

    TEST(FuncTools, CompileCartesianProduct) {
        auto container = std::vector{1, 2, 3};
        TestViewCompileability(CartesianProduct(container, container));
        const auto constContainer = std::vector{1, 2, 3};
        TestViewCompileability(CartesianProduct(constContainer, constContainer));
        const int arrayContainer[] = {1, 2, 3};
        TestViewCompileability(CartesianProduct(arrayContainer, arrayContainer));

        std::vector<std::pair<int, int>> res;
        FOR_DISPATCH_2(a, b, CartesianProduct(MakeMinimalisticContainer(), MakeMinimalisticContainer())) {
            res.push_back({a, b});
        }
        EXPECT_EQ(res, (std::vector<std::pair<int, int>>{
            {0, 0}, {0, 1}, {0, 2},
            {1, 0}, {1, 1}, {1, 2},
            {2, 0}, {2, 1}, {2, 2},
        }));
    }

    TEST(FuncTools, Concatenate2) {
        TVector<std::pair<TVector<i32>, TVector<i32>>> ts = {
            {{1, 2, 3}, {4, 5, 6}},
            {{1, 2, 3}, {4, 5, 6, 7}},
            {{1, 2, 3, 4}, {4, 5, 6}},
            {{1, 2, 3, 4}, {}},
            {{}, {1, 2, 3, 4}},
        };

        for (auto [a, b] : ts) {
            TVector<i32> c;
            for (auto ai : a) {
                c.push_back(ai);
            }
            for (auto bi : b) {
                c.push_back(bi);
            }

            TVector<i32> d;
            for (auto x : Concatenate(a, b)) {
                d.push_back(x);
            }

            EXPECT_EQ(c, d);
        }

        {
            TVector<i32> a = {1, 2, 3, 4};
            TVector<i32> c;
            for (auto x : Concatenate(a, TVector<i32>{5, 6})) {
                c.push_back(x);
            }
            EXPECT_EQ(c, (TVector<i32>{1, 2, 3, 4, 5, 6}));
        }
    }

    TEST(FuncTools, CompileConcatenate) {
        auto container = std::vector{1, 2, 3};
        TestViewCompileability(Concatenate(container, container));
        const auto constContainer = std::vector{1, 2, 3};
        TestViewCompileability(Concatenate(constContainer, constContainer));
        const int arrayContainer[] = {1, 2, 3};
        TestViewCompileability(Concatenate(arrayContainer, arrayContainer));

        std::vector<int> res;
        for (auto a : Concatenate(MakeMinimalisticContainer(), MakeMinimalisticContainer())) {
            res.push_back(a);
        }
        EXPECT_EQ(res, (std::vector{0, 1, 2, 0, 1, 2}));
    }

    TEST(FuncTools, Combo) {
        FOR_DISPATCH_2(i, j, Enumerate(xrange(10u))) {
            EXPECT_EQ(i, j);
        }

        FOR_DISPATCH_2(i, jk, Enumerate(Enumerate(xrange(10u)))) {
            EXPECT_EQ(i, std::get<0>(jk));
            EXPECT_EQ(std::get<0>(jk), std::get<1>(jk));
        }

        TVector<size_t> a = {0, 1, 2};
        FOR_DISPATCH_2(i, j, Enumerate(Reversed(a))) {
            EXPECT_EQ(i, 2 - j);
        }

        FOR_DISPATCH_2(i, j, Enumerate(Map<float>(a))) {
            EXPECT_EQ(i, (size_t)j);
        }

        FOR_DISPATCH_2(i, j, Zip(a, Map<float>(a))) {
            EXPECT_EQ(i, (size_t)j);
        }

        auto mapper = [](auto&& x) {
            return std::get<0>(x) + std::get<1>(x);
        };
        FOR_DISPATCH_2(i, j, Zip(a, Map(mapper, Zip(a, a)))) {
            EXPECT_EQ(j, 2 * i);
        }
    }


    TEST(FuncTools, CopyIterator) {
        TVector a = {1, 2, 3, 4};
        TVector b = {4, 5, 6, 7};

        // calls f on 2nd, 3d and 4th positions (numeration from 1st)
        auto testIterator = [](auto it, auto f) {
            ++it;
            auto it2 = it;
            ++it2;
            ++it2;
            auto it3 = it;
            ++it3;
            f(*it, *it3, *it2);
        };

        {
            auto iterable = Enumerate(a);
            testIterator(std::begin(iterable),
                [](auto p2, auto p3, auto p4) {
                    EXPECT_EQ(std::get<0>(p2), 1u);
                    EXPECT_EQ(std::get<1>(p2), 2);
                    EXPECT_EQ(std::get<0>(p3), 2u);
                    EXPECT_EQ(std::get<1>(p3), 3);
                    EXPECT_EQ(std::get<0>(p4), 3u);
                    EXPECT_EQ(std::get<1>(p4), 4);
                });
        }

        {
            auto iterable = Map([](i32 x) { return x*x; }, a);
            testIterator(std::begin(iterable),
                [](auto p2, auto p3, auto p4) {
                    EXPECT_EQ(p2, 4);
                    EXPECT_EQ(p3, 9);
                    EXPECT_EQ(p4, 16);
                });
        }

        {
            auto iterable = Zip(a, b);
            testIterator(std::begin(iterable),
                [](auto p2, auto p3, auto p4) {
                    EXPECT_EQ(std::get<0>(p2), 2);
                    EXPECT_EQ(std::get<1>(p2), 5);
                    EXPECT_EQ(std::get<0>(p3), 3);
                    EXPECT_EQ(std::get<1>(p3), 6);
                    EXPECT_EQ(std::get<0>(p4), 4);
                    EXPECT_EQ(std::get<1>(p4), 7);
                });
        }

        {
            auto c = {1, 2, 3, 4, 5, 6, 7, 8};
            auto iterable = Filter([](i32 x) { return !(x & 1); }, c);
            testIterator(std::begin(iterable),
                [](auto p2, auto p3, auto p4) {
                    EXPECT_EQ(p2, 4);
                    EXPECT_EQ(p3, 6);
                    EXPECT_EQ(p4, 8);
                });
        }

        {
            auto iterable = CartesianProduct(TVector{0, 1}, TVector{2, 3});
            // (0, 2), (0, 3), (1, 2), (1, 3)
            testIterator(std::begin(iterable),
                [](auto p2, auto p3, auto p4) {
                    EXPECT_EQ(std::get<0>(p2), 0);
                    EXPECT_EQ(std::get<1>(p2), 3);
                    EXPECT_EQ(std::get<0>(p3), 1);
                    EXPECT_EQ(std::get<1>(p3), 2);
                    EXPECT_EQ(std::get<0>(p4), 1);
                    EXPECT_EQ(std::get<1>(p4), 3);
                });
        }
    }
