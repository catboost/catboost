#include <library/cpp/containers/flat_hash/lib/iterator.h>
#include <library/cpp/containers/flat_hash/lib/containers.h>

#include <library/cpp/testing/unittest/registar.h>

#include <util/random/random.h>
#include <util/generic/algorithm.h>

using namespace NFlatHash;

namespace {
    constexpr size_t INIT_SIZE = 128;

    template <class Container>
    void SmokingTest(Container& cont) {
        using value_type = typename Container::value_type;
        using iterator = TIterator<Container, value_type>;
        using size_type = typename Container::size_type;

        iterator f(&cont), l(&cont, cont.Size());
        UNIT_ASSERT_EQUAL(f, l);
        UNIT_ASSERT_EQUAL((size_type)std::distance(f, l), cont.Taken());

        TVector<std::pair<size_type, value_type>> toAdd{
            { 0, (int)RandomNumber<size_type>(INIT_SIZE) },
            { 1 + RandomNumber<size_type>(INIT_SIZE - 2), (int)RandomNumber<size_type>(INIT_SIZE) },
            { INIT_SIZE - 1, (int)RandomNumber<size_type>(INIT_SIZE) }
        };

        for (const auto& p : toAdd) {
            UNIT_ASSERT(cont.IsEmpty(p.first));
            cont.InitNode(p.first, p.second);
        }
        UNIT_ASSERT_EQUAL(cont.Size(), INIT_SIZE);
        f = iterator{ &cont };
        l = iterator{ &cont, INIT_SIZE };
        UNIT_ASSERT_UNEQUAL(f, l);
        UNIT_ASSERT_EQUAL((size_type)std::distance(f, l), cont.Taken());

        TVector<value_type> added(f, l);
        UNIT_ASSERT(::Equal(toAdd.begin(), toAdd.end(), added.begin(), [](const auto& p, auto v) {
            return p.second == v;
        }));
    }

    template <class Container>
    void ConstTest(Container& cont) {
        using value_type = typename Container::value_type;
        using iterator = TIterator<Container, value_type>;
        using const_iterator = TIterator<const Container, const value_type>;

        iterator it{ &cont, INIT_SIZE / 2 };
        const_iterator cit1{ it };
        const_iterator cit2{ &cont, INIT_SIZE / 2 };

        UNIT_ASSERT_EQUAL(cit1, cit2);

        static_assert(std::is_same<decltype(*it), value_type&>::value);
        static_assert(std::is_same<decltype(*cit1), const value_type&>::value);
    }
}

Y_UNIT_TEST_SUITE(TIteratorTest) {
    Y_UNIT_TEST(SmokingTest) {
        {
            TFlatContainer<int> cont(INIT_SIZE);
            SmokingTest(cont);
        }
        {
            TDenseContainer<int, NSet::TStaticValueMarker<-1>> cont(INIT_SIZE);
            SmokingTest(cont);
        }
    }

    Y_UNIT_TEST(ConstTest) {
        {
            TFlatContainer<int> cont(INIT_SIZE);
            ConstTest(cont);
        }
        {
            TDenseContainer<int, NSet::TStaticValueMarker<-1>> cont(INIT_SIZE);
            ConstTest(cont);
        }
    }
}
