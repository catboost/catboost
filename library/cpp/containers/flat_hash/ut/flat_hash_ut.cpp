#include <library/cpp/containers/flat_hash/flat_hash.h>

#include <library/cpp/testing/unittest/registar.h>

using namespace NFH;

namespace {

constexpr size_t TEST_INIT_SIZE = 25;
constexpr std::initializer_list<int> SET_INPUT_SAMPLE{1, 2, 3, 4, 5};
const std::initializer_list<std::pair<const int, TString>> MAP_INPUT_SAMPLE{
    {1, "a"},
    {2, "b"},
    {3, "c"},
    {4, "d"},
    {5, "e"}
};

}  // namespace

template <class Map>
class TMapTest : public TTestBase {
    void AllocatorTest();

    void SmokingTest() {
        Map mp;
        mp.emplace(5, "abc");

        UNIT_ASSERT_EQUAL(mp.size(), 1);
        UNIT_ASSERT(mp.contains(5));

        auto it = mp.find(5);
        UNIT_ASSERT_EQUAL(mp.begin(), it);
        UNIT_ASSERT(!mp.empty());
    }

    void CopyConstructionTest() {
        Map st(MAP_INPUT_SAMPLE);
        auto st2 = st;

        UNIT_ASSERT(!st.empty());
        UNIT_ASSERT(!st2.empty());
        UNIT_ASSERT_EQUAL(st, st2);
    }

    void MoveConstructionTest() {
        Map st(MAP_INPUT_SAMPLE);
        auto st2 = std::move(st);

        UNIT_ASSERT(st.empty());
        UNIT_ASSERT(!st2.empty());
        UNIT_ASSERT_UNEQUAL(st, st2);
    }

    void CopyAssignmentTest() {
        Map st(MAP_INPUT_SAMPLE);
        Map st2;
        UNIT_ASSERT_UNEQUAL(st, st2);
        UNIT_ASSERT(st2.empty());

        st2 = st;
        UNIT_ASSERT_EQUAL(st, st2);
        UNIT_ASSERT(!st2.empty());
    }

    void DoubleCopyAssignmentTest() {
        Map st(MAP_INPUT_SAMPLE);
        Map st2;
        UNIT_ASSERT_UNEQUAL(st, st2);
        UNIT_ASSERT(st2.empty());

        st2 = st;
        UNIT_ASSERT_EQUAL(st, st2);
        UNIT_ASSERT(!st2.empty());

        st2 = st;
        UNIT_ASSERT_EQUAL(st, st2);
        UNIT_ASSERT(!st2.empty());
    }

    void MoveAssignmentTest() {
        Map st(MAP_INPUT_SAMPLE);
        Map st2;
        UNIT_ASSERT_UNEQUAL(st, st2);
        UNIT_ASSERT(st2.empty());

        st2 = std::move(st);
        UNIT_ASSERT_UNEQUAL(st, st2);
        UNIT_ASSERT(!st2.empty());
        UNIT_ASSERT(st.empty());
    }

    void InsertOrAssignTest() {
        Map mp;

        auto p = mp.insert_or_assign(5, "abc");
        UNIT_ASSERT_EQUAL(p.first, mp.begin());
        UNIT_ASSERT(p.second);
        UNIT_ASSERT_EQUAL(p.first->first, 5);
        UNIT_ASSERT_EQUAL(p.first->second, "abc");

        auto p2 = mp.insert_or_assign(5, "def");
        UNIT_ASSERT_EQUAL(p.first, p2.first);
        UNIT_ASSERT(!p2.second);
        UNIT_ASSERT_EQUAL(p2.first->first, 5);
        UNIT_ASSERT_EQUAL(p2.first->second, "def");
    }

    void TryEmplaceTest() {
        Map mp;

        auto p = mp.try_emplace(5, "abc");
        UNIT_ASSERT_EQUAL(p.first, mp.begin());
        UNIT_ASSERT(p.second);
        UNIT_ASSERT_EQUAL(p.first->first, 5);
        UNIT_ASSERT_EQUAL(p.first->second, "abc");

        auto p2 = mp.try_emplace(5, "def");
        UNIT_ASSERT_EQUAL(p.first, p2.first);
        UNIT_ASSERT(!p2.second);
        UNIT_ASSERT_EQUAL(p2.first->first, 5);
        UNIT_ASSERT_EQUAL(p.first->second, "abc");
    }

    UNIT_TEST_SUITE_DEMANGLE(TMapTest);
    UNIT_TEST(AllocatorTest);
    UNIT_TEST(SmokingTest);
    UNIT_TEST(CopyConstructionTest);
    UNIT_TEST(MoveConstructionTest);
    UNIT_TEST(CopyAssignmentTest);
    UNIT_TEST(DoubleCopyAssignmentTest);
    UNIT_TEST(MoveAssignmentTest);
    UNIT_TEST(InsertOrAssignTest);
    UNIT_TEST(TryEmplaceTest);
    UNIT_TEST_SUITE_END();
};

template <>
void TMapTest<TFlatHashMap<int, TString>>::AllocatorTest() {
    using Map = TFlatHashMap<int, TString>;
    Map mp(3, typename Map::allocator_type());
}

template <>
void TMapTest<TDenseHashMapStaticMarker<int, TString, -1>>::AllocatorTest() {
    using Map = TDenseHashMapStaticMarker<int, TString, -1>;
    Map mp(3, NFlatHash::NMap::TStaticValueMarker<-1, TString>(), typename Map::allocator_type());
}

using TFlatHashMapTest = TMapTest<TFlatHashMap<int, TString>>;
using TDenseHashMapTest = TMapTest<TDenseHashMapStaticMarker<int, TString, -1>>;

UNIT_TEST_SUITE_REGISTRATION(TFlatHashMapTest);
UNIT_TEST_SUITE_REGISTRATION(TDenseHashMapTest);


template <class Set>
class TSetTest : public TTestBase {
    void AllocatorTest();
    void DefaultConstructTest() {
        Set st;

        UNIT_ASSERT(st.empty());
        UNIT_ASSERT_EQUAL(st.size(), 0);
        UNIT_ASSERT(st.bucket_count() > 0);
        UNIT_ASSERT_EQUAL(st.begin(), st.end());
        UNIT_ASSERT(st.load_factor() < std::numeric_limits<float>::epsilon());
    }

    void InitCapacityConstructTest() {
        Set st(TEST_INIT_SIZE);

        UNIT_ASSERT(st.empty());
        UNIT_ASSERT_EQUAL(st.size(), 0);
        UNIT_ASSERT(st.bucket_count() >= TEST_INIT_SIZE);
        UNIT_ASSERT_EQUAL(st.begin(), st.end());
        UNIT_ASSERT(st.load_factor() < std::numeric_limits<float>::epsilon());
    }

    void IteratorsConstructTest() {
        Set st(SET_INPUT_SAMPLE.begin(), SET_INPUT_SAMPLE.end());

        UNIT_ASSERT(!st.empty());
        UNIT_ASSERT_EQUAL(st.size(), SET_INPUT_SAMPLE.size());
        UNIT_ASSERT(st.bucket_count() >= st.size());
        UNIT_ASSERT_UNEQUAL(st.begin(), st.end());
        UNIT_ASSERT_EQUAL(static_cast<size_t>(std::distance(st.begin(), st.end())), st.size());
        UNIT_ASSERT(st.load_factor() > 0);
    }

    void InitializerListConstructTest() {
        Set st(SET_INPUT_SAMPLE);

        UNIT_ASSERT(!st.empty());
        UNIT_ASSERT(st.size() > 0);
        UNIT_ASSERT(st.bucket_count() > 0);
        UNIT_ASSERT_UNEQUAL(st.begin(), st.end());
        UNIT_ASSERT_EQUAL(static_cast<size_t>(std::distance(st.begin(), st.end())), st.size());
        UNIT_ASSERT(st.load_factor() > 0);
    }

    void CopyConstructionTest() {
        Set st(SET_INPUT_SAMPLE);
        auto st2 = st;

        UNIT_ASSERT(!st.empty());
        UNIT_ASSERT(!st2.empty());
        UNIT_ASSERT_EQUAL(st, st2);
    }

    void MoveConstructionTest() {
        Set st(SET_INPUT_SAMPLE);
        auto st2 = std::move(st);

        UNIT_ASSERT(st.empty());
        UNIT_ASSERT(!st2.empty());
        UNIT_ASSERT_UNEQUAL(st, st2);
    }

    void CopyAssignmentTest() {
        Set st(SET_INPUT_SAMPLE);
        Set st2;
        UNIT_ASSERT_UNEQUAL(st, st2);
        UNIT_ASSERT(st2.empty());

        st2 = st;
        UNIT_ASSERT_EQUAL(st, st2);
        UNIT_ASSERT(!st2.empty());
    }

    void MoveAssignmentTest() {
        Set st(SET_INPUT_SAMPLE);
        Set st2;
        UNIT_ASSERT_UNEQUAL(st, st2);
        UNIT_ASSERT(st2.empty());

        st2 = std::move(st);
        UNIT_ASSERT_UNEQUAL(st, st2);
        UNIT_ASSERT(!st2.empty());
        UNIT_ASSERT(st.empty());
    }

    UNIT_TEST_SUITE_DEMANGLE(TSetTest);
    UNIT_TEST(AllocatorTest);
    UNIT_TEST(DefaultConstructTest);
    UNIT_TEST(InitCapacityConstructTest);
    UNIT_TEST(IteratorsConstructTest);
    UNIT_TEST(InitializerListConstructTest);
    UNIT_TEST(CopyConstructionTest);
    UNIT_TEST(MoveConstructionTest);
    UNIT_TEST(CopyAssignmentTest);
    UNIT_TEST(MoveAssignmentTest);
    UNIT_TEST_SUITE_END();
};

template <>
void TSetTest<TFlatHashSet<int>>::AllocatorTest() {
    using Map = TFlatHashSet<int>;
    Map mp(3, typename Map::allocator_type());
}

template <>
void TSetTest<TDenseHashSetStaticMarker<int, -1>>::AllocatorTest() {
    using Map = TDenseHashSetStaticMarker<int, -1>;
    Map mp(3, NFlatHash::NSet::TStaticValueMarker<-1>(), typename Map::allocator_type());
}

using TFlatHashSetTest = TSetTest<TFlatHashSet<int, THash<int>>>;
using TDenseHashSetTest = TSetTest<TDenseHashSetStaticMarker<int, -1>>;

UNIT_TEST_SUITE_REGISTRATION(TFlatHashSetTest);
UNIT_TEST_SUITE_REGISTRATION(TDenseHashSetTest);
