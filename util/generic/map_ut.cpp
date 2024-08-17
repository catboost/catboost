#include "map.h"

#include <library/cpp/testing/unittest/registar.h>
#include <util/memory/pool.h>
#include <algorithm>

Y_UNIT_TEST_SUITE(TYMapTest) {
    template <typename TAlloc>
    void DoTestMap1(TMap<char, int, TLess<char>, TAlloc>& m);

    template <typename TAlloc>
    void DoTestMMap1(TMultiMap<char, int, TLess<char>, TAlloc>& mm);

    Y_UNIT_TEST(TestMap1) {
        {
            TMap<char, int, TLess<char>> m;
            DoTestMap1(m);
        }
        {
            TMemoryPool p(100);
            TMap<char, int, TLess<char>, TPoolAllocator> m(&p);
            DoTestMap1(m);
        }
    }

    Y_UNIT_TEST(TestMMap1) {
        {
            TMultiMap<char, int, TLess<char>> mm;
            DoTestMMap1(mm);
        }
        {
            TMemoryPool p(100);
            TMultiMap<char, int, TLess<char>, TPoolAllocator> mm(&p);
            DoTestMMap1(mm);
        }
    }

    template <typename TAlloc>
    void DoTestMap1(TMap<char, int, TLess<char>, TAlloc>& m) {
        using maptype = TMap<char, int, TLess<char>, TAlloc>;
        // Store mappings between roman numerals and decimals.
        m['l'] = 50;
        m['x'] = 20; // Deliberate mistake.
        m['v'] = 5;
        m['i'] = 1;

        UNIT_ASSERT(m['x'] == 20);
        m['x'] = 10; // Correct mistake.
        UNIT_ASSERT(m['x'] == 10);
        UNIT_ASSERT(m['z'] == 0);

        UNIT_ASSERT(m.count('z') == 1);

        std::pair<typename maptype::iterator, bool> p = m.insert(std::pair<const char, int>('c', 100));

        UNIT_ASSERT(p.second);
        UNIT_ASSERT(p.first != m.end());
        UNIT_ASSERT((*p.first).first == 'c');
        UNIT_ASSERT((*p.first).second == 100);

        p = m.insert(std::pair<const char, int>('c', 100));

        UNIT_ASSERT(!p.second); // already existing pair
        UNIT_ASSERT(p.first != m.end());
        UNIT_ASSERT((*p.first).first == 'c');
        UNIT_ASSERT((*p.first).second == 100);
    }

    template <typename TAlloc>
    void DoTestMMap1(TMultiMap<char, int, TLess<char>, TAlloc>& m) {
        using mmap = TMultiMap<char, int, TLess<char>, TAlloc>;

        UNIT_ASSERT(m.count('X') == 0);

        m.insert(std::pair<const char, int>('X', 10)); // Standard way.
        UNIT_ASSERT(m.count('X') == 1);

        m.insert(std::pair<const char, int>('X', 20)); // jbuck: standard way
        UNIT_ASSERT(m.count('X') == 2);

        m.insert(std::pair<const char, int>('Y', 32)); // jbuck: standard way
        typename mmap::iterator i = m.find('X');       // Find first match.
        ++i;
        UNIT_ASSERT((*i).first == 'X');
        UNIT_ASSERT((*i).second == 20);
        ++i;
        UNIT_ASSERT((*i).first == 'Y');
        UNIT_ASSERT((*i).second == 32);
        ++i;
        UNIT_ASSERT(i == m.end());

        size_t count = m.erase('X');
        UNIT_ASSERT(count == 2);
    }

    Y_UNIT_TEST(TestMMap2) {
        using pair_type = std::pair<const int, char>;

        pair_type p1(3, 'c');
        pair_type p2(6, 'f');
        pair_type p3(1, 'a');
        pair_type p4(2, 'b');
        pair_type p5(3, 'x');
        pair_type p6(6, 'f');

        using mmap = TMultiMap<int, char, TLess<int>>;

        pair_type array[] = {
            p1,
            p2,
            p3,
            p4,
            p5,
            p6};

        mmap m(array + 0, array + 6);
        mmap::iterator i;
        i = m.lower_bound(3);
        UNIT_ASSERT((*i).first == 3);
        UNIT_ASSERT((*i).second == 'c');

        i = m.upper_bound(3);
        UNIT_ASSERT((*i).first == 6);
        UNIT_ASSERT((*i).second == 'f');
    }

    Y_UNIT_TEST(TestIterators) {
        using int_map = TMap<int, char, TLess<int>>;
        int_map imap;

        {
            int_map::iterator ite(imap.begin());
            int_map::const_iterator cite(imap.begin());

            UNIT_ASSERT(ite == cite);
            UNIT_ASSERT(!(ite != cite));
            UNIT_ASSERT(cite == ite);
            UNIT_ASSERT(!(cite != ite));
        }

        using mmap = TMultiMap<int, char, TLess<int>>;
        using pair_type = mmap::value_type;

        pair_type p1(3, 'c');
        pair_type p2(6, 'f');
        pair_type p3(1, 'a');
        pair_type p4(2, 'b');
        pair_type p5(3, 'x');
        pair_type p6(6, 'f');

        pair_type array[] = {
            p1,
            p2,
            p3,
            p4,
            p5,
            p6};

        mmap m(array + 0, array + 6);

        {
            mmap::iterator ite(m.begin());
            mmap::const_iterator cite(m.begin());
            // test compare between const_iterator and iterator
            UNIT_ASSERT(ite == cite);
            UNIT_ASSERT(!(ite != cite));
            UNIT_ASSERT(cite == ite);
            UNIT_ASSERT(!(cite != ite));
        }

        mmap::reverse_iterator ri = m.rbegin();

        UNIT_ASSERT(ri != m.rend());
        UNIT_ASSERT(ri == m.rbegin());
        UNIT_ASSERT((*ri).first == 6);
        UNIT_ASSERT((*ri++).second == 'f');
        UNIT_ASSERT((*ri).first == 6);
        UNIT_ASSERT((*ri).second == 'f');

        mmap const& cm = m;
        mmap::const_reverse_iterator rci = cm.rbegin();

        UNIT_ASSERT(rci != cm.rend());
        UNIT_ASSERT((*rci).first == 6);
        UNIT_ASSERT((*rci++).second == 'f');
        UNIT_ASSERT((*rci).first == 6);
        UNIT_ASSERT((*rci).second == 'f');
    }

    Y_UNIT_TEST(TestEqualRange) {
        using maptype = TMap<char, int, TLess<char>>;

        {
            maptype m;
            m['x'] = 10;

            std::pair<maptype::iterator, maptype::iterator> ret;
            ret = m.equal_range('x');
            UNIT_ASSERT(ret.first != ret.second);
            UNIT_ASSERT((*(ret.first)).first == 'x');
            UNIT_ASSERT((*(ret.first)).second == 10);
            UNIT_ASSERT(++(ret.first) == ret.second);
        }

        {
            {
                maptype m;

                maptype::iterator i = m.lower_bound('x');
                UNIT_ASSERT(i == m.end());

                i = m.upper_bound('x');
                UNIT_ASSERT(i == m.end());

                std::pair<maptype::iterator, maptype::iterator> ret;
                ret = m.equal_range('x');
                UNIT_ASSERT(ret.first == ret.second);
                UNIT_ASSERT(ret.first == m.end());
            }

            {
                const maptype m;
                std::pair<maptype::const_iterator, maptype::const_iterator> ret;
                ret = m.equal_range('x');
                UNIT_ASSERT(ret.first == ret.second);
                UNIT_ASSERT(ret.first == m.end());
            }
        }
    }

    struct TKey {
        TKey()
            : m_data(0)
        {
        }

        explicit TKey(int data)
            : m_data(data)
        {
        }

        int m_data;
    };

    struct TKeyCmp {
        bool operator()(TKey lhs, TKey rhs) const {
            return lhs.m_data < rhs.m_data;
        }

        bool operator()(TKey lhs, int rhs) const {
            return lhs.m_data < rhs;
        }

        bool operator()(int lhs, TKey rhs) const {
            return lhs < rhs.m_data;
        }

        using is_transparent = void;
    };

    struct TKeyCmpPtr {
        bool operator()(TKey const volatile* lhs, TKey const volatile* rhs) const {
            return (*lhs).m_data < (*rhs).m_data;
        }

        bool operator()(TKey const volatile* lhs, int rhs) const {
            return (*lhs).m_data < rhs;
        }

        bool operator()(int lhs, TKey const volatile* rhs) const {
            return lhs < (*rhs).m_data;
        }

        using is_transparent = void;
    };

    Y_UNIT_TEST(TestTemplateMethods) {
        {
            using Container = TMap<TKey, int, TKeyCmp>;
            using value = Container::value_type;

            Container cont;

            cont.insert(value(TKey(1), 1));
            cont.insert(value(TKey(2), 2));
            cont.insert(value(TKey(3), 3));
            cont.insert(value(TKey(4), 4));

            UNIT_ASSERT(cont.count(TKey(1)) == 1);
            UNIT_ASSERT(cont.count(1) == 1);
            UNIT_ASSERT(cont.count(5) == 0);

            UNIT_ASSERT(cont.find(2) != cont.end());
            UNIT_ASSERT(cont.lower_bound(2) != cont.end());
            UNIT_ASSERT(cont.upper_bound(2) != cont.end());
            UNIT_ASSERT(cont.equal_range(2) != std::make_pair(cont.begin(), cont.end()));

            Container const& ccont = cont;

            UNIT_ASSERT(ccont.find(2) != ccont.end());
            UNIT_ASSERT(ccont.lower_bound(2) != ccont.end());
            UNIT_ASSERT(ccont.upper_bound(2) != ccont.end());
            UNIT_ASSERT(ccont.equal_range(2) != std::make_pair(ccont.end(), ccont.end()));
        }

        {
            using Container = TMap<TKey*, int, TKeyCmpPtr>;
            using value = Container::value_type;

            Container cont;

            TKey key1(1), key2(2), key3(3), key4(4);

            cont.insert(value(&key1, 1));
            cont.insert(value(&key2, 2));
            cont.insert(value(&key3, 3));
            cont.insert(value(&key4, 4));

            UNIT_ASSERT(cont.count(1) == 1);
            UNIT_ASSERT(cont.count(5) == 0);

            UNIT_ASSERT(cont.find(2) != cont.end());
            UNIT_ASSERT(cont.lower_bound(2) != cont.end());
            UNIT_ASSERT(cont.upper_bound(2) != cont.end());
            UNIT_ASSERT(cont.equal_range(2) != std::make_pair(cont.begin(), cont.end()));

            Container const& ccont = cont;

            UNIT_ASSERT(ccont.find(2) != ccont.end());
            UNIT_ASSERT(ccont.lower_bound(2) != ccont.end());
            UNIT_ASSERT(ccont.upper_bound(2) != ccont.end());
            UNIT_ASSERT(ccont.equal_range(2) != std::make_pair(ccont.begin(), ccont.end()));
        }

        {
            using Container = TMultiMap<TKey, int, TKeyCmp>;
            using value = Container::value_type;

            Container cont;

            cont.insert(value(TKey(1), 1));
            cont.insert(value(TKey(2), 2));
            cont.insert(value(TKey(3), 3));
            cont.insert(value(TKey(4), 4));

            UNIT_ASSERT(cont.count(TKey(1)) == 1);
            UNIT_ASSERT(cont.count(1) == 1);
            UNIT_ASSERT(cont.count(5) == 0);

            UNIT_ASSERT(cont.find(2) != cont.end());
            UNIT_ASSERT(cont.lower_bound(2) != cont.end());
            UNIT_ASSERT(cont.upper_bound(2) != cont.end());
            UNIT_ASSERT(cont.equal_range(2) != std::make_pair(cont.begin(), cont.end()));

            Container const& ccont = cont;

            UNIT_ASSERT(ccont.find(2) != ccont.end());
            UNIT_ASSERT(ccont.lower_bound(2) != ccont.end());
            UNIT_ASSERT(ccont.upper_bound(2) != ccont.end());
            UNIT_ASSERT(ccont.equal_range(2) != std::make_pair(ccont.end(), ccont.end()));
        }

        {
            using Container = TMultiMap<TKey const volatile*, int, TKeyCmpPtr>;
            using value = Container::value_type;

            Container cont;

            TKey key1(1), key2(2), key3(3), key4(4);

            cont.insert(value(&key1, 1));
            cont.insert(value(&key2, 2));
            cont.insert(value(&key3, 3));
            cont.insert(value(&key4, 4));

            UNIT_ASSERT(cont.count(1) == 1);
            UNIT_ASSERT(cont.count(5) == 0);

            UNIT_ASSERT(cont.find(2) != cont.end());
            UNIT_ASSERT(cont.lower_bound(2) != cont.end());
            UNIT_ASSERT(cont.upper_bound(2) != cont.end());
            UNIT_ASSERT(cont.equal_range(2) != std::make_pair(cont.begin(), cont.end()));

            Container const& ccont = cont;

            UNIT_ASSERT(ccont.find(2) != ccont.end());
            UNIT_ASSERT(ccont.lower_bound(2) != ccont.end());
            UNIT_ASSERT(ccont.upper_bound(2) != ccont.end());
            UNIT_ASSERT(ccont.equal_range(2) != std::make_pair(ccont.begin(), ccont.end()));
        }
    }

    template <typename T>
    static void EmptyAndInsertTest(typename T::value_type v) {
        T c;
        UNIT_ASSERT(!c);
        c.insert(v);
        UNIT_ASSERT(c);
    }

    Y_UNIT_TEST(TestEmpty) {
        EmptyAndInsertTest<TMap<char, int, TLess<char>>>(std::pair<char, int>('a', 1));
        EmptyAndInsertTest<TMultiMap<char, int, TLess<char>>>(std::pair<char, int>('a', 1));
    }

    struct TParametrizedKeyCmp {
        bool Inverse;

        TParametrizedKeyCmp(bool inverse = false)
            : Inverse(inverse)
        {
        }

        bool operator()(TKey lhs, TKey rhs) const {
            if (Inverse) {
                return lhs.m_data > rhs.m_data;
            } else {
                return lhs.m_data < rhs.m_data;
            }
        }
    };

    Y_UNIT_TEST(TestMoveComparator) {
        using Container = TMultiMap<TKey, int, TParametrizedKeyCmp>;

        TParametrizedKeyCmp direct(false);
        TParametrizedKeyCmp inverse(true);

        {
            Container c(direct);
            c = Container(inverse);

            c.insert(std::make_pair(TKey(1), 101));
            c.insert(std::make_pair(TKey(2), 102));
            c.insert(std::make_pair(TKey(3), 103));

            TVector<int> values;
            for (auto& i : c) {
                values.push_back(i.second);
            }

            UNIT_ASSERT_VALUES_EQUAL(values.size(), 3);
            UNIT_ASSERT_VALUES_EQUAL(values[0], 103);
            UNIT_ASSERT_VALUES_EQUAL(values[1], 102);
            UNIT_ASSERT_VALUES_EQUAL(values[2], 101);
        }
    }

    Y_UNIT_TEST(TestMapInitializerList) {
        TMap<TString, int> m = {
            {"one", 1},
            {"two", 2},
            {"three", 3},
            {"four", 4},
        };

        UNIT_ASSERT_VALUES_EQUAL(m.size(), 4);
        UNIT_ASSERT_VALUES_EQUAL(m["one"], 1);
        UNIT_ASSERT_VALUES_EQUAL(m["two"], 2);
        UNIT_ASSERT_VALUES_EQUAL(m["three"], 3);
        UNIT_ASSERT_VALUES_EQUAL(m["four"], 4);
    }

    Y_UNIT_TEST(TestMMapInitializerList) {
        TMultiMap<TString, int> mm = {
            {"one", 1},
            {"two", 2},
            {"two", -2},
            {"three", 3},
        };
        UNIT_ASSERT(mm.contains("two"));
        TMultiMap<TString, int> expected;
        expected.emplace("one", 1);
        expected.emplace("two", 2);
        expected.emplace("two", -2);
        expected.emplace("three", 3);
        UNIT_ASSERT_VALUES_EQUAL(mm, expected);
    }

    Y_UNIT_TEST(TestMovePoolAlloc) {
        using TMapInPool = TMap<int, int, TLess<int>, TPoolAllocator>;

        TMemoryPool pool(1);

        TMapInPool m(&pool);
        m.emplace(0, 1);

        UNIT_ASSERT(m.contains(0));
        UNIT_ASSERT_VALUES_EQUAL(1, m[0]);

        TMapInPool movedM = std::move(m);

        UNIT_ASSERT(movedM.contains(0));
        UNIT_ASSERT_VALUES_EQUAL(1, movedM[0]);
    }
}
