#include "set.h"

#include <library/cpp/testing/unittest/registar.h>

#include <utility>

#include <algorithm>

Y_UNIT_TEST_SUITE(YSetTest) {
    Y_UNIT_TEST(TestSet1) {
        TSet<int, TLess<int>> s;
        UNIT_ASSERT(!s);
        UNIT_ASSERT(s.count(42) == 0);
        s.insert(42);
        UNIT_ASSERT(s);
        UNIT_ASSERT(s.count(42) == 1);
        s.insert(42);
        UNIT_ASSERT(s.count(42) == 1);
        size_t count = s.erase(42);
        UNIT_ASSERT(count == 1);
    }

    Y_UNIT_TEST(TestSet2) {
        using int_set = TSet<int, TLess<int>>;
        int_set s;
        std::pair<int_set::iterator, bool> p = s.insert(42);
        UNIT_ASSERT(p.second == true);
        p = s.insert(42);
        UNIT_ASSERT(p.second == false);

        int array1[] = {1, 3, 6, 7};
        s.insert(array1, array1 + 4);
        UNIT_ASSERT(distance(s.begin(), s.end()) == 5);

        int_set s2;
        s2.swap(s);
        UNIT_ASSERT(distance(s2.begin(), s2.end()) == 5);
        UNIT_ASSERT(distance(s.begin(), s.end()) == 0);

        int_set s3;
        s3.swap(s);
        s3.swap(s2);
        UNIT_ASSERT(distance(s.begin(), s.end()) == 0);
        UNIT_ASSERT(distance(s2.begin(), s2.end()) == 0);
        UNIT_ASSERT(distance(s3.begin(), s3.end()) == 5);
    }

    Y_UNIT_TEST(TestErase) {
        TSet<int, TLess<int>> s;
        s.insert(1);
        s.erase(s.begin());
        UNIT_ASSERT(s.empty());

        size_t nb = s.erase(1);
        UNIT_ASSERT(nb == 0);
    }

    Y_UNIT_TEST(TestInsert) {
        TSet<int> s;
        TSet<int>::iterator i = s.insert(s.end(), 0);
        UNIT_ASSERT(*i == 0);
    }

    Y_UNIT_TEST(TestFind) {
        TSet<int> s;

        UNIT_ASSERT(s.find(0) == s.end());

        TSet<int> const& crs = s;

        UNIT_ASSERT(crs.find(0) == crs.end());
    }

    Y_UNIT_TEST(TestHas) {
        TSet<int> s;
        UNIT_ASSERT(!s.contains(0));

        TSet<int> const& crs = s;
        UNIT_ASSERT(!crs.contains(0));

        s.insert(1);
        s.insert(42);
        s.insert(100);
        s.insert(2);

        UNIT_ASSERT(s.contains(1));
        UNIT_ASSERT(s.contains(2));
        UNIT_ASSERT(s.contains(42));
        UNIT_ASSERT(s.contains(100));
    }

    Y_UNIT_TEST(TestBounds) {
        int array1[] = {1, 3, 6, 7};
        TSet<int> s(array1, array1 + sizeof(array1) / sizeof(array1[0]));
        TSet<int> const& crs = s;

        TSet<int>::iterator sit;
        TSet<int>::const_iterator scit;
        std::pair<TSet<int>::iterator, TSet<int>::iterator> pit;
        std::pair<TSet<int>::const_iterator, TSet<int>::const_iterator> pcit;

        // Check iterator on mutable set
        sit = s.lower_bound(2);
        UNIT_ASSERT(sit != s.end());
        UNIT_ASSERT(*sit == 3);

        sit = s.upper_bound(5);
        UNIT_ASSERT(sit != s.end());
        UNIT_ASSERT(*sit == 6);

        pit = s.equal_range(6);
        UNIT_ASSERT(pit.first != pit.second);
        UNIT_ASSERT(pit.first != s.end());
        UNIT_ASSERT(*pit.first == 6);
        UNIT_ASSERT(pit.second != s.end());
        UNIT_ASSERT(*pit.second == 7);

        pit = s.equal_range(4);
        UNIT_ASSERT(pit.first == pit.second);
        UNIT_ASSERT(pit.first != s.end());
        UNIT_ASSERT(*pit.first == 6);
        UNIT_ASSERT(pit.second != s.end());
        UNIT_ASSERT(*pit.second == 6);

        // Check const_iterator on mutable set
        scit = s.lower_bound(2);
        UNIT_ASSERT(scit != s.end());
        UNIT_ASSERT(*scit == 3);

        scit = s.upper_bound(5);
        UNIT_ASSERT(scit != s.end());
        UNIT_ASSERT(*scit == 6);

        pcit = s.equal_range(6);
        UNIT_ASSERT(pcit.first != pcit.second);
        UNIT_ASSERT(pcit.first != s.end());
        UNIT_ASSERT(*pcit.first == 6);
        UNIT_ASSERT(pcit.second != s.end());
        UNIT_ASSERT(*pcit.second == 7);

        // Check const_iterator on const set
        scit = crs.lower_bound(2);
        UNIT_ASSERT(scit != crs.end());
        UNIT_ASSERT(*scit == 3);

        scit = crs.upper_bound(5);
        UNIT_ASSERT(scit != crs.end());
        UNIT_ASSERT(*scit == 6);

        pcit = crs.equal_range(6);
        UNIT_ASSERT(pcit.first != pcit.second);
        UNIT_ASSERT(pcit.first != crs.end());
        UNIT_ASSERT(*pcit.first == 6);
        UNIT_ASSERT(pcit.second != crs.end());
        UNIT_ASSERT(*pcit.second == 7);
    }

    Y_UNIT_TEST(TestImplementationCheck) {
        TSet<int> tree;
        tree.insert(1);
        TSet<int>::iterator it = tree.begin();
        int const& int_ref = *it++;
        UNIT_ASSERT(int_ref == 1);

        UNIT_ASSERT(it == tree.end());
        UNIT_ASSERT(it != tree.begin());

        TSet<int>::const_iterator cit = tree.begin();
        int const& int_cref = *cit++;
        UNIT_ASSERT(int_cref == 1);
    }

    Y_UNIT_TEST(TestReverseIteratorTest) {
        TSet<int> tree;
        tree.insert(1);
        tree.insert(2);

        {
            TSet<int>::reverse_iterator rit(tree.rbegin());
            UNIT_ASSERT(*(rit++) == 2);
            UNIT_ASSERT(*(rit++) == 1);
            UNIT_ASSERT(rit == tree.rend());
        }

        {
            TSet<int> const& ctree = tree;
            TSet<int>::const_reverse_iterator rit(ctree.rbegin());
            UNIT_ASSERT(*(rit++) == 2);
            UNIT_ASSERT(*(rit++) == 1);
            UNIT_ASSERT(rit == ctree.rend());
        }
    }

    Y_UNIT_TEST(TestConstructorsAndAssignments) {
        {
            using container = TSet<int>;

            container c1;
            c1.insert(100);
            c1.insert(200);

            container c2(c1);

            UNIT_ASSERT_VALUES_EQUAL(2, c1.size());
            UNIT_ASSERT_VALUES_EQUAL(2, c2.size());
            UNIT_ASSERT(c1.contains(100));
            UNIT_ASSERT(c2.contains(200));

            container c3(std::move(c1));

            UNIT_ASSERT_VALUES_EQUAL(0, c1.size());
            UNIT_ASSERT_VALUES_EQUAL(2, c3.size());
            UNIT_ASSERT(c3.contains(100));

            c2.insert(300);
            c3 = c2;

            UNIT_ASSERT_VALUES_EQUAL(3, c2.size());
            UNIT_ASSERT_VALUES_EQUAL(3, c3.size());
            UNIT_ASSERT(c3.contains(300));

            c2.insert(400);
            c3 = std::move(c2);

            UNIT_ASSERT_VALUES_EQUAL(0, c2.size());
            UNIT_ASSERT_VALUES_EQUAL(4, c3.size());
            UNIT_ASSERT(c3.contains(400));
        }

        {
            using container = TMultiSet<int>;

            container c1;
            c1.insert(100);
            c1.insert(200);

            container c2(c1);

            UNIT_ASSERT_VALUES_EQUAL(2, c1.size());
            UNIT_ASSERT_VALUES_EQUAL(2, c2.size());
            UNIT_ASSERT(c1.find(100) != c1.end());
            UNIT_ASSERT(c2.find(200) != c2.end());

            container c3(std::move(c1));

            UNIT_ASSERT_VALUES_EQUAL(0, c1.size());
            UNIT_ASSERT_VALUES_EQUAL(2, c3.size());
            UNIT_ASSERT(c3.find(100) != c3.end());

            c2.insert(300);
            c3 = c2;

            UNIT_ASSERT_VALUES_EQUAL(3, c2.size());
            UNIT_ASSERT_VALUES_EQUAL(3, c3.size());
            UNIT_ASSERT(c3.find(300) != c3.end());

            c2.insert(400);
            c3 = std::move(c2);

            UNIT_ASSERT_VALUES_EQUAL(0, c2.size());
            UNIT_ASSERT_VALUES_EQUAL(4, c3.size());
            UNIT_ASSERT(c3.find(400) != c3.end());
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
            using KeySet = TSet<TKey, TKeyCmp>;
            KeySet keySet;
            keySet.insert(TKey(1));
            keySet.insert(TKey(2));
            keySet.insert(TKey(3));
            keySet.insert(TKey(4));

            UNIT_ASSERT(keySet.count(TKey(1)) == 1);
            UNIT_ASSERT(keySet.count(1) == 1);
            UNIT_ASSERT(keySet.count(5) == 0);

            UNIT_ASSERT(keySet.find(2) != keySet.end());
            UNIT_ASSERT(keySet.lower_bound(2) != keySet.end());
            UNIT_ASSERT(keySet.upper_bound(2) != keySet.end());
            UNIT_ASSERT(keySet.equal_range(2) != std::make_pair(keySet.begin(), keySet.end()));

            KeySet const& ckeySet = keySet;
            UNIT_ASSERT(ckeySet.find(2) != ckeySet.end());
            UNIT_ASSERT(ckeySet.lower_bound(2) != ckeySet.end());
            UNIT_ASSERT(ckeySet.upper_bound(2) != ckeySet.end());
            UNIT_ASSERT(ckeySet.equal_range(2) != std::make_pair(ckeySet.begin(), ckeySet.end()));
        }

        {
            using KeySet = TSet<TKey*, TKeyCmpPtr>;
            KeySet keySet;
            TKey key1(1), key2(2), key3(3), key4(4);
            keySet.insert(&key1);
            keySet.insert(&key2);
            keySet.insert(&key3);
            keySet.insert(&key4);

            UNIT_ASSERT(keySet.count(1) == 1);
            UNIT_ASSERT(keySet.count(5) == 0);

            UNIT_ASSERT(keySet.find(2) != keySet.end());
            UNIT_ASSERT(keySet.lower_bound(2) != keySet.end());
            UNIT_ASSERT(keySet.upper_bound(2) != keySet.end());
            UNIT_ASSERT(keySet.equal_range(2) != std::make_pair(keySet.begin(), keySet.end()));

            KeySet const& ckeySet = keySet;
            UNIT_ASSERT(ckeySet.find(2) != ckeySet.end());
            UNIT_ASSERT(ckeySet.lower_bound(2) != ckeySet.end());
            UNIT_ASSERT(ckeySet.upper_bound(2) != ckeySet.end());
            UNIT_ASSERT(ckeySet.equal_range(2) != std::make_pair(ckeySet.begin(), ckeySet.end()));
        }
        {
            using KeySet = TMultiSet<TKey, TKeyCmp>;
            KeySet keySet;
            keySet.insert(TKey(1));
            keySet.insert(TKey(2));
            keySet.insert(TKey(3));
            keySet.insert(TKey(4));

            UNIT_ASSERT(keySet.count(TKey(1)) == 1);
            UNIT_ASSERT(keySet.count(1) == 1);
            UNIT_ASSERT(keySet.count(5) == 0);

            UNIT_ASSERT(keySet.find(2) != keySet.end());
            UNIT_ASSERT(keySet.lower_bound(2) != keySet.end());
            UNIT_ASSERT(keySet.upper_bound(2) != keySet.end());
            UNIT_ASSERT(keySet.equal_range(2) != std::make_pair(keySet.begin(), keySet.end()));

            KeySet const& ckeySet = keySet;
            UNIT_ASSERT(ckeySet.find(2) != ckeySet.end());
            UNIT_ASSERT(ckeySet.lower_bound(2) != ckeySet.end());
            UNIT_ASSERT(ckeySet.upper_bound(2) != ckeySet.end());
            UNIT_ASSERT(ckeySet.equal_range(2) != std::make_pair(ckeySet.begin(), ckeySet.end()));
        }

        {
            using KeySet = TMultiSet<TKey const volatile*, TKeyCmpPtr>;
            KeySet keySet;
            TKey key1(1), key2(2), key3(3), key4(4);
            keySet.insert(&key1);
            keySet.insert(&key2);
            keySet.insert(&key3);
            keySet.insert(&key4);

            UNIT_ASSERT(keySet.count(1) == 1);
            UNIT_ASSERT(keySet.count(5) == 0);

            UNIT_ASSERT(keySet.find(2) != keySet.end());
            UNIT_ASSERT(keySet.lower_bound(2) != keySet.end());
            UNIT_ASSERT(keySet.upper_bound(2) != keySet.end());
            UNIT_ASSERT(keySet.equal_range(2) != std::make_pair(keySet.begin(), keySet.end()));

            KeySet const& ckeySet = keySet;
            UNIT_ASSERT(ckeySet.find(2) != ckeySet.end());
            UNIT_ASSERT(ckeySet.lower_bound(2) != ckeySet.end());
            UNIT_ASSERT(ckeySet.upper_bound(2) != ckeySet.end());
            UNIT_ASSERT(ckeySet.equal_range(2) != std::make_pair(ckeySet.begin(), ckeySet.end()));
        }
    }
} // Y_UNIT_TEST_SUITE(YSetTest)
