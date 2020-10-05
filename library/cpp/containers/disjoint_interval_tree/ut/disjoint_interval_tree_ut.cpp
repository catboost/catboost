#include <library/cpp/testing/unittest/registar.h>

#include <library/cpp/containers/disjoint_interval_tree/disjoint_interval_tree.h>

Y_UNIT_TEST_SUITE(DisjointIntervalTreeTest) {
    Y_UNIT_TEST(GenericTest) {
        TDisjointIntervalTree<ui64> tree;
        tree.Insert(1);
        tree.Insert(50);
        UNIT_ASSERT_VALUES_EQUAL(tree.GetNumIntervals(), 2);
        UNIT_ASSERT_VALUES_EQUAL(tree.GetNumElements(), 2);

        tree.InsertInterval(10, 30);
        UNIT_ASSERT_VALUES_EQUAL(tree.GetNumIntervals(), 3);
        UNIT_ASSERT_VALUES_EQUAL(tree.GetNumElements(), 22);

        UNIT_ASSERT_VALUES_EQUAL(tree.Min(), 1);
        UNIT_ASSERT_VALUES_EQUAL(tree.Max(), 51);

        tree.Erase(20);
        UNIT_ASSERT_VALUES_EQUAL(tree.GetNumIntervals(), 4);
        UNIT_ASSERT_VALUES_EQUAL(tree.GetNumElements(), 21);

        tree.Clear();
        UNIT_ASSERT_VALUES_EQUAL(tree.GetNumIntervals(), 0);
        UNIT_ASSERT_VALUES_EQUAL(tree.GetNumElements(), 0);
    }

    Y_UNIT_TEST(MergeIntervalsTest) {
        TDisjointIntervalTree<ui64> tree;
        tree.Insert(1);
        tree.Insert(2);

        UNIT_ASSERT_VALUES_EQUAL(tree.GetNumIntervals(), 1);
        UNIT_ASSERT_VALUES_EQUAL(tree.GetNumElements(), 2);

        auto begin = tree.begin();
        UNIT_ASSERT_VALUES_EQUAL(begin->first, 1);
        UNIT_ASSERT_VALUES_EQUAL(begin->second, 3);

        ++begin;
        UNIT_ASSERT_EQUAL(begin, tree.end());
    }
}
