#include "skiplist.h"

#include <library/cpp/testing/unittest/registar.h>

namespace NThreading {
    namespace {
        struct TTestObject {
            static size_t Count;
            int Tag;

            TTestObject(int tag)
                : Tag(tag)
            {
                ++Count;
            }

            TTestObject(const TTestObject& other)
                : Tag(other.Tag)
            {
                ++Count;
            }

            ~TTestObject() {
                --Count;
            }

            bool operator<(const TTestObject& other) const {
                return Tag < other.Tag;
            }
        };

        size_t TTestObject::Count = 0;

    }

    ////////////////////////////////////////////////////////////////////////////////

    Y_UNIT_TEST_SUITE(TSkipListTest) {
        Y_UNIT_TEST(ShouldBeEmptyAfterCreation) {
            TMemoryPool pool(1024);
            TSkipList<int> list(pool);

            UNIT_ASSERT_EQUAL(list.GetSize(), 0);
        }

        Y_UNIT_TEST(ShouldAllowInsertion) {
            TMemoryPool pool(1024);
            TSkipList<int> list(pool);

            UNIT_ASSERT(list.Insert(12345678));
            UNIT_ASSERT_EQUAL(list.GetSize(), 1);
        }

        Y_UNIT_TEST(ShouldNotAllowDuplicates) {
            TMemoryPool pool(1024);
            TSkipList<int> list(pool);

            UNIT_ASSERT(list.Insert(12345678));
            UNIT_ASSERT_EQUAL(list.GetSize(), 1);

            UNIT_ASSERT(!list.Insert(12345678));
            UNIT_ASSERT_EQUAL(list.GetSize(), 1);
        }

        Y_UNIT_TEST(ShouldContainInsertedItem) {
            TMemoryPool pool(1024);
            TSkipList<int> list(pool);

            UNIT_ASSERT(list.Insert(12345678));
            UNIT_ASSERT(list.Contains(12345678));
        }

        Y_UNIT_TEST(ShouldNotContainNotInsertedItem) {
            TMemoryPool pool(1024);
            TSkipList<int> list(pool);

            UNIT_ASSERT(list.Insert(12345678));
            UNIT_ASSERT(!list.Contains(87654321));
        }

        Y_UNIT_TEST(ShouldIterateAllItems) {
            TMemoryPool pool(1024);
            TSkipList<int> list(pool);

            for (int i = 8; i > 0; --i) {
                UNIT_ASSERT(list.Insert(i));
            }

            TSkipList<int>::TIterator it = list.SeekToFirst();
            for (int i = 1; i <= 8; ++i) {
                UNIT_ASSERT(it.IsValid());
                UNIT_ASSERT_EQUAL(it.GetValue(), i);
                it.Next();
            }
            UNIT_ASSERT(!it.IsValid());
        }

        Y_UNIT_TEST(ShouldIterateAllItemsInReverseDirection) {
            TMemoryPool pool(1024);
            TSkipList<int> list(pool);

            for (int i = 8; i > 0; --i) {
                UNIT_ASSERT(list.Insert(i));
            }

            TSkipList<int>::TIterator it = list.SeekToLast();
            for (int i = 8; i > 0; --i) {
                UNIT_ASSERT(it.IsValid());
                UNIT_ASSERT_EQUAL(it.GetValue(), i);
                it.Prev();
            }
            UNIT_ASSERT(!it.IsValid());
        }

        Y_UNIT_TEST(ShouldSeekToFirstItem) {
            TMemoryPool pool(1024);
            TSkipList<int> list(pool);

            for (int i = 1; i < 10; ++i) {
                UNIT_ASSERT(list.Insert(i));
            }

            TSkipList<int>::TIterator it = list.SeekToFirst();
            UNIT_ASSERT(it.IsValid());
            UNIT_ASSERT_EQUAL(it.GetValue(), 1);
        }

        Y_UNIT_TEST(ShouldSeekToLastItem) {
            TMemoryPool pool(1024);
            TSkipList<int> list(pool);

            for (int i = 1; i < 10; ++i) {
                UNIT_ASSERT(list.Insert(i));
            }

            TSkipList<int>::TIterator it = list.SeekToLast();
            UNIT_ASSERT(it.IsValid());
            UNIT_ASSERT_EQUAL(it.GetValue(), 9);
        }

        Y_UNIT_TEST(ShouldSeekToExistingItem) {
            TMemoryPool pool(1024);
            TSkipList<int> list(pool);

            UNIT_ASSERT(list.Insert(12345678));

            TSkipList<int>::TIterator it = list.SeekTo(12345678);
            UNIT_ASSERT(it.IsValid());
        }

        Y_UNIT_TEST(ShouldSeekAfterMissedItem) {
            TMemoryPool pool(1024);
            TSkipList<int> list(pool);

            UNIT_ASSERT(list.Insert(100));
            UNIT_ASSERT(list.Insert(300));

            TSkipList<int>::TIterator it = list.SeekTo(200);
            UNIT_ASSERT(it.IsValid());
            UNIT_ASSERT_EQUAL(it.GetValue(), 300);

            it.Prev();
            UNIT_ASSERT(it.IsValid());
            UNIT_ASSERT_EQUAL(it.GetValue(), 100);
        }

        Y_UNIT_TEST(ShouldCallDtorsOfNonPodTypes) {
            UNIT_ASSERT(!TTypeTraits<TTestObject>::IsPod);
            UNIT_ASSERT_EQUAL(TTestObject::Count, 0);

            {
                TMemoryPool pool(1024);
                TSkipList<TTestObject> list(pool);

                UNIT_ASSERT(list.Insert(TTestObject(1)));
                UNIT_ASSERT(list.Insert(TTestObject(2)));

                UNIT_ASSERT_EQUAL(TTestObject::Count, 2);
            }

            UNIT_ASSERT_EQUAL(TTestObject::Count, 0);
        }
    }

}
