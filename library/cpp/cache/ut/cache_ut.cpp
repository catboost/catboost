#include <library/cpp/cache/cache.h>
#include <library/cpp/cache/thread_safe_cache.h>
#include <library/cpp/testing/unittest/registar.h>

struct TStrokaWeighter {
    static size_t Weight(const TString& s) {
        return s.size();
    }
};

Y_UNIT_TEST_SUITE(TCacheTest) {
    Y_UNIT_TEST(LRUListTest) {
        typedef TLRUList<int, TString> TListType;
        TListType list(2);

        TListType::TItem x1(1, "ttt");
        list.Insert(&x1);
        UNIT_ASSERT_EQUAL(list.GetOldest()->Key, 1);

        TListType::TItem x2(2, "yyy");
        list.Insert(&x2);
        UNIT_ASSERT_EQUAL(list.GetOldest()->Key, 1);

        list.Promote(list.GetOldest());
        UNIT_ASSERT_EQUAL(list.GetOldest()->Key, 2);

        TListType::TItem x3(3, "zzz");
        list.Insert(&x3);
        UNIT_ASSERT_EQUAL(list.GetOldest()->Key, 1);
    }

    Y_UNIT_TEST(LRUListWeightedTest) {
        typedef TLRUList<int, TString, size_t (*)(const TString&)> TListType;
        TListType list(7, [](auto& string) {
            return string.size();
        });

        TListType::TItem x1(1, "ttt");
        list.Insert(&x1);
        while (list.RemoveIfOverflown()) {
        }
        UNIT_ASSERT_EQUAL(list.GetOldest()->Key, 1);

        TListType::TItem x2(2, "yyy");
        list.Insert(&x2);
        while (list.RemoveIfOverflown()) {
        }
        UNIT_ASSERT_EQUAL(list.GetOldest()->Key, 1);

        list.Promote(list.GetOldest());
        while (list.RemoveIfOverflown()) {
        }
        UNIT_ASSERT_EQUAL(list.GetOldest()->Key, 2);

        TListType::TItem x3(3, "zzz");
        list.Insert(&x3);
        while (list.RemoveIfOverflown()) {
        }
        UNIT_ASSERT_EQUAL(list.GetOldest()->Key, 1);

        TListType::TItem x4(4, "longlong");
        list.Insert(&x4);
        while (list.RemoveIfOverflown()) {
        }
        UNIT_ASSERT_EQUAL(list.GetOldest()->Key, 4);
    }

    Y_UNIT_TEST(LFUListTest) {
        typedef TLFUList<int, TString> TListType;
        TListType list(2);

        TListType::TItem x1(1, "ttt");
        list.Insert(&x1);
        UNIT_ASSERT_EQUAL(list.GetLeastFrequentlyUsed()->Key, 1);

        TListType::TItem x2(2, "yyy");
        list.Insert(&x2);
        UNIT_ASSERT_EQUAL(list.GetLeastFrequentlyUsed()->Key, 1);

        list.Promote(list.GetLeastFrequentlyUsed());
        UNIT_ASSERT_EQUAL(list.GetLeastFrequentlyUsed()->Key, 2);

        TListType::TItem x3(3, "zzz");
        list.Insert(&x3);
        UNIT_ASSERT_EQUAL(list.GetLeastFrequentlyUsed()->Key, 1);
    }

    Y_UNIT_TEST(LWListTest) {
        typedef TLWList<int, TString, size_t, TStrokaWeighter> TListType;
        TListType list(2);

        TListType::TItem x1(1, "tt");
        list.Insert(&x1);
        UNIT_ASSERT_EQUAL(list.GetLightest()->Key, 1);
        UNIT_ASSERT_EQUAL(list.GetSize(), 1);

        TListType::TItem x2(2, "yyyy");
        list.Insert(&x2);
        UNIT_ASSERT_EQUAL(list.GetLightest()->Key, 1);
        UNIT_ASSERT_EQUAL(list.GetSize(), 2);

        TListType::TItem x3(3, "z");
        list.Insert(&x3);
        UNIT_ASSERT_EQUAL(list.GetLightest()->Key, 1);
        UNIT_ASSERT_EQUAL(list.GetSize(), 2);

        TListType::TItem x4(4, "xxxxxx");
        list.Insert(&x4);
        UNIT_ASSERT_EQUAL(list.GetLightest()->Key, 2);
        UNIT_ASSERT_EQUAL(list.GetSize(), 2);

        list.Erase(&x2);
        UNIT_ASSERT_EQUAL(list.GetLightest()->Key, 4);
        UNIT_ASSERT_EQUAL(list.GetSize(), 1);
    }

    Y_UNIT_TEST(SimpleTest) {
        typedef TLRUCache<int, TString> TCache;
        TCache s(2); // size 2
        s.Insert(1, "abcd");
        UNIT_ASSERT(s.Find(1) != s.End());
        UNIT_ASSERT_EQUAL(*s.Find(1), "abcd");
        s.Insert(2, "defg");
        UNIT_ASSERT(s.GetOldest() == "abcd");
        s.Insert(3, "hjkl");
        UNIT_ASSERT(s.GetOldest() == "defg");
        // key 1 will be deleted
        UNIT_ASSERT(s.Find(1) == s.End());
        UNIT_ASSERT(s.Find(2) != s.End());
        UNIT_ASSERT(*s.Find(2) == "defg");
        UNIT_ASSERT(s.Find(3) != s.End());
        UNIT_ASSERT(*s.Find(3) == "hjkl");

        UNIT_ASSERT(!s.Insert(3, "abcd"));
        UNIT_ASSERT(*s.Find(3) == "hjkl");
        s.Update(3, "abcd");
        UNIT_ASSERT(*s.Find(3) == "abcd");

        TCache::TIterator it = s.Find(3);
        s.Erase(it);
        UNIT_ASSERT(s.Find(3) == s.End());
    }

    Y_UNIT_TEST(LRUWithCustomSizeProviderTest) {
        typedef TLRUCache<int, TString, TNoopDelete, size_t(*)(const TString&)> TCache;
        TCache s(10, false, [](auto& string) { return string.size(); }); // size 10
        s.Insert(1, "abcd");
        UNIT_ASSERT(s.Find(1) != s.End());
        UNIT_ASSERT_EQUAL(*s.Find(1), "abcd");
        s.Insert(2, "defg");
        UNIT_ASSERT(s.GetOldest() == "abcd");
        s.Insert(3, "2c");
        UNIT_ASSERT(s.GetOldest() == "abcd");
        s.Insert(4, "hjkl");
        UNIT_ASSERT(s.GetOldest() == "defg");
        // key 1 will be deleted
        UNIT_ASSERT(s.Find(1) == s.End());
        UNIT_ASSERT(s.Find(2) != s.End());
        UNIT_ASSERT(*s.Find(2) == "defg");
        UNIT_ASSERT(s.Find(3) != s.End());
        UNIT_ASSERT(*s.Find(3) == "2c");
        UNIT_ASSERT(s.Find(4) != s.End());
        UNIT_ASSERT(*s.Find(4) == "hjkl");

        UNIT_ASSERT(!s.Insert(3, "abcd"));
        UNIT_ASSERT(*s.Find(3) == "2c");
        s.Update(3, "abcd");
        UNIT_ASSERT(*s.Find(3) == "abcd");

        TCache::TIterator it = s.Find(3);
        s.Erase(it);
        UNIT_ASSERT(s.Find(3) == s.End());
    }

    Y_UNIT_TEST(LRUSetMaxSizeTest) {
        typedef TLRUCache<int, TString> TCache;
        TCache s(2); // size 2
        s.Insert(1, "abcd");
        s.Insert(2, "efgh");
        s.Insert(3, "ijkl");
        UNIT_ASSERT(s.GetOldest() == "efgh");
        UNIT_ASSERT(s.Find(1) == s.End());
        UNIT_ASSERT(s.Find(2) != s.End());
        UNIT_ASSERT(s.Find(3) != s.End());

        // Increasing size should not change anything
        s.SetMaxSize(3);
        UNIT_ASSERT(s.GetOldest() == "efgh");
        UNIT_ASSERT(s.Find(1) == s.End());
        UNIT_ASSERT(s.Find(2) != s.End());
        UNIT_ASSERT(s.Find(3) != s.End());

        // And we should be able to add fit more entries
        s.Insert(4, "mnop");
        s.Insert(5, "qrst");
        UNIT_ASSERT(s.GetOldest() == "ijkl");
        UNIT_ASSERT(s.Find(1) == s.End());
        UNIT_ASSERT(s.Find(2) == s.End());
        UNIT_ASSERT(s.Find(3) != s.End());
        UNIT_ASSERT(s.Find(4) != s.End());
        UNIT_ASSERT(s.Find(5) != s.End());

        // Decreasing size should remove oldest entries
        s.SetMaxSize(2);
        UNIT_ASSERT(s.GetOldest() == "mnop");
        UNIT_ASSERT(s.Find(1) == s.End());
        UNIT_ASSERT(s.Find(2) == s.End());
        UNIT_ASSERT(s.Find(3) == s.End());
        UNIT_ASSERT(s.Find(4) != s.End());
        UNIT_ASSERT(s.Find(5) != s.End());

        // Ano no more entries will fit
        s.Insert(6, "uvwx");
        UNIT_ASSERT(s.GetOldest() == "qrst");
        UNIT_ASSERT(s.Find(1) == s.End());
        UNIT_ASSERT(s.Find(2) == s.End());
        UNIT_ASSERT(s.Find(3) == s.End());
        UNIT_ASSERT(s.Find(4) == s.End());
        UNIT_ASSERT(s.Find(5) != s.End());
        UNIT_ASSERT(s.Find(6) != s.End());
    }

    Y_UNIT_TEST(LWSetMaxSizeTest) {
        typedef TLWCache<int, TString, size_t, TStrokaWeighter> TCache;
        TCache s(2); // size 2
        s.Insert(1, "a");
        s.Insert(2, "aa");
        s.Insert(3, "aaa");
        UNIT_ASSERT(s.GetLightest() == "aa");
        UNIT_ASSERT(s.Find(1) == s.End());
        UNIT_ASSERT(s.Find(2) != s.End());
        UNIT_ASSERT(s.Find(3) != s.End());

        // Increasing size should not change anything
        s.SetMaxSize(3);
        UNIT_ASSERT(s.GetLightest() == "aa");
        UNIT_ASSERT(s.Find(1) == s.End());
        UNIT_ASSERT(s.Find(2) != s.End());
        UNIT_ASSERT(s.Find(3) != s.End());

        // And we should be able to add fit more entries
        s.Insert(4, "aaaa");
        s.Insert(5, "aaaaa");
        UNIT_ASSERT(s.GetLightest() == "aaa");
        UNIT_ASSERT(s.Find(1) == s.End());
        UNIT_ASSERT(s.Find(2) == s.End());
        UNIT_ASSERT(s.Find(3) != s.End());
        UNIT_ASSERT(s.Find(4) != s.End());
        UNIT_ASSERT(s.Find(5) != s.End());

        // Decreasing size should remove oldest entries
        s.SetMaxSize(2);
        UNIT_ASSERT(s.GetLightest() == "aaaa");
        UNIT_ASSERT(s.Find(1) == s.End());
        UNIT_ASSERT(s.Find(2) == s.End());
        UNIT_ASSERT(s.Find(3) == s.End());
        UNIT_ASSERT(s.Find(4) != s.End());
        UNIT_ASSERT(s.Find(5) != s.End());

        // Ano no more entries will fit
        s.Insert(6, "aaaaaa");
        UNIT_ASSERT(s.GetLightest() == "aaaaa");
        UNIT_ASSERT(s.Find(1) == s.End());
        UNIT_ASSERT(s.Find(2) == s.End());
        UNIT_ASSERT(s.Find(3) == s.End());
        UNIT_ASSERT(s.Find(4) == s.End());
        UNIT_ASSERT(s.Find(5) != s.End());
        UNIT_ASSERT(s.Find(6) != s.End());
    }

    Y_UNIT_TEST(LFUSetMaxSizeTest) {
        typedef TLFUCache<int, TString> TCache;
        TCache s(2); // size 2
        s.Insert(1, "abcd");
        s.Insert(2, "efgh");
        s.Insert(3, "ijkl");
        UNIT_ASSERT(s.Find(1) == s.End());
        UNIT_ASSERT(s.Find(2) != s.End());
        UNIT_ASSERT(s.Find(3) != s.End());

        // Increasing size should not change anything
        s.SetMaxSize(3);
        UNIT_ASSERT(s.Find(1) == s.End());
        UNIT_ASSERT(s.Find(2) != s.End());
        UNIT_ASSERT(s.Find(3) != s.End());

        // And we should be able to add fit more entries
        s.Insert(4, "mnop");
        s.Insert(5, "qrst");
        UNIT_ASSERT(s.Find(1) == s.End());
        UNIT_ASSERT(s.Find(2) == s.End());
        UNIT_ASSERT(s.Find(3) != s.End());
        UNIT_ASSERT(s.Find(4) != s.End());
        UNIT_ASSERT(s.Find(5) != s.End());

        // Decreasing size should remove oldest entries
        s.SetMaxSize(2);
        UNIT_ASSERT(s.Find(1) == s.End());
        UNIT_ASSERT(s.Find(2) == s.End());
        UNIT_ASSERT(s.Find(3) != s.End());
        UNIT_ASSERT(s.Find(4) == s.End());
        UNIT_ASSERT(s.Find(5) != s.End());

        // Ano no more entries will fit
        s.Insert(6, "uvwx");
        UNIT_ASSERT(s.Find(1) == s.End());
        UNIT_ASSERT(s.Find(2) == s.End());
        UNIT_ASSERT(s.Find(3) != s.End());
        UNIT_ASSERT(s.Find(4) == s.End());
        UNIT_ASSERT(s.Find(5) == s.End());
        UNIT_ASSERT(s.Find(6) != s.End());
    }

    Y_UNIT_TEST(MultiCacheTest) {
        typedef TLRUCache<int, TString> TCache;
        TCache s(3, true);
        UNIT_ASSERT(s.Insert(1, "abcd"));
        UNIT_ASSERT(s.Insert(1, "bcde"));
        UNIT_ASSERT(s.Insert(2, "fghi"));
        UNIT_ASSERT(s.Insert(2, "ghij"));
        // (1, "abcd") will be deleted
        UNIT_ASSERT(*s.Find(1) == "bcde");
        // (1, "bcde") will be promoted
        UNIT_ASSERT(*s.FindOldest() == "fghi");
    }

    struct TMyDelete {
        static int count;
        template <typename T>
        static void Destroy(const T&) {
            ++count;
        }
    };
    int TMyDelete::count = 0;

    Y_UNIT_TEST(DeleterTest) {
        typedef TLRUCache<int, TString, TMyDelete> TCache;
        TCache s(2);
        s.Insert(1, "123");
        s.Insert(2, "456");
        s.Insert(3, "789");
        UNIT_ASSERT(TMyDelete::count == 1);
        TCache::TIterator it = s.Find(2);
        UNIT_ASSERT(it != s.End());
        s.Erase(it);
        UNIT_ASSERT(TMyDelete::count == 2);
    }

    Y_UNIT_TEST(PromoteOnFind) {
        typedef TLRUCache<int, TString> TCache;
        TCache s(2);
        s.Insert(1, "123");
        s.Insert(2, "456");
        UNIT_ASSERT(s.Find(1) != s.End());
        s.Insert(3, "789");
        UNIT_ASSERT(s.Find(1) != s.End()); // Key 2 should have been deleted
    }
}

Y_UNIT_TEST_SUITE(TThreadSafeCacheTest) {
    typedef TThreadSafeCache<ui32, TString, ui32> TCache;

    const char* VALS[] = {"abcd", "defg", "hjkl"};

    class TCallbacks: public TCache::ICallbacks {
    public:
        TKey GetKey(ui32 i) const override {
            return i;
        }
        TValue* CreateObject(ui32 i) const override {
            Creations++;
            return new TString(VALS[i]);
        }

        mutable i32 Creations = 0;
    };

    Y_UNIT_TEST(SimpleTest) {
        for (ui32 i = 0; i < Y_ARRAY_SIZE(VALS); ++i) {
            const TString data = *TCache::Get<TCallbacks>(i);
            UNIT_ASSERT(data == VALS[i]);
        }
    }

    Y_UNIT_TEST(InsertUpdateTest) {
        TCallbacks callbacks;
        TCache cache(callbacks, 10);

        cache.Insert(2, MakeAtomicShared<TString>("hj"));
        TAtomicSharedPtr<TString> item = cache.Get(2);

        UNIT_ASSERT(callbacks.Creations == 0);
        UNIT_ASSERT(*item == "hj");

        cache.Insert(2, MakeAtomicShared<TString>("hjk"));
        item = cache.Get(2);

        UNIT_ASSERT(callbacks.Creations == 0);
        UNIT_ASSERT(*item == "hj");

        cache.Update(2, MakeAtomicShared<TString>("hjk"));
        item = cache.Get(2);

        UNIT_ASSERT(callbacks.Creations == 0);
        UNIT_ASSERT(*item == "hjk");
    }
}

Y_UNIT_TEST_SUITE(TThreadSafeCacheUnsafeTest) {
    typedef TThreadSafeCache<ui32, TString, ui32> TCache;

    const char* VALS[] = {"abcd", "defg", "hjkl"};
    const ui32 FAILED_IDX = 1;

    class TCallbacks: public TCache::ICallbacks {
    public:
        TKey GetKey(ui32 i) const override {
            return i;
        }
        TValue* CreateObject(ui32 i) const override {
            if (i == FAILED_IDX) {
                return nullptr;
            }
            return new TString(VALS[i]);
        }
    };

    Y_UNIT_TEST(SimpleTest) {
        TCallbacks callbacks;
        TCache cache(callbacks, Y_ARRAY_SIZE(VALS));
        for (ui32 i = 0; i < Y_ARRAY_SIZE(VALS); ++i) {
            const TString* data = cache.GetUnsafe(i).Get();
            if (i == FAILED_IDX) {
                UNIT_ASSERT(data == nullptr);
            } else {
                UNIT_ASSERT(*data == VALS[i]);
            }
        }
    }
}
