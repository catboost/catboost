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
        UNIT_ASSERT_EQUAL(list.GetTotalSize(), 1);
        UNIT_ASSERT_EQUAL(list.GetSize(), 1);
        UNIT_ASSERT_EQUAL(list.GetOldest()->Key, 1);

        TListType::TItem x2(2, "yyy");
        list.Insert(&x2);
        UNIT_ASSERT_EQUAL(list.GetTotalSize(), 2);
        UNIT_ASSERT_EQUAL(list.GetSize(), 2);
        UNIT_ASSERT_EQUAL(list.GetOldest()->Key, 1);

        list.Promote(list.GetOldest());
        UNIT_ASSERT_EQUAL(list.GetOldest()->Key, 2);

        TListType::TItem x3(3, "zzz");
        list.Insert(&x3);
        UNIT_ASSERT_EQUAL(list.GetTotalSize(), 2);
        UNIT_ASSERT_EQUAL(list.GetSize(), 2);
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
        UNIT_ASSERT_EQUAL(list.GetTotalSize(), 3);
        UNIT_ASSERT_EQUAL(list.GetSize(), 1);
        UNIT_ASSERT_EQUAL(list.GetOldest()->Key, 1);

        TListType::TItem x2(2, "yyy");
        list.Insert(&x2);
        while (list.RemoveIfOverflown()) {
        }
        UNIT_ASSERT_EQUAL(list.GetTotalSize(), 6);
        UNIT_ASSERT_EQUAL(list.GetSize(), 2);
        UNIT_ASSERT_EQUAL(list.GetOldest()->Key, 1);

        list.Promote(list.GetOldest());
        while (list.RemoveIfOverflown()) {
        }
        UNIT_ASSERT_EQUAL(list.GetOldest()->Key, 2);

        TListType::TItem x3(3, "zzz");
        list.Insert(&x3);
        while (list.RemoveIfOverflown()) {
        }
        UNIT_ASSERT_EQUAL(list.GetTotalSize(), 6);
        UNIT_ASSERT_EQUAL(list.GetSize(), 2);
        UNIT_ASSERT_EQUAL(list.GetOldest()->Key, 1);

        TListType::TItem x4(4, "longlong");
        list.Insert(&x4);
        while (list.RemoveIfOverflown()) {
        }
        UNIT_ASSERT_EQUAL(list.GetTotalSize(), 8);
        UNIT_ASSERT_EQUAL(list.GetSize(), 1);
        UNIT_ASSERT_EQUAL(list.GetOldest()->Key, 4);

        TListType::TItem x5(5, "xxx");
        list.Insert(&x5);
        while (list.RemoveIfOverflown()) {
        }
        UNIT_ASSERT_EQUAL(list.GetTotalSize(), 3);
        UNIT_ASSERT_EQUAL(list.GetSize(), 1);
        UNIT_ASSERT_EQUAL(list.GetOldest()->Key, 5);
    }

    Y_UNIT_TEST(LFUListTest) {
        typedef TLFUList<int, TString> TListType;
        TListType list(2);

        TListType::TItem x1(1, "ttt");
        list.Insert(&x1);
        UNIT_ASSERT_EQUAL(list.GetTotalSize(), 1);
        UNIT_ASSERT_EQUAL(list.GetSize(), 1);
        UNIT_ASSERT_EQUAL(list.GetLeastFrequentlyUsed()->Key, 1);

        TListType::TItem x2(2, "yyy");
        list.Insert(&x2);
        UNIT_ASSERT_EQUAL(list.GetTotalSize(), 2);
        UNIT_ASSERT_EQUAL(list.GetSize(), 2);
        UNIT_ASSERT_EQUAL(list.GetLeastFrequentlyUsed()->Key, 1);

        list.Promote(list.GetLeastFrequentlyUsed());
        UNIT_ASSERT_EQUAL(list.GetLeastFrequentlyUsed()->Key, 2);

        TListType::TItem x3(3, "zzz");
        list.Insert(&x3);
        UNIT_ASSERT_EQUAL(list.GetTotalSize(), 2);
        UNIT_ASSERT_EQUAL(list.GetSize(), 2);
        UNIT_ASSERT_EQUAL(list.GetLeastFrequentlyUsed()->Key, 1);
    }

    Y_UNIT_TEST(LWListTest) {
        typedef TLWList<int, TString, size_t, TStrokaWeighter> TListType;
        TListType list(2);

        TListType::TItem x1(1, "tt");
        list.Insert(&x1);
        UNIT_ASSERT_EQUAL(list.GetLightest()->Key, 1);
        UNIT_ASSERT_EQUAL(list.GetTotalSize(), 1);
        UNIT_ASSERT_EQUAL(list.GetSize(), 1);

        TListType::TItem x2(2, "yyyy");
        list.Insert(&x2);
        UNIT_ASSERT_EQUAL(list.GetLightest()->Key, 1);
        UNIT_ASSERT_EQUAL(list.GetTotalSize(), 2);
        UNIT_ASSERT_EQUAL(list.GetSize(), 2);

        TListType::TItem x3(3, "z");
        list.Insert(&x3);
        UNIT_ASSERT_EQUAL(list.GetLightest()->Key, 1);
        UNIT_ASSERT_EQUAL(list.GetTotalSize(), 2);
        UNIT_ASSERT_EQUAL(list.GetSize(), 2);

        TListType::TItem x4(4, "xxxxxx");
        list.Insert(&x4);
        UNIT_ASSERT_EQUAL(list.GetLightest()->Key, 2);
        UNIT_ASSERT_EQUAL(list.GetTotalSize(), 2);
        UNIT_ASSERT_EQUAL(list.GetSize(), 2);

        list.Erase(&x2);
        UNIT_ASSERT_EQUAL(list.GetLightest()->Key, 4);
        UNIT_ASSERT_EQUAL(list.GetTotalSize(), 1);
        UNIT_ASSERT_EQUAL(list.GetSize(), 1);
    }

    Y_UNIT_TEST(SimpleTest) {
        typedef TLRUCache<int, TString> TCache;
        TCache s(2); // size 2
        s.Insert(1, "abcd");
        UNIT_ASSERT_EQUAL(s.TotalSize(), 1);
        UNIT_ASSERT_EQUAL(s.Size(), 1);
        UNIT_ASSERT(s.Find(1) != s.End());
        UNIT_ASSERT_EQUAL(*s.Find(1), "abcd");
        s.Insert(2, "defg");
        UNIT_ASSERT_EQUAL(s.TotalSize(), 2);
        UNIT_ASSERT_EQUAL(s.Size(), 2);
        UNIT_ASSERT(s.GetOldest() == "abcd");
        s.Insert(3, "hjkl");
        UNIT_ASSERT_EQUAL(s.TotalSize(), 2);
        UNIT_ASSERT_EQUAL(s.Size(), 2);
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
        UNIT_ASSERT_EQUAL(s.TotalSize(), 2);
        UNIT_ASSERT_EQUAL(s.Size(), 2);

        TCache::TIterator it = s.Find(3);
        s.Erase(it);
        UNIT_ASSERT_EQUAL(s.TotalSize(), 1);
        UNIT_ASSERT_EQUAL(s.Size(), 1);
        UNIT_ASSERT(s.Find(3) == s.End());
    }

    Y_UNIT_TEST(LRUWithCustomSizeProviderTest) {
        typedef TLRUCache<int, TString, TNoopDelete, size_t(*)(const TString&)> TCache;
        TCache s(10, false, [](auto& string) { return string.size(); }); // size 10
        s.Insert(1, "abcd");
        UNIT_ASSERT_EQUAL(s.TotalSize(), 4);
        UNIT_ASSERT_EQUAL(s.Size(), 1);
        UNIT_ASSERT(s.Find(1) != s.End());
        UNIT_ASSERT_EQUAL(*s.Find(1), "abcd");
        s.Insert(2, "defg");
        UNIT_ASSERT_EQUAL(s.TotalSize(), 8);
        UNIT_ASSERT_EQUAL(s.Size(), 2);
        UNIT_ASSERT(s.GetOldest() == "abcd");
        s.Insert(3, "2c");
        UNIT_ASSERT_EQUAL(s.TotalSize(), 10);
        UNIT_ASSERT_EQUAL(s.Size(), 3);
        UNIT_ASSERT(s.GetOldest() == "abcd");
        s.Insert(4, "hjkl");
        UNIT_ASSERT_EQUAL(s.TotalSize(), 10);
        UNIT_ASSERT_EQUAL(s.Size(), 3);
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
        UNIT_ASSERT_EQUAL(s.TotalSize(), 8);
        UNIT_ASSERT_EQUAL(s.Size(), 2);
        UNIT_ASSERT(*s.Find(3) == "abcd");

        TCache::TIterator it = s.Find(3);
        s.Erase(it);
        UNIT_ASSERT_EQUAL(s.TotalSize(), 4);
        UNIT_ASSERT_EQUAL(s.Size(), 1);
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
        // (1, "abcd") will be deleted
        UNIT_ASSERT(s.Insert(2, "ghij"));

        UNIT_ASSERT_EQUAL(s.TotalSize(), 3);
        UNIT_ASSERT_EQUAL(s.Size(), 3);

        // (1, "bcde") will be promoted
        UNIT_ASSERT(*s.Find(1) == "bcde");
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

    class TMoveOnlyInt {
    public:
        ui32 Value = 0;

        explicit TMoveOnlyInt(ui32 value = 0) : Value(value) {}
        TMoveOnlyInt(TMoveOnlyInt&&) = default;
        TMoveOnlyInt& operator=(TMoveOnlyInt&&) = default;

        TMoveOnlyInt(const TMoveOnlyInt&) = delete;
        TMoveOnlyInt& operator=(const TMoveOnlyInt&) = delete;

        bool operator==(const TMoveOnlyInt& rhs) const {
            return Value == rhs.Value;
        }

        explicit operator size_t() const {
            return Value;
        }
    };

    Y_UNIT_TEST(MoveOnlySimpleTest) {
        typedef TLRUCache<TMoveOnlyInt, TMoveOnlyInt> TCache;
        TCache s(2); // size 2
        s.Insert(TMoveOnlyInt(1), TMoveOnlyInt(0x11111111));
        TMoveOnlyInt lookup1(1), lookup2(2), lookup3(3);
        UNIT_ASSERT(s.Find(lookup1) != s.End());
        UNIT_ASSERT_EQUAL(s.Find(lookup1)->Value, 0x11111111);
        s.Insert(TMoveOnlyInt(2), TMoveOnlyInt(0x22222222));
        UNIT_ASSERT(s.GetOldest().Value == 0x11111111);
        s.Insert(TMoveOnlyInt(3), TMoveOnlyInt(0x33333333));
        UNIT_ASSERT(s.GetOldest().Value == 0x22222222);
        // key 1 will be deleted
        UNIT_ASSERT(s.Find(lookup1) == s.End());
        UNIT_ASSERT(s.Find(lookup2) != s.End());
        UNIT_ASSERT(s.Find(lookup2)->Value == 0x22222222);
        UNIT_ASSERT(s.Find(lookup3) != s.End());
        UNIT_ASSERT(s.Find(lookup3)->Value == 0x33333333);

        UNIT_ASSERT(!s.Insert(TMoveOnlyInt(3), TMoveOnlyInt(0x11111111)));
        UNIT_ASSERT(s.Find(lookup3)->Value == 0x33333333);
        s.Update(TMoveOnlyInt(3), TMoveOnlyInt(0x11111111));
        UNIT_ASSERT(s.Find(lookup3)->Value == 0x11111111);

        TCache::TIterator it = s.Find(lookup3);
        s.Erase(it);
        UNIT_ASSERT(s.Find(lookup3) == s.End());
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

        UNIT_ASSERT_EQUAL(cache.TotalSize(), 1);
        UNIT_ASSERT_EQUAL(cache.Size(), 1);
        UNIT_ASSERT(callbacks.Creations == 0);
        UNIT_ASSERT(*item == "hjk");
    }

    Y_UNIT_TEST(GetOrNullTest) {
        TCallbacks callbacks;
        TCache cache(callbacks, 10);
        i32 expectedCreations = 0;

        auto item = cache.GetOrNull(0);
        UNIT_ASSERT(item == nullptr);
        UNIT_ASSERT(callbacks.Creations == expectedCreations);
        UNIT_ASSERT(cache.TotalSize() == 0);

        item = cache.Get(0);
        UNIT_ASSERT(*item == "abcd");
        UNIT_ASSERT(callbacks.Creations == ++expectedCreations);
        UNIT_ASSERT(cache.TotalSize() == 1);

        item = cache.GetOrNull(0);
        UNIT_ASSERT(*item == "abcd");
        UNIT_ASSERT(callbacks.Creations == expectedCreations);
        UNIT_ASSERT(cache.TotalSize() == 1);
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
        UNIT_ASSERT_EQUAL(cache.TotalSize(), Y_ARRAY_SIZE(VALS) - 1);
        UNIT_ASSERT_EQUAL(cache.Size(), Y_ARRAY_SIZE(VALS) - 1);
    }
}

Y_UNIT_TEST_SUITE(TThreadSafeLRUCacheTest) {
    typedef TThreadSafeLRUCache<size_t, TString, size_t> TCache;

    TVector<TString> Values = {"zero", "one", "two", "three", "four"};

    class TCallbacks: public TCache::ICallbacks {
    public:
        TKey GetKey(size_t i) const override {
            return i;
        }
        TValue* CreateObject(size_t i) const override {
            UNIT_ASSERT(i < Values.size());
            Creations++;
            return new TString(Values[i]);
        }

        mutable size_t Creations = 0;
    };

    Y_UNIT_TEST(SimpleTest) {
        for (size_t i = 0; i < Values.size(); ++i) {
            const TString data = *TCache::Get<TCallbacks>(i);
            UNIT_ASSERT(data == Values[i]);
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

        UNIT_ASSERT_EQUAL(cache.TotalSize(), 1);
        UNIT_ASSERT_EQUAL(cache.Size(), 1);
        UNIT_ASSERT(callbacks.Creations == 0);
        UNIT_ASSERT(*item == "hjk");
    }

    Y_UNIT_TEST(LRUTest) {
        TCallbacks callbacks;
        TCache cache(callbacks, 3);

        UNIT_ASSERT_EQUAL(cache.GetMaxSize(), 3);

        for (size_t i = 0; i < Values.size(); ++i) {
            TAtomicSharedPtr<TString> item = cache.Get(i);
            UNIT_ASSERT(*item == Values[i]);
        }
        UNIT_ASSERT(callbacks.Creations == Values.size());

        size_t expectedCreations = Values.size();
        TAtomicSharedPtr<TString> item;


        item = cache.Get(4);
        UNIT_ASSERT_EQUAL(callbacks.Creations, expectedCreations);
        UNIT_ASSERT(*item == "four");

        item = cache.Get(2);
        UNIT_ASSERT_EQUAL(callbacks.Creations, expectedCreations);
        UNIT_ASSERT(*item == "two");

        item = cache.Get(0);
        expectedCreations++;
        UNIT_ASSERT_EQUAL(callbacks.Creations, expectedCreations);
        UNIT_ASSERT(*item == "zero");

        UNIT_ASSERT(cache.Contains(1) == false);
        UNIT_ASSERT(cache.Contains(3) == false);
        UNIT_ASSERT(cache.Contains(4));
        UNIT_ASSERT(cache.Contains(2));
        UNIT_ASSERT(cache.Contains(0));

        item = cache.Get(3);
        expectedCreations++;
        UNIT_ASSERT_EQUAL(callbacks.Creations, expectedCreations);
        UNIT_ASSERT(*item == "three");

        item = cache.Get(2);
        UNIT_ASSERT_EQUAL(callbacks.Creations, expectedCreations);
        UNIT_ASSERT(*item == "two");

        item = cache.Get(0);
        UNIT_ASSERT_EQUAL(callbacks.Creations, expectedCreations);
        UNIT_ASSERT(*item == "zero");

        item = cache.Get(1);
        expectedCreations++;
        UNIT_ASSERT_EQUAL(callbacks.Creations, expectedCreations);
        UNIT_ASSERT(*item == "one");

        item = cache.Get(2);
        UNIT_ASSERT_EQUAL(callbacks.Creations, expectedCreations);
        UNIT_ASSERT(*item == "two");

        item = cache.Get(4);
        expectedCreations++;
        UNIT_ASSERT_EQUAL(callbacks.Creations, expectedCreations);
        UNIT_ASSERT(*item == "four");
    }

    Y_UNIT_TEST(ChangeMaxSizeTest) {
        TCallbacks callbacks;
        TCache cache(callbacks, 3);

        UNIT_ASSERT_EQUAL(cache.GetMaxSize(), 3);

        for (size_t i = 0; i < Values.size(); ++i) {
            TAtomicSharedPtr<TString> item = cache.Get(i);
            UNIT_ASSERT(*item == Values[i]);
        }

        size_t expectedCreations = Values.size();
        TAtomicSharedPtr<TString> item;

        item = cache.Get(4);
        UNIT_ASSERT_EQUAL(callbacks.Creations, expectedCreations);
        UNIT_ASSERT(*item == "four");

        item = cache.Get(3);
        UNIT_ASSERT_EQUAL(callbacks.Creations, expectedCreations);
        UNIT_ASSERT(*item == "three");

        item = cache.Get(2);
        UNIT_ASSERT_EQUAL(callbacks.Creations, expectedCreations);
        UNIT_ASSERT(*item == "two");

        item = cache.Get(1);
        expectedCreations++;
        UNIT_ASSERT_EQUAL(callbacks.Creations, expectedCreations);
        UNIT_ASSERT(*item == "one");

        UNIT_ASSERT_EQUAL(cache.TotalSize(), 3);
        UNIT_ASSERT_EQUAL(cache.Size(), 3);
        cache.SetMaxSize(4);

        item = cache.Get(0);
        expectedCreations++;
        UNIT_ASSERT_EQUAL(callbacks.Creations, expectedCreations);
        UNIT_ASSERT(*item == "zero");
        UNIT_ASSERT_EQUAL(cache.TotalSize(), 4);
        UNIT_ASSERT_EQUAL(cache.Size(), 4);

        item = cache.Get(4);
        expectedCreations++;
        UNIT_ASSERT_EQUAL(callbacks.Creations, expectedCreations);
        UNIT_ASSERT(*item == "four");
        UNIT_ASSERT_EQUAL(cache.TotalSize(), 4);
        UNIT_ASSERT_EQUAL(cache.Size(), 4);

        item = cache.Get(3);
        expectedCreations++;
        UNIT_ASSERT_EQUAL(callbacks.Creations, expectedCreations);
        UNIT_ASSERT(*item == "three");
        UNIT_ASSERT(cache.Contains(2) == false);

        cache.SetMaxSize(2);
        UNIT_ASSERT(cache.Contains(3));
        UNIT_ASSERT(cache.Contains(4));
        UNIT_ASSERT(cache.Contains(2) == false);
        UNIT_ASSERT(cache.Contains(1) == false);
        UNIT_ASSERT(cache.Contains(0) == false);

        item = cache.Get(0);
        expectedCreations++;
        UNIT_ASSERT_EQUAL(callbacks.Creations, expectedCreations);
        UNIT_ASSERT(*item == "zero");
        UNIT_ASSERT(cache.Contains(4) == false);
        UNIT_ASSERT(cache.Contains(3));
        UNIT_ASSERT(cache.Contains(0));
    }
}

Y_UNIT_TEST_SUITE(TThreadSafeLFUCacheTest) {
    using TCache = TThreadSafeLFUCache<size_t, TString, size_t>;

    TVector<TString> Values = {"a", "bb", "ccc", "dddd", "eeeee"};

    class TCallbacks: public TCache::ICallbacks {
    public:
        TKey GetKey(size_t i) const override {
            return i;
        }

        TValue* CreateObject(size_t i) const override {
            UNIT_ASSERT(i < Values.size());
            ++Creations;
            return new TString(Values[i]);
        }

        mutable size_t Creations = 0;
    };

    Y_UNIT_TEST(SimpleTest) {
        for (size_t i = 0; i < Values.size(); ++i) {
            const TString data = *TCache::Get<TCallbacks>(i);
            UNIT_ASSERT_EQUAL(data, Values[i]);
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

        UNIT_ASSERT_EQUAL(cache.TotalSize(), 1);
        UNIT_ASSERT_EQUAL(cache.Size(), 1);
        UNIT_ASSERT(callbacks.Creations == 0);
        UNIT_ASSERT(*item == "hjk");
    }

    Y_UNIT_TEST(LFUTest) {
        TCallbacks callbacks;
        TCache cache(callbacks, 3);
        size_t expectedCreations = 0;

        UNIT_ASSERT_EQUAL(cache.GetMaxSize(), 3);
        auto item = cache.Get(0);
        UNIT_ASSERT(*item == "a");
        UNIT_ASSERT(cache.TotalSize() == 1);
        UNIT_ASSERT(callbacks.Creations == ++expectedCreations);

        item = cache.Get(1);
        UNIT_ASSERT(*item == "bb");
        UNIT_ASSERT(cache.TotalSize() == 2);
        UNIT_ASSERT(callbacks.Creations == ++expectedCreations);

        item = cache.Get(2);
        UNIT_ASSERT(*item == "ccc");
        UNIT_ASSERT(cache.TotalSize() == 3);
        UNIT_ASSERT(callbacks.Creations == ++expectedCreations);

        cache.Get(0);
        cache.Get(0);
        cache.Get(0);

        cache.Get(1);

        cache.Get(2);
        cache.Get(2);

        // evict 1
        item = cache.Get(3);
        UNIT_ASSERT(*item == "dddd");
        UNIT_ASSERT(cache.TotalSize() == 3);
        UNIT_ASSERT(callbacks.Creations == ++expectedCreations);

        // check that 0 was evicted and left only 1 2 3
        item = cache.Get(0);
        UNIT_ASSERT(*item == "a");

        item = cache.Get(2);
        UNIT_ASSERT(*item == "ccc");

        item = cache.Get(3);
        UNIT_ASSERT(*item == "dddd");
        UNIT_ASSERT(callbacks.Creations == expectedCreations);

        cache.Get(0);
        cache.Get(2);
        cache.Get(3);

        // evict 3
        item = cache.Get(4);
        UNIT_ASSERT(*item == "eeeee");
        UNIT_ASSERT(cache.TotalSize() == 3);
        UNIT_ASSERT(callbacks.Creations == ++expectedCreations);

        // check that 1 was evicted and left only 2 3 4
        item = cache.Get(0);
        UNIT_ASSERT(*item == "a");

        item = cache.Get(2);
        UNIT_ASSERT(*item == "ccc");

        item = cache.Get(4);
        UNIT_ASSERT(*item == "eeeee");
        UNIT_ASSERT(callbacks.Creations == expectedCreations);
    }
}

Y_UNIT_TEST_SUITE(TThreadSafeLRUCacheWithSizeProviderTest) {
    struct TStringLengthSizeProvider {
        size_t operator()(const TString& s) const {
            return s.size();
        }
    };
    using TCache = TThreadSafeLRUCacheWithSizeProvider<size_t, TString, TStringLengthSizeProvider, size_t>;

    TVector<TString> Values = {"a", "bb", "ccc", "dddd", "eeeee"};

    class TCallbacks: public TCache::ICallbacks {
    public:
        TKey GetKey(size_t i) const override {
            return i;
        }
        TValue* CreateObject(size_t i) const override {
            UNIT_ASSERT(i < Values.size());
            return new TString(Values[i]);
        }
    };

    Y_UNIT_TEST(Test) {
        TCallbacks callbacks;
        TCache cache(callbacks, 6);

        auto item = cache.Get(0);
        UNIT_ASSERT(*item == "a");
        UNIT_ASSERT(cache.TotalSize() == 1);
        UNIT_ASSERT(cache.Size() == 1);

        item = cache.Get(1);
        UNIT_ASSERT(*item == "bb");
        UNIT_ASSERT(cache.TotalSize() == 3);
        UNIT_ASSERT(cache.Size() == 2);

        item = cache.Get(2);
        UNIT_ASSERT(*item == "ccc");
        UNIT_ASSERT(cache.TotalSize() == 6);
        UNIT_ASSERT(cache.Size() == 3);

        item = cache.Get(3);
        UNIT_ASSERT(*item == "dddd");
        UNIT_ASSERT(cache.TotalSize() == 4);
        UNIT_ASSERT(cache.Size() == 1);

        item = cache.Get(0);
        UNIT_ASSERT(*item == "a");
        UNIT_ASSERT(cache.TotalSize() == 5);
        UNIT_ASSERT(cache.Size() == 2);

        item = cache.Get(4);
        UNIT_ASSERT(*item == "eeeee");
        UNIT_ASSERT(cache.TotalSize() == 6);
        UNIT_ASSERT(cache.Size() == 2);

        cache.Update(0, MakeAtomicShared<TString>("aaa"));
        item = cache.Get(0);
        UNIT_ASSERT(*item == "aaa");
        UNIT_ASSERT(cache.TotalSize() == 3);
        UNIT_ASSERT(cache.Size() == 1);
    }
}

Y_UNIT_TEST_SUITE(TThreadSafeLFUCacheWithSizeProviderTest) {
    struct TStringLengthSizeProvider {
        size_t operator()(const TString& s) const {
            return s.size();
        }
    };
    using TCache = TThreadSafeLFUCacheWithSizeProvider<size_t, TString, TStringLengthSizeProvider, size_t>;

    TVector<TString> Values = {"a", "bb", "ccc", "dddd", "eeeee"};

    class TCallbacks: public TCache::ICallbacks {
    public:
        TKey GetKey(size_t i) const override {
            return i;
        }
        TValue* CreateObject(size_t i) const override {
            UNIT_ASSERT(i < Values.size());
            return new TString(Values[i]);
        }
    };

    Y_UNIT_TEST(Test) {
        TCallbacks callbacks;
        TCache cache(callbacks, 6);

        auto item = cache.Get(0);
        UNIT_ASSERT(*item == "a");
        UNIT_ASSERT(cache.TotalSize() == 1);
        UNIT_ASSERT(cache.Size() == 1);

        item = cache.Get(1);
        UNIT_ASSERT(*item == "bb");
        UNIT_ASSERT(cache.TotalSize() == 3);
        UNIT_ASSERT(cache.Size() == 2);

        item = cache.Get(2);
        UNIT_ASSERT(*item == "ccc");
        UNIT_ASSERT(cache.TotalSize() == 6);
        UNIT_ASSERT(cache.Size() == 3);

        cache.Get(0);
        cache.Get(0);
        cache.Get(0);

        cache.Get(1);

        cache.Get(2);
        cache.Get(2);

        // evict 1. 0 and 3 left
        item = cache.Get(3);
        UNIT_ASSERT(*item == "dddd");
        UNIT_ASSERT(cache.TotalSize() == 5);
        UNIT_ASSERT(cache.Size() == 2);

        cache.Get(0);
        UNIT_ASSERT(cache.TotalSize() == 5);
        UNIT_ASSERT(cache.Size() == 2);

        // evict 3. 0 and 4 left
        cache.Get(4);
        UNIT_ASSERT(cache.TotalSize() == 6);
        UNIT_ASSERT(cache.Size() == 2);

        cache.Get(4);
        cache.Get(4);
        cache.Get(4);
        cache.Get(4);
        cache.Get(4);
        // evict both 0 and 4, even if evict only 4 was ok to fit size
        // thats because 4 used more times, so it is deleted only after 0
        cache.Get(2);
        UNIT_ASSERT(cache.TotalSize() == 3);
        UNIT_ASSERT(cache.Size() == 1);
    }
}
