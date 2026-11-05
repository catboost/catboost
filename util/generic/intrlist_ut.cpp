#include "intrlist.h"

#include <library/cpp/testing/unittest/registar.h>

#include <util/generic/algorithm.h>
#include <util/stream/output.h>

class TListTest: public TTestBase {
    UNIT_TEST_SUITE(TListTest);
    UNIT_TEST(TestIterate);
    UNIT_TEST(TestRIterate);
    UNIT_TEST(TestForEach);
    UNIT_TEST(TestForEachWithDelete);
    UNIT_TEST(TestSize);
    UNIT_TEST(TestQuickSort);
    UNIT_TEST(TestCut);
    UNIT_TEST(TestAppend);
    UNIT_TEST(TestMoveCtor);
    UNIT_TEST(TestMoveOpEq);
    UNIT_TEST(TestListWithAutoDelete);
    UNIT_TEST(TestListWithAutoDeleteMoveCtor);
    UNIT_TEST(TestListWithAutoDeleteMoveOpEq);
    UNIT_TEST(TestListWithAutoDeleteClear);
    UNIT_TEST(TestSecondTag);
    UNIT_TEST(TestItemMoveCtor);
    UNIT_TEST(TestItemMoveAssign);
    UNIT_TEST_SUITE_END();

private:
    void TestSize();
    void TestIterate();
    void TestRIterate();
    void TestForEach();
    void TestForEachWithDelete();
    void TestQuickSort();
    void TestCut();
    void TestAppend();
    void TestMoveCtor();
    void TestMoveOpEq();
    void TestListWithAutoDelete();
    void TestListWithAutoDeleteMoveCtor();
    void TestListWithAutoDeleteMoveOpEq();
    void TestListWithAutoDeleteClear();
    void TestSecondTag();
    void TestItemMoveCtor();
    void TestItemMoveAssign();
};

UNIT_TEST_SUITE_REGISTRATION(TListTest);

class TInt: public TIntrusiveListItem<TInt> {
public:
    inline TInt(int value) noexcept
        : Value_(value)
    {
    }

    TInt(TInt&& rhs) noexcept
        : Value_(rhs.Value_)
    {
        rhs.Value_ = 0xDEAD;
    }

    TInt& operator=(TInt&& rhs) noexcept {
        Value_ = rhs.Value_;
        rhs.Value_ = 0xBEEF;
        return *this;
    }

    inline operator int&() noexcept {
        return Value_;
    }

    inline operator const int&() const noexcept {
        return Value_;
    }

private:
    int Value_;
};

class TMyList: public TIntrusiveList<TInt> {
public:
    inline TMyList(int count) {
        while (count > 0) {
            PushFront(new TInt(count--));
        }
    }

    // TMyList(const TMyList& rhs) = default;
    TMyList(TMyList&& rhs) noexcept = default;

    // operator=(const TMyList& rhs) = default;
    TMyList& operator=(TMyList&& rhs) noexcept = default;

    inline ~TMyList() {
        while (!Empty()) {
            delete PopBack();
        }
    }
};

struct TIntGreater: private TGreater<int> {
    inline bool operator()(const TInt& l, const TInt& r) const noexcept {
        return TGreater<int>::operator()(l, r);
    }
};

void TListTest::TestQuickSort() {
    TMyList l(1000);
    size_t c = 0;

    l.QuickSort(TIntGreater());

    UNIT_ASSERT_EQUAL(l.Size(), 1000);

    for (TMyList::TIterator it = l.Begin(); it != l.End(); ++it) {
        UNIT_ASSERT_EQUAL(*it, int(1000 - c++));
    }
}

void TListTest::TestSize() {
    TMyList l(1024);

    UNIT_ASSERT_EQUAL(l.Size(), 1024);
}

void TListTest::TestIterate() {
    TMyList l(1000);
    size_t c = 0;

    for (TMyList::TIterator it = l.Begin(); it != l.End(); ++it) {
        ++c;

        UNIT_ASSERT_EQUAL(*it, (int)c);
    }

    UNIT_ASSERT_EQUAL(c, 1000);
}

void TListTest::TestRIterate() {
    TMyList l(1000);
    size_t c = 1000;

    UNIT_ASSERT_EQUAL(l.RBegin(), TMyList::TReverseIterator(l.End()));
    UNIT_ASSERT_EQUAL(l.REnd(), TMyList::TReverseIterator(l.Begin()));

    for (TMyList::TReverseIterator it = l.RBegin(); it != l.REnd(); ++it) {
        UNIT_ASSERT_EQUAL(*it, (int)c--);
    }

    UNIT_ASSERT_EQUAL(c, 0);
}

class TSum {
public:
    inline TSum(size_t& sum)
        : Sum_(sum)
    {
    }

    inline void operator()(const TInt* v) noexcept {
        Sum_ += *v;
    }

private:
    size_t& Sum_;
};

class TSumWithDelete {
public:
    inline TSumWithDelete(size_t& sum)
        : Sum_(sum)
    {
    }

    inline void operator()(TInt* v) noexcept {
        if (*v % 2) {
            Sum_ += *v;
        } else {
            delete v;
        }
    }

private:
    size_t& Sum_;
};

void TListTest::TestForEach() {
    TMyList l(1000);
    size_t sum = 0;
    TSum functor(sum);

    l.ForEach(functor);

    UNIT_ASSERT_EQUAL(sum, 1000 * 1001 / 2);
}

void TListTest::TestForEachWithDelete() {
    TMyList l(1000);
    size_t sum = 0;
    TSumWithDelete functor(sum);

    l.ForEach(functor);

    UNIT_ASSERT_EQUAL(sum, 500 * 500 /*== n * (x + y * (n - 1) / 2), x == 1, y == 2*/);
}

static void CheckIterationAfterCut(const TMyList& l, const TMyList& l2, size_t N, size_t M) {
    size_t c = 0;
    for (TMyList::TConstIterator it = l.Begin(); it != l.End(); ++it) {
        ++c;

        UNIT_ASSERT_EQUAL(*it, (int)c);
    }

    UNIT_ASSERT_EQUAL(c, M);

    for (TMyList::TConstIterator it = l2.Begin(); it != l2.End(); ++it) {
        ++c;

        UNIT_ASSERT_EQUAL(*it, (int)c);
    }

    UNIT_ASSERT_EQUAL(c, N);

    for (TMyList::TConstIterator it = l2.End(); it != l2.Begin(); --c) {
        --it;

        UNIT_ASSERT_EQUAL(*it, (int)c);
    }

    UNIT_ASSERT_EQUAL(c, M);

    for (TMyList::TConstIterator it = l.End(); it != l.Begin(); --c) {
        --it;

        UNIT_ASSERT_EQUAL(*it, (int)c);
    }

    UNIT_ASSERT_EQUAL(c, 0);
}

static void TestCutFront(int N, int M) {
    TMyList l(N);
    TMyList l2(0);

    TMyList::TIterator it = l.Begin();
    for (int i = 0; i < M; ++i) {
        ++it;
    }

    TMyList::Cut(l.Begin(), it, l2.End());
    CheckIterationAfterCut(l2, l, N, M);
}

static void TestCutBack(int N, int M) {
    TMyList l(N);
    TMyList l2(0);

    TMyList::TIterator it = l.Begin();
    for (int i = 0; i < M; ++i) {
        ++it;
    }

    TMyList::Cut(it, l.End(), l2.End());
    CheckIterationAfterCut(l, l2, N, M);
}

void TListTest::TestCut() {
    TestCutFront(1000, 500);
    TestCutBack(1000, 500);
    TestCutFront(1, 0);
    TestCutBack(1, 0);
    TestCutFront(1, 1);
    TestCutBack(1, 1);
    TestCutFront(2, 0);
    TestCutBack(2, 0);
    TestCutFront(2, 1);
    TestCutBack(2, 1);
    TestCutFront(2, 2);
    TestCutBack(2, 2);
}

static void CheckIterationAfterAppend(const TMyList& l, size_t N, size_t M) {
    TMyList::TConstIterator it = l.Begin();

    for (size_t i = 1; i <= N; ++i, ++it) {
        UNIT_ASSERT_EQUAL((int)i, *it);
    }

    for (size_t i = 1; i <= M; ++i, ++it) {
        UNIT_ASSERT_EQUAL((int)i, *it);
    }

    UNIT_ASSERT_EQUAL(it, l.End());
}

static void TestAppend(int N, int M) {
    TMyList l(N);
    TMyList l2(M);
    l.Append(l2);

    UNIT_ASSERT(l2.Empty());
    CheckIterationAfterAppend(l, N, M);
}

void TListTest::TestAppend() {
    ::TestAppend(500, 500);
    ::TestAppend(0, 0);
    ::TestAppend(1, 0);
    ::TestAppend(0, 1);
    ::TestAppend(1, 1);
}

template <typename TListType>
static void CheckList(const TListType& lst) {
    int i = 1;
    for (typename TListType::TConstIterator it = lst.Begin(); it != lst.End(); ++it, ++i) {
        UNIT_ASSERT_EQUAL(*it, i);
    }
}

void TListTest::TestMoveCtor() {
    const int N{42};
    TMyList lst{N};
    UNIT_ASSERT(!lst.Empty());
    UNIT_ASSERT_EQUAL(lst.Size(), N);

    CheckList(lst);
    TMyList nextLst{std::move(lst)};
    UNIT_ASSERT(lst.Empty());
    CheckList(nextLst);
}

void TListTest::TestMoveOpEq() {
    const int N{42};
    TMyList lst{N};
    UNIT_ASSERT(!lst.Empty());
    UNIT_ASSERT_EQUAL(lst.Size(), N);
    CheckList(lst);

    const int M{2};
    TMyList nextLst(M);
    UNIT_ASSERT(!nextLst.Empty());
    UNIT_ASSERT_EQUAL(nextLst.Size(), M);
    CheckList(nextLst);

    nextLst = std::move(lst);
    UNIT_ASSERT(!nextLst.Empty());
    UNIT_ASSERT_EQUAL(nextLst.Size(), N);
    CheckList(nextLst);
}

class TSelfCountingInt: public TIntrusiveListItem<TSelfCountingInt> {
public:
    TSelfCountingInt(int& counter, int value) noexcept
        : Counter_(counter)
        , Value_(value)
    {
        ++Counter_;
    }

    TSelfCountingInt(TSelfCountingInt&& rhs) noexcept
        : Counter_(rhs.Counter_)
        , Value_(rhs.Value_)
    {
        rhs.Value_ = 0xDEAD;
    }

    TSelfCountingInt& operator=(TSelfCountingInt&& rhs) noexcept {
        Value_ = rhs.Value_;
        rhs.Value_ = 0xBEEF;
        return *this;
    }

    ~TSelfCountingInt() noexcept {
        --Counter_;
    }

    inline operator int&() noexcept {
        return Value_;
    }

    inline operator const int&() const noexcept {
        return Value_;
    }

private:
    int& Counter_;
    int Value_;
};

struct TSelfCountingIntDelete {
    static void Destroy(TSelfCountingInt* i) noexcept {
        delete i;
    }
};

void TListTest::TestListWithAutoDelete() {
    using TListType = TIntrusiveListWithAutoDelete<TSelfCountingInt, TSelfCountingIntDelete>;
    int counter{0};
    {
        TListType lst;
        UNIT_ASSERT(lst.Empty());
        lst.PushFront(new TSelfCountingInt(counter, 2));
        UNIT_ASSERT_EQUAL(lst.Size(), 1);
        UNIT_ASSERT_EQUAL(counter, 1);
        lst.PushFront(new TSelfCountingInt(counter, 1));
        UNIT_ASSERT_EQUAL(lst.Size(), 2);
        UNIT_ASSERT_EQUAL(counter, 2);
        CheckList(lst);
    }

    UNIT_ASSERT_EQUAL(counter, 0);
}

void TListTest::TestListWithAutoDeleteMoveCtor() {
    using TListType = TIntrusiveListWithAutoDelete<TSelfCountingInt, TSelfCountingIntDelete>;
    int counter{0};
    {
        TListType lst;
        lst.PushFront(new TSelfCountingInt(counter, 2));
        lst.PushFront(new TSelfCountingInt(counter, 1));
        UNIT_ASSERT_EQUAL(lst.Size(), 2);
        UNIT_ASSERT_EQUAL(counter, 2);
        CheckList(lst);

        TListType nextList(std::move(lst));
        UNIT_ASSERT_EQUAL(nextList.Size(), 2);
        CheckList(nextList);
        UNIT_ASSERT_EQUAL(counter, 2);
    }

    UNIT_ASSERT_EQUAL(counter, 0);
}

void TListTest::TestListWithAutoDeleteMoveOpEq() {
    using TListType = TIntrusiveListWithAutoDelete<TSelfCountingInt, TSelfCountingIntDelete>;
    int counter{0};
    {
        TListType lst;
        lst.PushFront(new TSelfCountingInt(counter, 2));
        lst.PushFront(new TSelfCountingInt(counter, 1));
        UNIT_ASSERT_EQUAL(lst.Size(), 2);
        UNIT_ASSERT_EQUAL(counter, 2);
        CheckList(lst);

        TListType nextList;
        UNIT_ASSERT(nextList.Empty());
        nextList = std::move(lst);
        UNIT_ASSERT_EQUAL(nextList.Size(), 2);
        CheckList(nextList);
        UNIT_ASSERT_EQUAL(counter, 2);
    }

    UNIT_ASSERT_EQUAL(counter, 0);
}

void TListTest::TestListWithAutoDeleteClear() {
    using TListType = TIntrusiveListWithAutoDelete<TSelfCountingInt, TSelfCountingIntDelete>;
    int counter{0};
    {
        TListType lst;
        UNIT_ASSERT(lst.Empty());
        lst.PushFront(new TSelfCountingInt(counter, 2));
        UNIT_ASSERT_EQUAL(lst.Size(), 1);
        UNIT_ASSERT_EQUAL(counter, 1);
        lst.PushFront(new TSelfCountingInt(counter, 1));
        UNIT_ASSERT_EQUAL(lst.Size(), 2);
        UNIT_ASSERT_EQUAL(counter, 2);
        CheckList(lst);

        lst.Clear();
        UNIT_ASSERT(lst.Empty());
        UNIT_ASSERT_EQUAL(counter, 0);
        lst.PushFront(new TSelfCountingInt(counter, 1));
        UNIT_ASSERT_EQUAL(lst.Size(), 1);
    }

    UNIT_ASSERT_EQUAL(counter, 0);
}

struct TSecondTag {};

class TDoubleNode
    : public TInt,
      public TIntrusiveListItem<TDoubleNode, TSecondTag> {
public:
    TDoubleNode(int value) noexcept
        : TInt(value)
    {
    }
};

void TListTest::TestSecondTag() {
    TDoubleNode zero(0), one(1);
    TIntrusiveList<TInt> first;
    TIntrusiveList<TDoubleNode, TSecondTag> second;

    first.PushFront(&zero);
    first.PushFront(&one);
    second.PushBack(&zero);
    second.PushBack(&one);

    UNIT_ASSERT_EQUAL(*first.Front(), 1);
    UNIT_ASSERT_EQUAL(*++first.Begin(), 0);
    UNIT_ASSERT_EQUAL(*first.Back(), 0);

    UNIT_ASSERT_EQUAL(*second.Front(), 0);
    UNIT_ASSERT_EQUAL(*++second.Begin(), 1);
    UNIT_ASSERT_EQUAL(*second.Back(), 1);

    second.Remove(&zero);
    UNIT_ASSERT_EQUAL(*second.Front(), 1);
    UNIT_ASSERT_EQUAL(*first.Back(), 0);
}

namespace {
    struct TStringItem: public TIntrusiveListItem<TStringItem> {
        explicit TStringItem(TString s)
            : Value(std::move(s))
        {
        }

        TString Value;

        friend bool operator<(const TStringItem& lhs, const TStringItem& rhs) noexcept {
            return lhs.Value < rhs.Value;
        }

        friend bool operator==(const TStringItem& lhs, const TStringItem& rhs) noexcept {
            return lhs.Value == rhs.Value;
        }

        friend bool operator==(const TStringItem& lhs, const TString& rhs) noexcept {
            return lhs.Value == rhs;
        }
    };
} // namespace

template <>
void Out<TStringItem>(IOutputStream& out, const TStringItem& item) {
    out << item.Value;
}

void TListTest::TestItemMoveCtor() {
    {
        TStringItem a("foobar");
        TIntrusiveList<TStringItem> list;
        list.PushBack(&a);
        for (const TStringItem& i : list) {
            UNIT_ASSERT_VALUES_EQUAL("foobar", i);
        }
        TStringItem b(std::move(a));
        UNIT_ASSERT_VALUES_EQUAL("foobar", b);
        for (const TStringItem& i : list) {
            UNIT_ASSERT_VALUES_EQUAL("foobar", i);
        }
    }

    {
        TVector<TStringItem> array;
        TIntrusiveList<TStringItem> list;
        for (int i = 0; i < 1000; ++i) {
            array.emplace_back(TString(i, '-'));
            list.PushBack(&array.back());
        }

        int expected = 0;
        for (const TStringItem& i : list) {
            UNIT_ASSERT_EQUAL(TString(expected, '-'), i);
            ++expected;
        }
    }
}

void TListTest::TestItemMoveAssign() {
    TVector<TStringItem> array;
    TVector<TString> ref;
    TIntrusiveList<TStringItem> list;

    for (int i = 0; i < 5000; ++i) {
        array.emplace_back(ToString(i));
        ref.emplace_back(ToString(i));
        list.PushBack(&array.back());
    }

    UNIT_ASSERT(Equal(array.begin(), array.end(), ref.begin(), ref.end()));
    UNIT_ASSERT(Equal(array.begin(), array.end(), list.begin(), list.end()));
    Sort(array);
    UNIT_ASSERT(!Equal(array.begin(), array.end(), ref.begin(), ref.end())); // array is shuffled
    UNIT_ASSERT(Equal(ref.begin(), ref.end(), list.begin(), list.end()));    // but list keeps the original order
    UNIT_ASSERT(!Equal(array.begin(), array.end(), list.begin(), list.end()));
    SortBy(array, [](const TStringItem& item) { return FromString<int>(item.Value); });
    UNIT_ASSERT(Equal(array.begin(), array.end(), ref.begin(), ref.end()));
    UNIT_ASSERT(Equal(ref.begin(), ref.end(), list.begin(), list.end())); // list still keeps the original order
    UNIT_ASSERT(Equal(array.begin(), array.end(), list.begin(), list.end()));

    {
        TStringItem a("foo");
        TStringItem b("bar");
        TIntrusiveList<TStringItem> list;
        list.PushBack(&b);
        a = std::move(b);
        UNIT_ASSERT_VALUES_EQUAL(a, "bar");
        for (const TStringItem& i : list) {
            UNIT_ASSERT_VALUES_EQUAL(i, "bar");
        }
        UNIT_ASSERT_VALUES_EQUAL(list.Size(), 1);

        TStringItem& aa = a;
        aa = std::move(a);
        UNIT_ASSERT_LE(list.Size(), 1); // self-move assignment results in a valid but unspecified state
    }

    {
        TStringItem a("adjacent node 1");
        TStringItem b("adjacent node 2");
        TIntrusiveList<TStringItem> list;
        list.PushBack(&a);
        list.PushBack(&b);
        a = std::move(b);
        UNIT_ASSERT_VALUES_EQUAL(a, "adjacent node 2");
        for (const TStringItem& i : list) {
            UNIT_ASSERT_VALUES_EQUAL(i, "adjacent node 2");
        }
        UNIT_ASSERT_VALUES_EQUAL(list.Size(), 1);
        UNIT_ASSERT(b.Empty());
    }
}
