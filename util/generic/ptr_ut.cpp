#include "ptr.h"
#include "vector.h"
#include "noncopyable.h"

#include <library/cpp/testing/common/probe.h>
#include <library/cpp/testing/unittest/registar.h>

#include <util/generic/hash_set.h>
#include <util/generic/is_in.h>
#include <util/system/thread.h>

class TPointerTest: public TTestBase {
    UNIT_TEST_SUITE(TPointerTest);
    UNIT_TEST(TestTypedefs);
    UNIT_TEST(TestSimpleIntrPtr);
    UNIT_TEST(TestHolderPtr);
    UNIT_TEST(TestHolderPtrMoveConstructor);
    UNIT_TEST(TestHolderPtrMoveConstructorInheritance);
    UNIT_TEST(TestHolderPtrMoveAssignment);
    UNIT_TEST(TestHolderPtrMoveAssignmentInheritance);
    UNIT_TEST(TestMakeHolder);
    UNIT_TEST(TestTrulePtr);
    UNIT_TEST(TestAutoToHolder);
    UNIT_TEST(TestCopyPtr);
    UNIT_TEST(TestIntrPtr);
    UNIT_TEST(TestIntrusiveConvertion);
    UNIT_TEST(TestIntrusiveConstConvertion);
    UNIT_TEST(TestIntrusiveConstConstruction);
    UNIT_TEST(TestMakeIntrusive);
    UNIT_TEST(TestCopyOnWritePtr1);
    UNIT_TEST(TestCopyOnWritePtr2);
    UNIT_TEST(TestOperatorBool);
    UNIT_TEST(TestMakeShared);
    UNIT_TEST(TestComparison);
    UNIT_TEST(TestSimpleIntrusivePtrCtorTsan);
    UNIT_TEST(TestRefCountedPtrsInHashSet);
    UNIT_TEST(TestSharedPtrDowncast);
    UNIT_TEST(TestStdCompatibility);
    UNIT_TEST_SUITE_END();

private:
    void TestSimpleIntrusivePtrCtorTsan() {
        struct S: public TAtomicRefCount<S> {
        };

        struct TLocalThread: public ISimpleThread {
            void* ThreadProc() override {
                TSimpleIntrusivePtr<S> ptr;
                return nullptr;
            }
        };

        // Create TSimpleIntrusivePtr in different threads
        // Its constructor should be thread-safe

        TLocalThread t1, t2;

        t1.Start();
        t2.Start();
        t1.Join();
        t2.Join();
    }

    inline void TestTypedefs() {
        TAtomicSharedPtr<int>(new int(1));
        TSimpleSharedPtr<int>(new int(1));
    }

    void TestSimpleIntrPtr();
    void TestHolderPtr();
    void TestHolderPtrMoveConstructor();
    void TestHolderPtrMoveConstructorInheritance();
    void TestHolderPtrMoveAssignment();
    void TestHolderPtrMoveAssignmentInheritance();
    void TestMakeHolder();
    void TestTrulePtr();
    void TestAutoToHolder();
    void TestCopyPtr();
    void TestIntrPtr();
    void TestIntrusiveConvertion();
    void TestIntrusiveConstConvertion();
    void TestIntrusiveConstConstruction();
    void TestMakeIntrusive();
    void TestCopyOnWritePtr1();
    void TestCopyOnWritePtr2();
    void TestOperatorBool();
    void TestMakeShared();
    void TestComparison();
    template <class T, class TRefCountedPtr>
    void TestRefCountedPtrsInHashSetImpl();
    void TestRefCountedPtrsInHashSet();
    void TestSharedPtrDowncast();
    void TestStdCompatibility();
};

UNIT_TEST_SUITE_REGISTRATION(TPointerTest);

static int cnt = 0;

class A: public TAtomicRefCount<A> {
public:
    inline A() {
        ++cnt;
    }

    inline A(const A&)
        : TAtomicRefCount<A>(*this)
    {
        ++cnt;
    }

    inline ~A() {
        --cnt;
    }
};

static A* MakeA() {
    return new A();
}

/*
 * test compileability
 */
class B;
static TSimpleIntrusivePtr<B> GetB() {
    throw 1;
}

void Func() {
    TSimpleIntrusivePtr<B> b = GetB();
}

void TPointerTest::TestSimpleIntrPtr() {
    {
        TSimpleIntrusivePtr<A> a1(MakeA());
        TSimpleIntrusivePtr<A> a2(MakeA());
        TSimpleIntrusivePtr<A> a3 = a2;

        a1 = a2;
        a2 = a3;
    }

    UNIT_ASSERT_VALUES_EQUAL(cnt, 0);
}

void TPointerTest::TestHolderPtr() {
    {
        THolder<A> a1(MakeA());
        THolder<A> a2(a1.Release());
    }

    UNIT_ASSERT_VALUES_EQUAL(cnt, 0);
}

THolder<int> CreateInt(int value) {
    THolder<int> res(new int);
    *res = value;
    return res;
}

void TPointerTest::TestHolderPtrMoveConstructor() {
    THolder<int> h = CreateInt(42);
    UNIT_ASSERT_VALUES_EQUAL(*h, 42);
}

void TPointerTest::TestHolderPtrMoveAssignment() {
    THolder<int> h(new int);
    h = CreateInt(42);
    UNIT_ASSERT_VALUES_EQUAL(*h, 42);
}

struct TBase {
    virtual ~TBase() = default;
};

struct TDerived: public TBase {
};

void TPointerTest::TestHolderPtrMoveConstructorInheritance() {
    // compileability test
    THolder<TBase> basePtr(THolder<TDerived>(new TDerived));
}

void TPointerTest::TestHolderPtrMoveAssignmentInheritance() {
    // compileability test
    THolder<TBase> basePtr;
    basePtr = THolder<TDerived>(new TDerived);
}

void TPointerTest::TestMakeHolder() {
    {
        auto ptr = MakeHolder<int>(5);
        UNIT_ASSERT_VALUES_EQUAL(*ptr, 5);
    }
    {
        struct TRec {
            int X, Y;
            TRec()
                : X(1)
                , Y(2)
            {
            }
        };
        THolder<TRec> ptr = MakeHolder<TRec>();
        UNIT_ASSERT_VALUES_EQUAL(ptr->X, 1);
        UNIT_ASSERT_VALUES_EQUAL(ptr->Y, 2);
    }
    {
        struct TRec {
            int X, Y;
            TRec(int x, int y)
                : X(x)
                , Y(y)
            {
            }
        };
        auto ptr = MakeHolder<TRec>(1, 2);
        UNIT_ASSERT_VALUES_EQUAL(ptr->X, 1);
        UNIT_ASSERT_VALUES_EQUAL(ptr->Y, 2);
    }
    {
        class TRec {
        private:
            int X_, Y_;

        public:
            TRec(int x, int y)
                : X_(x)
                , Y_(y)
            {
            }

            int GetX() const {
                return X_;
            }
            int GetY() const {
                return Y_;
            }
        };
        auto ptr = MakeHolder<TRec>(1, 2);
        UNIT_ASSERT_VALUES_EQUAL(ptr->GetX(), 1);
        UNIT_ASSERT_VALUES_EQUAL(ptr->GetY(), 2);
    }
}

void TPointerTest::TestTrulePtr() {
    {
        TAutoPtr<A> a1(MakeA());
        TAutoPtr<A> a2(a1);
        a1 = a2;
    }

    UNIT_ASSERT_VALUES_EQUAL(cnt, 0);
}

void TPointerTest::TestAutoToHolder() {
    {
        TAutoPtr<A> a1(MakeA());
        THolder<A> a2(a1);

        UNIT_ASSERT_EQUAL(a1.Get(), nullptr);
        UNIT_ASSERT_VALUES_EQUAL(cnt, 1);
    }

    UNIT_ASSERT_VALUES_EQUAL(cnt, 0);

    {
        TAutoPtr<A> x(new A());
        THolder<const A> y = x;
    }

    UNIT_ASSERT_VALUES_EQUAL(cnt, 0);

    {
        class B1: public A {
        };

        TAutoPtr<B1> x(new B1());
        THolder<A> y = x;
    }

    UNIT_ASSERT_VALUES_EQUAL(cnt, 0);
}

void TPointerTest::TestCopyPtr() {
    TCopyPtr<A> a1(MakeA());
    {
        TCopyPtr<A> a2(MakeA());
        TCopyPtr<A> a3 = a2;
        UNIT_ASSERT_VALUES_EQUAL(cnt, 3);

        a1 = a2;
        a2 = a3;
    }
    UNIT_ASSERT_VALUES_EQUAL(cnt, 1);
    a1.Destroy();

    UNIT_ASSERT_VALUES_EQUAL(cnt, 0);
}

class TOp: public TSimpleRefCount<TOp>, public TNonCopyable {
public:
    static int Cnt;

public:
    TOp() {
        ++Cnt;
    }
    virtual ~TOp() {
        --Cnt;
    }
};

int TOp::Cnt = 0;

class TOp2: public TOp {
public:
    TIntrusivePtr<TOp> Op;

public:
    TOp2(const TIntrusivePtr<TOp>& op)
        : Op(op)
    {
        ++Cnt;
    }
    ~TOp2() override {
        --Cnt;
    }
};

class TOp3 {
public:
    TIntrusivePtr<TOp2> Op2;
};

void Attach(TOp3* op3, TIntrusivePtr<TOp>* op) {
    TIntrusivePtr<TOp2> op2 = new TOp2(*op);
    op3->Op2 = op2.Get();
    *op = op2.Get();
}

void TPointerTest::TestIntrPtr() {
    {
        TIntrusivePtr<TOp> p, p2;
        TOp3 op3;
        {
            TVector<TIntrusivePtr<TOp>> f1;
            {
                TVector<TIntrusivePtr<TOp>> f2;
                f2.push_back(new TOp);
                p = new TOp;
                f2.push_back(p);
                Attach(&op3, &f2[1]);
                f1 = f2;
                UNIT_ASSERT_VALUES_EQUAL(f1[0]->RefCount(), 2);
                UNIT_ASSERT_VALUES_EQUAL(f1[1]->RefCount(), 3);
                UNIT_ASSERT_EQUAL(f1[1].Get(), op3.Op2.Get());
                UNIT_ASSERT_VALUES_EQUAL(op3.Op2->RefCount(), 3);
                UNIT_ASSERT_VALUES_EQUAL(op3.Op2->Op->RefCount(), 2);
                UNIT_ASSERT_VALUES_EQUAL(TOp::Cnt, 4);
            }
            p2 = p;
        }
        UNIT_ASSERT_VALUES_EQUAL(op3.Op2->RefCount(), 1);
        UNIT_ASSERT_VALUES_EQUAL(op3.Op2->Op->RefCount(), 3);
        UNIT_ASSERT_VALUES_EQUAL(TOp::Cnt, 3);
    }
    UNIT_ASSERT_VALUES_EQUAL(TOp::Cnt, 0);
}

namespace NTestIntrusiveConvertion {
    struct TA: public TSimpleRefCount<TA> {
    };
    struct TAA: public TA {
    };
    struct TB: public TSimpleRefCount<TB> {
    };

    void Func(TIntrusivePtr<TA>) {
    }

    void Func(TIntrusivePtr<TB>) {
    }

    void Func(TIntrusiveConstPtr<TA>) {
    }

    void Func(TIntrusiveConstPtr<TB>) {
    }
} // namespace NTestIntrusiveConvertion

void TPointerTest::TestIntrusiveConvertion() {
    using namespace NTestIntrusiveConvertion;

    TIntrusivePtr<TAA> aa = new TAA;

    UNIT_ASSERT_VALUES_EQUAL(aa->RefCount(), 1);
    TIntrusivePtr<TA> a = aa;
    UNIT_ASSERT_VALUES_EQUAL(aa->RefCount(), 2);
    UNIT_ASSERT_VALUES_EQUAL(a->RefCount(), 2);
    aa.Reset();
    UNIT_ASSERT_VALUES_EQUAL(a->RefCount(), 1);

    // test that Func(TIntrusivePtr<TB>) doesn't participate in overload resolution
    Func(aa);
}

void TPointerTest::TestIntrusiveConstConvertion() {
    using namespace NTestIntrusiveConvertion;

    TIntrusiveConstPtr<TAA> aa = new TAA;

    UNIT_ASSERT_VALUES_EQUAL(aa->RefCount(), 1);
    TIntrusiveConstPtr<TA> a = aa;
    UNIT_ASSERT_VALUES_EQUAL(aa->RefCount(), 2);
    UNIT_ASSERT_VALUES_EQUAL(a->RefCount(), 2);
    aa.Reset();
    UNIT_ASSERT_VALUES_EQUAL(a->RefCount(), 1);

    // test that Func(TIntrusiveConstPtr<TB>) doesn't participate in overload resolution
    Func(aa);
}

void TPointerTest::TestMakeIntrusive() {
    {
        UNIT_ASSERT_VALUES_EQUAL(0, TOp::Cnt);
        auto p = MakeIntrusive<TOp>();
        UNIT_ASSERT_VALUES_EQUAL(1, p->RefCount());
        UNIT_ASSERT_VALUES_EQUAL(1, TOp::Cnt);
    }
    UNIT_ASSERT_VALUES_EQUAL(TOp::Cnt, 0);
}

void TPointerTest::TestCopyOnWritePtr1() {
    using TPtr = TCowPtr<TSimpleSharedPtr<int>>;
    TPtr p1;
    UNIT_ASSERT(!p1.Shared());

    p1.Reset(new int(123));
    UNIT_ASSERT(!p1.Shared());

    {
        TPtr pTmp = p1;

        UNIT_ASSERT(p1.Shared());
        UNIT_ASSERT(pTmp.Shared());
        UNIT_ASSERT_EQUAL(p1.Get(), pTmp.Get());
    }

    UNIT_ASSERT(!p1.Shared());

    TPtr p2 = p1;
    TPtr p3;
    p3 = p2;

    UNIT_ASSERT(p2.Shared());
    UNIT_ASSERT(p3.Shared());
    UNIT_ASSERT_EQUAL(p1.Get(), p2.Get());
    UNIT_ASSERT_EQUAL(p1.Get(), p3.Get());

    *(p1.Mutable()) = 456;

    UNIT_ASSERT(!p1.Shared());
    UNIT_ASSERT(p2.Shared());
    UNIT_ASSERT(p3.Shared());
    UNIT_ASSERT_EQUAL(*p1, 456);
    UNIT_ASSERT_EQUAL(*p2, 123);
    UNIT_ASSERT_EQUAL(*p3, 123);
    UNIT_ASSERT_UNEQUAL(p1.Get(), p2.Get());
    UNIT_ASSERT_EQUAL(p2.Get(), p3.Get());

    p2.Mutable();

    UNIT_ASSERT(!p2.Shared());
    UNIT_ASSERT(!p3.Shared());
    UNIT_ASSERT_EQUAL(*p2, 123);
    UNIT_ASSERT_EQUAL(*p3, 123);
    UNIT_ASSERT_UNEQUAL(p2.Get(), p3.Get());
}

struct X: public TSimpleRefCount<X> {
    inline X(int v = 0)
        : V(v)
    {
    }

    int V;
};

void TPointerTest::TestCopyOnWritePtr2() {
    using TPtr = TCowPtr<TIntrusivePtr<X>>;
    TPtr p1;
    UNIT_ASSERT(!p1.Shared());

    p1.Reset(new X(123));
    UNIT_ASSERT(!p1.Shared());

    {
        TPtr pTmp = p1;

        UNIT_ASSERT(p1.Shared());
        UNIT_ASSERT(pTmp.Shared());
        UNIT_ASSERT_EQUAL(p1.Get(), pTmp.Get());
    }

    UNIT_ASSERT(!p1.Shared());

    TPtr p2 = p1;
    TPtr p3;
    p3 = p2;

    UNIT_ASSERT(p2.Shared());
    UNIT_ASSERT(p3.Shared());
    UNIT_ASSERT_EQUAL(p1.Get(), p2.Get());
    UNIT_ASSERT_EQUAL(p1.Get(), p3.Get());

    p1.Mutable()->V = 456;

    UNIT_ASSERT(!p1.Shared());
    UNIT_ASSERT(p2.Shared());
    UNIT_ASSERT(p3.Shared());
    UNIT_ASSERT_EQUAL(p1->V, 456);
    UNIT_ASSERT_EQUAL(p2->V, 123);
    UNIT_ASSERT_EQUAL(p3->V, 123);
    UNIT_ASSERT_UNEQUAL(p1.Get(), p2.Get());
    UNIT_ASSERT_EQUAL(p2.Get(), p3.Get());

    p2.Mutable();

    UNIT_ASSERT(!p2.Shared());
    UNIT_ASSERT(!p3.Shared());
    UNIT_ASSERT_EQUAL(p2->V, 123);
    UNIT_ASSERT_EQUAL(p3->V, 123);
    UNIT_ASSERT_UNEQUAL(p2.Get(), p3.Get());
}

namespace {
    template <class TFrom, class TTo>
    struct TImplicitlyCastable {
        struct RTYes {
            char t[2];
        };

        using RTNo = char;

        static RTYes Func(TTo);
        static RTNo Func(...);
        static TFrom Get();

        /*
         * Result == (TFrom could be converted to TTo implicitly)
         */
        enum {
            Result = (sizeof(Func(Get())) != sizeof(RTNo))
        };
    };

    struct TImplicitlyCastableToBool {
        inline operator bool() const {
            return true;
        }
    };

} // namespace

void TPointerTest::TestOperatorBool() {
    using TVec = TVector<ui32>;

    // to be sure TImplicitlyCastable works as expected
    UNIT_ASSERT((TImplicitlyCastable<int, bool>::Result));
    UNIT_ASSERT((TImplicitlyCastable<double, int>::Result));
    UNIT_ASSERT((TImplicitlyCastable<int*, void*>::Result));
    UNIT_ASSERT(!(TImplicitlyCastable<void*, int*>::Result));
    UNIT_ASSERT((TImplicitlyCastable<TImplicitlyCastableToBool, bool>::Result));
    UNIT_ASSERT((TImplicitlyCastable<TImplicitlyCastableToBool, int>::Result));
    UNIT_ASSERT((TImplicitlyCastable<TImplicitlyCastableToBool, ui64>::Result));
    UNIT_ASSERT(!(TImplicitlyCastable<TImplicitlyCastableToBool, void*>::Result));

    // pointers
    UNIT_ASSERT(!(TImplicitlyCastable<TSimpleSharedPtr<TVec>, int>::Result));
    UNIT_ASSERT(!(TImplicitlyCastable<TAutoPtr<ui64>, ui64>::Result));
    UNIT_ASSERT(!(TImplicitlyCastable<THolder<TVec>, bool>::Result)); // even this

    {
        // mostly a compilability test

        THolder<TVec> a;
        UNIT_ASSERT(!a);
        UNIT_ASSERT(!bool(a));
        if (a) {
            UNIT_ASSERT(false);
        }
        if (!a) {
            UNIT_ASSERT(true);
        }

        a.Reset(new TVec);
        UNIT_ASSERT(a);
        UNIT_ASSERT(bool(a));
        if (a) {
            UNIT_ASSERT(true);
        }
        if (!a) {
            UNIT_ASSERT(false);
        }

        THolder<TVec> b(new TVec);
        UNIT_ASSERT(a.Get() != b.Get());
        UNIT_ASSERT(a != b);
        if (a == b) {
            UNIT_ASSERT(false);
        }
        if (a != b) {
            UNIT_ASSERT(true);
        }
        if (!(a && b)) {
            UNIT_ASSERT(false);
        }
        if (a && b) {
            UNIT_ASSERT(true);
        }

        // int i = a;          // does not compile
        // bool c = (a < b);   // does not compile
    }
}

void TPointerTest::TestMakeShared() {
    {
        TSimpleSharedPtr<int> ptr = MakeSimpleShared<int>(5);
        UNIT_ASSERT_VALUES_EQUAL(*ptr, 5);
    }
    {
        struct TRec {
            int X, Y;
            TRec()
                : X(1)
                , Y(2)
            {
            }
        };
        auto ptr = MakeAtomicShared<TRec>();
        UNIT_ASSERT_VALUES_EQUAL(ptr->X, 1);
        UNIT_ASSERT_VALUES_EQUAL(ptr->Y, 2);
    }
    {
        struct TRec {
            int X, Y;
        };
        TAtomicSharedPtr<TRec> ptr = MakeAtomicShared<TRec>(1, 2);
        UNIT_ASSERT_VALUES_EQUAL(ptr->X, 1);
        UNIT_ASSERT_VALUES_EQUAL(ptr->Y, 2);
    }
    {
        class TRec {
        private:
            int X_, Y_;

        public:
            TRec(int x, int y)
                : X_(x)
                , Y_(y)
            {
            }

            int GetX() const {
                return X_;
            }
            int GetY() const {
                return Y_;
            }
        };
        TSimpleSharedPtr<TRec> ptr = MakeSimpleShared<TRec>(1, 2);
        UNIT_ASSERT_VALUES_EQUAL(ptr->GetX(), 1);
        UNIT_ASSERT_VALUES_EQUAL(ptr->GetY(), 2);
    }
    {
        enum EObjectState {
            OS_NOT_CREATED,
            OS_CREATED,
            OS_DESTROYED,
        };

        struct TObject {
            EObjectState& State;

            TObject(EObjectState& state)
                : State(state)
            {
                State = OS_CREATED;
            }

            ~TObject() {
                State = OS_DESTROYED;
            }
        };

        auto throwsException = []() {
            throw yexception();
            return 5;
        };

        auto testFunction = [](TSimpleSharedPtr<TObject>, int) {
        };

        EObjectState state = OS_NOT_CREATED;
        try {
            testFunction(MakeSimpleShared<TObject>(state), throwsException());
        } catch (yexception&) {
        }

        UNIT_ASSERT(state == OS_NOT_CREATED || state == OS_DESTROYED);
    }
}

template <class TPtr>
void TestPtrComparison(const TPtr& ptr) {
    UNIT_ASSERT(ptr == ptr);
    UNIT_ASSERT(!(ptr != ptr));
    UNIT_ASSERT(ptr == ptr.Get());
    UNIT_ASSERT(!(ptr != ptr.Get()));
}

void TPointerTest::TestComparison() {
    THolder<A> ptr1(new A);
    TAutoPtr<A> ptr2;
    TSimpleSharedPtr<int> ptr3(new int(6));
    TIntrusivePtr<A> ptr4;
    TIntrusiveConstPtr<A> ptr5 = ptr4;

    UNIT_ASSERT(ptr1 != nullptr);
    UNIT_ASSERT(ptr2 == nullptr);
    UNIT_ASSERT(ptr3 != nullptr);
    UNIT_ASSERT(ptr4 == nullptr);
    UNIT_ASSERT(ptr5 == nullptr);

    TestPtrComparison(ptr1);
    TestPtrComparison(ptr2);
    TestPtrComparison(ptr3);
    TestPtrComparison(ptr4);
    TestPtrComparison(ptr5);
}

template <class T, class TRefCountedPtr>
void TPointerTest::TestRefCountedPtrsInHashSetImpl() {
    THashSet<TRefCountedPtr> hashSet;
    TRefCountedPtr p1(new T());
    UNIT_ASSERT(!IsIn(hashSet, p1));
    UNIT_ASSERT(hashSet.insert(p1).second);
    UNIT_ASSERT(IsIn(hashSet, p1));
    UNIT_ASSERT_VALUES_EQUAL(hashSet.size(), 1);
    UNIT_ASSERT(!hashSet.insert(p1).second);

    TRefCountedPtr p2(new T());
    UNIT_ASSERT(!IsIn(hashSet, p2));
    UNIT_ASSERT(hashSet.insert(p2).second);
    UNIT_ASSERT(IsIn(hashSet, p2));
    UNIT_ASSERT_VALUES_EQUAL(hashSet.size(), 2);
}

struct TCustomIntrusivePtrOps: TDefaultIntrusivePtrOps<A> {
};

struct TCustomDeleter: TDelete {
};

struct TCustomCounter: TSimpleCounter {
    using TSimpleCounterTemplate::TSimpleCounterTemplate;
};

void TPointerTest::TestRefCountedPtrsInHashSet() {
    // test common case
    TestRefCountedPtrsInHashSetImpl<TString, TSimpleSharedPtr<TString>>();
    TestRefCountedPtrsInHashSetImpl<TString, TAtomicSharedPtr<TString>>();
    TestRefCountedPtrsInHashSetImpl<A, TIntrusivePtr<A>>();
    TestRefCountedPtrsInHashSetImpl<A, TIntrusiveConstPtr<A>>();

    // test with custom ops
    TestRefCountedPtrsInHashSetImpl<TString, TSharedPtr<TString, TCustomCounter, TCustomDeleter>>();
    TestRefCountedPtrsInHashSetImpl<A, TIntrusivePtr<A, TCustomIntrusivePtrOps>>();
    TestRefCountedPtrsInHashSetImpl<A, TIntrusiveConstPtr<A, TCustomIntrusivePtrOps>>();
}

class TRefCountedWithStatistics: public TNonCopyable {
public:
    struct TExternalCounter {
        std::atomic<size_t> Counter{0};
        std::atomic<size_t> Increments{0};
    };

    TRefCountedWithStatistics(TExternalCounter& cnt)
        : ExternalCounter_(cnt)
    {
        // Reset counters
        ExternalCounter_.Counter.store(0);
        ExternalCounter_.Increments.store(0);
    }

    void Ref() noexcept {
        ++ExternalCounter_.Counter;
        ++ExternalCounter_.Increments;
    }

    void UnRef() noexcept {
        if (--ExternalCounter_.Counter == 0) {
            TDelete::Destroy(this);
        }
    }

    void DecRef() noexcept {
        Y_ABORT_UNLESS(--ExternalCounter_.Counter != 0);
    }

private:
    TExternalCounter& ExternalCounter_;
};

void TPointerTest::TestIntrusiveConstConstruction() {
    {
        TRefCountedWithStatistics::TExternalCounter cnt;
        UNIT_ASSERT_VALUES_EQUAL(cnt.Counter.load(), 0);
        UNIT_ASSERT_VALUES_EQUAL(cnt.Increments.load(), 0);
        TIntrusivePtr<TRefCountedWithStatistics> i{MakeIntrusive<TRefCountedWithStatistics>(cnt)};
        UNIT_ASSERT_VALUES_EQUAL(cnt.Counter.load(), 1);
        UNIT_ASSERT_VALUES_EQUAL(cnt.Increments.load(), 1);
        i.Reset();
        UNIT_ASSERT_VALUES_EQUAL(cnt.Counter.load(), 0);
        UNIT_ASSERT_VALUES_EQUAL(cnt.Increments.load(), 1);
    }
    {
        TRefCountedWithStatistics::TExternalCounter cnt;
        UNIT_ASSERT_VALUES_EQUAL(cnt.Counter.load(), 0);
        UNIT_ASSERT_VALUES_EQUAL(cnt.Increments.load(), 0);
        TIntrusiveConstPtr<TRefCountedWithStatistics> c{MakeIntrusive<TRefCountedWithStatistics>(cnt)};
        UNIT_ASSERT_VALUES_EQUAL(cnt.Counter.load(), 1);
        UNIT_ASSERT_VALUES_EQUAL(cnt.Increments.load(), 1);
        c.Reset();
        UNIT_ASSERT_VALUES_EQUAL(cnt.Counter.load(), 0);
        UNIT_ASSERT_VALUES_EQUAL(cnt.Increments.load(), 1);
    }
}

class TVirtualProbe: public NTesting::TProbe {
public:
    using NTesting::TProbe::TProbe;

    virtual ~TVirtualProbe() = default;
};

class TDerivedProbe: public TVirtualProbe {
public:
    using TVirtualProbe::TVirtualProbe;
};

class TDerivedProbeSibling: public TVirtualProbe {
public:
    using TVirtualProbe::TVirtualProbe;
};

void TPointerTest::TestSharedPtrDowncast() {
    {
        NTesting::TProbeState probeState = {};

        {
            TSimpleSharedPtr<TVirtualProbe> base = MakeSimpleShared<TDerivedProbe>(&probeState);
            UNIT_ASSERT_VALUES_EQUAL(probeState.Constructors, 1);

            {
                auto derived = base.As<TDerivedProbe>();
                UNIT_ASSERT_VALUES_EQUAL(probeState.Constructors, 1);

                UNIT_ASSERT_VALUES_EQUAL(base.Get(), derived.Get());
                UNIT_ASSERT_VALUES_EQUAL(base.ReferenceCounter(), derived.ReferenceCounter());

                UNIT_ASSERT_VALUES_EQUAL(base.RefCount(), 2l);
                UNIT_ASSERT_VALUES_EQUAL(derived.RefCount(), 2l);
            }

            UNIT_ASSERT_VALUES_EQUAL(probeState.Destructors, 0);
        }

        UNIT_ASSERT_VALUES_EQUAL(probeState.Destructors, 1);
    }
    {
        NTesting::TProbeState probeState = {};

        {
            TSimpleSharedPtr<TVirtualProbe> base = MakeSimpleShared<TDerivedProbe>(&probeState);
            UNIT_ASSERT_VALUES_EQUAL(probeState.Constructors, 1);

            auto derived = std::move(base).As<TDerivedProbe>();
            UNIT_ASSERT_VALUES_EQUAL(probeState.Constructors, 1);
            UNIT_ASSERT_VALUES_EQUAL(probeState.CopyConstructors, 0);
            UNIT_ASSERT_VALUES_EQUAL(probeState.Destructors, 0);
        }

        UNIT_ASSERT_VALUES_EQUAL(probeState.Destructors, 1);
    }
    {
        NTesting::TProbeState probeState = {};

        {
            TSimpleSharedPtr<TVirtualProbe> base = MakeSimpleShared<TDerivedProbe>(&probeState);
            UNIT_ASSERT_VALUES_EQUAL(probeState.Constructors, 1);

            {
                auto derivedSibling = base.As<TDerivedProbeSibling>();
                UNIT_ASSERT_VALUES_EQUAL(probeState.Constructors, 1);

                UNIT_ASSERT_VALUES_EQUAL(derivedSibling.Get(), nullptr);
                UNIT_ASSERT_VALUES_UNEQUAL(base.ReferenceCounter(), derivedSibling.ReferenceCounter());

                UNIT_ASSERT_VALUES_EQUAL(base.RefCount(), 1l);
                UNIT_ASSERT_VALUES_EQUAL(derivedSibling.RefCount(), 0l);
            }

            UNIT_ASSERT_VALUES_EQUAL(probeState.Destructors, 0);
        }

        UNIT_ASSERT_VALUES_EQUAL(probeState.Destructors, 1);
    }
    {
        NTesting::TProbeState probeState = {};

        {
            TSimpleSharedPtr<TVirtualProbe> base = MakeSimpleShared<TDerivedProbe>(&probeState);
            UNIT_ASSERT_VALUES_EQUAL(probeState.Constructors, 1);

            auto derived = std::move(base).As<TDerivedProbeSibling>();
            UNIT_ASSERT_VALUES_EQUAL(derived.Get(), nullptr);
            UNIT_ASSERT_VALUES_EQUAL(probeState.Constructors, 1);
            UNIT_ASSERT_VALUES_EQUAL(probeState.CopyConstructors, 0);
            UNIT_ASSERT_VALUES_EQUAL(probeState.Destructors, 0);
        }

        UNIT_ASSERT_VALUES_EQUAL(probeState.Destructors, 1);
    }
}

void TPointerTest::TestStdCompatibility() {
    {
        TSimpleSharedPtr<int> ptr = MakeSimpleShared<int>(5);
        UNIT_ASSERT_TYPES_EQUAL(decltype(ptr)::element_type, int);
        UNIT_ASSERT_VALUES_EQUAL(ptr.get(), ptr.Get());
    }

    {
        TAtomicSharedPtr<int> ptr = MakeAtomicShared<int>(5);
        UNIT_ASSERT_TYPES_EQUAL(decltype(ptr)::element_type, int);
        UNIT_ASSERT_VALUES_EQUAL(ptr.get(), ptr.Get());
    }

    {
        TAutoPtr<int> ptr = MakeHolder<int>(5);
        UNIT_ASSERT_TYPES_EQUAL(decltype(ptr)::element_type, int);
        UNIT_ASSERT_VALUES_EQUAL(ptr.get(), ptr.Get());
    }

    {
        TIntrusivePtr<TOp> ptr;
        UNIT_ASSERT_TYPES_EQUAL(decltype(ptr)::element_type, TOp);
        UNIT_ASSERT_VALUES_EQUAL(ptr.get(), ptr.Get());
    }
}
