#include <library/cpp/testing/gtest/gtest.h>

#include <library/cpp/yt/memory/new.h>
#include <library/cpp/yt/memory/ref_counted.h>
#include <library/cpp/yt/memory/atomic_intrusive_ptr.h>

namespace NYT {
namespace {

////////////////////////////////////////////////////////////////////////////////

using ::testing::IsNull;
using ::testing::NotNull;
using ::testing::InSequence;
using ::testing::MockFunction;
using ::testing::StrictMock;

////////////////////////////////////////////////////////////////////////////////
// Auxiliary types and functions.
////////////////////////////////////////////////////////////////////////////////

// This object tracks number of increments and decrements
// to the reference counter (see traits specialization below).
struct TIntricateObject
    : private TNonCopyable
{
    mutable int Increments = 0;
    mutable int Decrements = 0;
    mutable int Zeros = 0;

    void Ref(int n) const
    {
        Increments += n;
    }

    void Unref(int n) const
    {
        Decrements += n;
        if (Increments == Decrements) {
            ++Zeros;
        }
    }
};

using TIntricateObjectPtr = TIntrusivePtr<TIntricateObject>;
using TConstIntricateObjectPtr = TIntrusivePtr<const TIntricateObject>;

void Ref(TIntricateObject* obj, int n = 1)
{
    obj->Ref(n);
}

void Unref(TIntricateObject* obj, int n = 1)
{
    obj->Unref(n);
}

void Ref(const TIntricateObject* obj, int n = 1)
{
    obj->Ref(n);
}

void Unref(const TIntricateObject* obj, int n = 1)
{
    obj->Unref(n);
}

MATCHER_P3(HasRefCounts, increments, decrements, zeros,
    "Reference counter " \
    "was incremented " + ::testing::PrintToString(increments) + " times, " +
    "was decremented " + ::testing::PrintToString(decrements) + " times, " +
    "vanished to zero " + ::testing::PrintToString(zeros) + " times")
{
    Y_UNUSED(result_listener);
    return
        arg.Increments == increments &&
        arg.Decrements == decrements &&
        arg.Zeros == zeros;
}

void PrintTo(const TIntricateObject& arg, ::std::ostream* os)
{
    *os << arg.Increments << " increments, "
        << arg.Decrements << " decrements and "
        << arg.Zeros << " times vanished";
}

// This is an object which creates intrusive pointers to the self
// during its construction.
class TObjectWithSelfPointers
    : public TRefCounted
{
public:
    explicit TObjectWithSelfPointers(IOutputStream* output)
        : Output_(output)
    {
        *Output_ << "Cb";

        for (int i = 0; i < 3; ++i) {
            *Output_ << '!';
            TIntrusivePtr<TObjectWithSelfPointers> ptr(this);
        }

        *Output_ << "Ca";
    }

    virtual ~TObjectWithSelfPointers()
    {
        *Output_ << 'D';
    }

private:
    IOutputStream* const Output_;

};

// This is a simple object with simple reference counting.
class TObjectWithSimpleRC
    : public TRefCounted
{
public:
    explicit TObjectWithSimpleRC(IOutputStream* output)
        : Output_(output)
    {
        *Output_ << 'C';
    }

    virtual ~TObjectWithSimpleRC()
    {
        *Output_ << 'D';
    }

    void DoSomething()
    {
        *Output_ << '!';
    }

private:
    IOutputStream* const Output_;

};

// This is a simple object with full-fledged reference counting.
class TObjectWithFullRC
    : public TRefCounted
{
public:
    explicit TObjectWithFullRC(IOutputStream* output)
        : Output_(output)
    {
        *Output_ << 'C';
    }

    virtual ~TObjectWithFullRC()
    {
        *Output_ << 'D';
    }

    void DoSomething()
    {
        *Output_ << '!';
    }

private:
    IOutputStream* const Output_;

};

////////////////////////////////////////////////////////////////////////////////

TEST(TAtomicPtrTest, Empty)
{
    TIntricateObjectPtr emptyPointer;
    EXPECT_EQ(nullptr, emptyPointer.Get());
}

// Reserved ref count.
constexpr int RRC = 65535;

TEST(TAtomicPtrTest, Basic)
{
    TIntricateObject object;

    EXPECT_THAT(object, HasRefCounts(0, 0, 0));

    {
        TIntricateObjectPtr owningPointer(&object);
        EXPECT_THAT(object, HasRefCounts(1, 0, 0));
        EXPECT_EQ(&object, owningPointer.Get());
    }

    EXPECT_THAT(object, HasRefCounts(1, 1, 1));


    {
        TIntricateObjectPtr owningPointer(&object);
        TAtomicIntrusivePtr<TIntricateObject> atomicPointer(owningPointer);

        EXPECT_THAT(object, HasRefCounts(2 + RRC, 1, 1));
        EXPECT_EQ(&object, owningPointer.Get());


        auto p1 = atomicPointer.Acquire();

        EXPECT_THAT(object, HasRefCounts(2 + RRC, 1, 1));

        p1.Reset();

        EXPECT_THAT(object, HasRefCounts(2 + RRC, 2, 1));

        owningPointer.Reset();

        EXPECT_THAT(object, HasRefCounts(2 + RRC, 3, 1));
    }

    EXPECT_THAT(object, HasRefCounts(2 + RRC, 2 + RRC, 2));
}

TEST(TAtomicPtrTest, BasicConst)
{
    const TIntricateObject object;

    EXPECT_THAT(object, HasRefCounts(0, 0, 0));

    {
        TConstIntricateObjectPtr owningPointer(&object);
        EXPECT_THAT(object, HasRefCounts(1, 0, 0));
        EXPECT_EQ(&object, owningPointer.Get());
    }

    EXPECT_THAT(object, HasRefCounts(1, 1, 1));


    {
        TConstIntricateObjectPtr owningPointer(&object);
        TAtomicIntrusivePtr<const TIntricateObject> atomicPointer(owningPointer);

        EXPECT_THAT(object, HasRefCounts(2 + RRC, 1, 1));
        EXPECT_EQ(&object, owningPointer.Get());


        auto p1 = atomicPointer.Acquire();

        EXPECT_THAT(object, HasRefCounts(2 + RRC, 1, 1));

        p1.Reset();

        EXPECT_THAT(object, HasRefCounts(2 + RRC, 2, 1));

        owningPointer.Reset();

        EXPECT_THAT(object, HasRefCounts(2 + RRC, 3, 1));
    }

    EXPECT_THAT(object, HasRefCounts(2 + RRC, 2 + RRC, 2));
}

TEST(TAtomicPtrTest, Acquire)
{
    TIntricateObject object;
    {
        TAtomicIntrusivePtr<TIntricateObject> atomicPtr{TIntricateObjectPtr(&object)};
        EXPECT_THAT(object, HasRefCounts(RRC, 0, 0));

        for (int i = 0; i < RRC / 2; ++i) {
            {
                auto tmp = atomicPtr.Acquire();
                EXPECT_THAT(object, HasRefCounts(RRC, i, 0));
            }
            EXPECT_THAT(object, HasRefCounts(RRC, i + 1, 0));
        }

        {
            auto tmp = atomicPtr.Acquire();
            EXPECT_THAT(object, HasRefCounts( RRC + RRC / 2, RRC - 1, 0));
        }

        EXPECT_THAT(object, HasRefCounts(RRC + RRC / 2, RRC, 0));
    }

    EXPECT_THAT(object, HasRefCounts(RRC + RRC / 2, RRC + RRC / 2, 1));
}

TEST(TAtomicPtrTest, AcquireConst)
{
    const TIntricateObject object;
    {
        TAtomicIntrusivePtr<const TIntricateObject> atomicPtr{TConstIntricateObjectPtr(&object)};
        EXPECT_THAT(object, HasRefCounts(RRC, 0, 0));

        for (int i = 0; i < RRC / 2; ++i) {
            {
                auto tmp = atomicPtr.Acquire();
                EXPECT_THAT(object, HasRefCounts(RRC, i, 0));
            }
            EXPECT_THAT(object, HasRefCounts(RRC, i + 1, 0));
        }

        {
            auto tmp = atomicPtr.Acquire();
            EXPECT_THAT(object, HasRefCounts( RRC + RRC / 2, RRC - 1, 0));
        }

        EXPECT_THAT(object, HasRefCounts(RRC + RRC / 2, RRC, 0));
    }

    EXPECT_THAT(object, HasRefCounts(RRC + RRC / 2, RRC + RRC / 2, 1));
}

TEST(TAtomicPtrTest, CAS)
{
    TIntricateObject o1;
    TIntricateObject o2;
    {

        TAtomicIntrusivePtr<TIntricateObject> atomicPtr{TIntricateObjectPtr(&o1)};
        EXPECT_THAT(o1, HasRefCounts(RRC, 0, 0));

        TIntricateObjectPtr p2(&o2);
        EXPECT_THAT(o2, HasRefCounts(1, 0, 0));

        void* rawPtr = &o1;
        EXPECT_TRUE(atomicPtr.CompareAndSwap(rawPtr, std::move(p2)));
        EXPECT_EQ(rawPtr, &o1);

        EXPECT_THAT(o1, HasRefCounts(RRC, RRC, 1));
        EXPECT_THAT(o2, HasRefCounts(RRC, 0, 0));

        rawPtr = nullptr;
        EXPECT_FALSE(atomicPtr.CompareAndSwap(rawPtr, TIntricateObjectPtr(&o1)));
        EXPECT_EQ(rawPtr, &o2);

        EXPECT_THAT(o1, HasRefCounts(2 * RRC, 2 * RRC, 2));
        EXPECT_THAT(o2, HasRefCounts(RRC, 0, 0));
    }

    EXPECT_THAT(o2, HasRefCounts(RRC, RRC, 1));
}

TEST(TAtomicPtrTest, CASConst)
{
    const TIntricateObject o1;
    const TIntricateObject o2;
    {

        TAtomicIntrusivePtr<const TIntricateObject> atomicPtr{TConstIntricateObjectPtr(&o1)};
        EXPECT_THAT(o1, HasRefCounts(RRC, 0, 0));

        TConstIntricateObjectPtr p2(&o2);
        EXPECT_THAT(o2, HasRefCounts(1, 0, 0));

        const void* rawPtr = &o1;
        EXPECT_TRUE(atomicPtr.CompareAndSwap(rawPtr, std::move(p2)));
        EXPECT_EQ(rawPtr, &o1);

        EXPECT_THAT(o1, HasRefCounts(RRC, RRC, 1));
        EXPECT_THAT(o2, HasRefCounts(RRC, 0, 0));

        rawPtr = nullptr;
        EXPECT_FALSE(atomicPtr.CompareAndSwap(rawPtr, TConstIntricateObjectPtr(&o1)));
        EXPECT_EQ(rawPtr, &o2);

        EXPECT_THAT(o1, HasRefCounts(2 * RRC, 2 * RRC, 2));
        EXPECT_THAT(o2, HasRefCounts(RRC, 0, 0));
    }

    EXPECT_THAT(o2, HasRefCounts(RRC, RRC, 1));
}

////////////////////////////////////////////////////////////////////////////////

} // namespace
} // namespace NYT
