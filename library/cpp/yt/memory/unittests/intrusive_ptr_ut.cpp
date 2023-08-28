#include <library/cpp/testing/gtest/gtest.h>

#include <library/cpp/yt/memory/new.h>
#include <library/cpp/yt/memory/ref_counted.h>
#include <library/cpp/yt/memory/serialize.h>

#include <util/generic/buffer.h>

#include <util/stream/buffer.h>

#include <util/ysaveload.h>

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
    int Increments = 0;
    int Decrements = 0;
    int Zeros = 0;

    void Ref()
    {
        ++Increments;
    }

    void Unref()
    {
        ++Decrements;
        if (Increments == Decrements) {
            ++Zeros;
        }
    }
};

using TIntricateObjectPtr = TIntrusivePtr<TIntricateObject>;

void Ref(TIntricateObject* obj, int /*n*/ = 1)
{
    obj->Ref();
}

void Unref(TIntricateObject* obj, int /*n*/ = 1)
{
    obj->Unref();
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

class TObjectWithExceptionInConstructor
    : public TRefCounted
{
public:
    TObjectWithExceptionInConstructor()
    {
        throw int(1);
    }

    volatile int Number = 0;
};

////////////////////////////////////////////////////////////////////////////////

TEST(TIntrusivePtrTest, Empty)
{
    TIntricateObjectPtr emptyPointer;
    EXPECT_EQ(nullptr, emptyPointer.Get());
}

TEST(TIntrusivePtrTest, Basic)
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
        TIntricateObjectPtr nonOwningPointer(&object, false);
        EXPECT_THAT(object, HasRefCounts(1, 1, 1));
        EXPECT_EQ(&object, nonOwningPointer.Get());
    }

    EXPECT_THAT(object, HasRefCounts(1, 2, 1));
}

TEST(TIntrusivePtrTest, ResetToNull)
{
    TIntricateObject object;
    TIntricateObjectPtr ptr(&object);

    EXPECT_THAT(object, HasRefCounts(1, 0, 0));
    EXPECT_EQ(&object, ptr.Get());

    ptr.Reset();

    EXPECT_THAT(object, HasRefCounts(1, 1, 1));
    EXPECT_EQ(nullptr, ptr.Get());
}

TEST(TIntrusivePtrTest, ResetToOtherObject)
{
    TIntricateObject firstObject;
    TIntricateObject secondObject;

    TIntricateObjectPtr ptr(&firstObject);

    EXPECT_THAT(firstObject, HasRefCounts(1, 0, 0));
    EXPECT_THAT(secondObject, HasRefCounts(0, 0, 0));
    EXPECT_EQ(&firstObject, ptr.Get());

    ptr.Reset(&secondObject);

    EXPECT_THAT(firstObject, HasRefCounts(1, 1, 1));
    EXPECT_THAT(secondObject, HasRefCounts(1, 0, 0));
    EXPECT_EQ(&secondObject, ptr.Get());
}

TEST(TIntrusivePtrTest, CopySemantics)
{
    TIntricateObject object;

    TIntricateObjectPtr foo(&object);
    EXPECT_THAT(object, HasRefCounts(1, 0, 0));

    {
        TIntricateObjectPtr bar(foo);
        EXPECT_THAT(object, HasRefCounts(2, 0, 0));
        EXPECT_EQ(&object, foo.Get());
        EXPECT_EQ(&object, bar.Get());
    }

    EXPECT_THAT(object, HasRefCounts(2, 1, 0));

    {
        TIntricateObjectPtr bar;
        bar = foo;

        EXPECT_THAT(object, HasRefCounts(3, 1, 0));
        EXPECT_EQ(&object, foo.Get());
        EXPECT_EQ(&object, bar.Get());
    }

    EXPECT_THAT(object, HasRefCounts(3, 2, 0));
}

TEST(TIntrusivePtrTest, MoveSemantics)
{
    TIntricateObject object;

    TIntricateObjectPtr foo(&object);
    EXPECT_THAT(object, HasRefCounts(1, 0, 0));

    {
        TIntricateObjectPtr bar(std::move(foo));
        EXPECT_THAT(object, HasRefCounts(1, 0, 0));
        EXPECT_THAT(foo.Get(), IsNull());
        EXPECT_EQ(&object, bar.Get());
    }

    EXPECT_THAT(object, HasRefCounts(1, 1, 1));
    foo.Reset(&object);
    EXPECT_THAT(object, HasRefCounts(2, 1, 1));

    {
        TIntricateObjectPtr bar;
        bar = std::move(foo);
        EXPECT_THAT(object, HasRefCounts(2, 1, 1));
        EXPECT_THAT(foo.Get(), IsNull());
        EXPECT_EQ(&object, bar.Get());
    }
}

TEST(TIntrusivePtrTest, Swap)
{
    TIntricateObject object;

    TIntricateObjectPtr foo(&object);
    TIntricateObjectPtr bar;

    EXPECT_THAT(object, HasRefCounts(1, 0, 0));
    EXPECT_THAT(foo.Get(), NotNull());
    EXPECT_THAT(bar.Get(), IsNull());

    foo.Swap(bar);

    EXPECT_THAT(object, HasRefCounts(1, 0, 0));
    EXPECT_THAT(foo.Get(), IsNull());
    EXPECT_THAT(bar.Get(), NotNull());

    foo.Swap(bar);

    EXPECT_THAT(object, HasRefCounts(1, 0, 0));
    EXPECT_THAT(foo.Get(), NotNull());
    EXPECT_THAT(bar.Get(), IsNull());
}

TEST(TIntrusivePtrTest, UpCast)
{
    //! This is a simple typical reference-counted object.
    class TSimpleObject
        : public TRefCounted
    { };

    //! This is a simple inherited reference-counted object.
    class TAnotherObject
        : public TSimpleObject
    { };

    auto foo = New<TSimpleObject>();
    auto bar = New<TAnotherObject>();
    auto baz = New<TAnotherObject>();

    foo = baz;

    EXPECT_TRUE(foo == baz);
}

TEST(TIntrusivePtrTest, DownCast)
{
    class TBaseObject
        : public TRefCounted
    { };

    class TDerivedObject
        : public TBaseObject
    { };

    //! This is a simple inherited reference-counted object.
    class TAnotherObject
        : public TBaseObject
    { };

    TIntrusivePtr<TBaseObject> foo = New<TDerivedObject>();
    TIntrusivePtr<TBaseObject> bar = New<TAnotherObject>();
    {
        auto baz = StaticPointerCast<TDerivedObject>(foo);
        EXPECT_TRUE(foo == baz);
    }
    {
        auto baz = StaticPointerCast<TDerivedObject>(TIntrusivePtr<TBaseObject>{foo});
        EXPECT_TRUE(foo == baz);
    }
    {
        auto baz = DynamicPointerCast<TDerivedObject>(foo);
        EXPECT_TRUE(foo == baz);
    }
    {
        auto baz = DynamicPointerCast<TDerivedObject>(bar);
        EXPECT_TRUE(nullptr == baz);
    }
    {
        auto baz = ConstPointerCast<const TBaseObject>(foo);
        EXPECT_TRUE(foo.Get() == baz.Get());
    }
    {
        auto baz = ConstPointerCast<const TBaseObject>(TIntrusivePtr<TBaseObject>{foo});
        EXPECT_TRUE(foo.Get() == baz.Get());
    }
}

TEST(TIntrusivePtrTest, UnspecifiedBoolType)
{
    TIntricateObject object;

    TIntricateObjectPtr foo;
    TIntricateObjectPtr bar(&object);

    EXPECT_FALSE(foo);
    EXPECT_TRUE(bar);
}

TEST(TIntrusivePtrTest, ObjectIsNotDestroyedPrematurely)
{
    TStringStream output;
    New<TObjectWithSelfPointers>(&output);

    // TObject... appends symbols to the output; see definitions.
    EXPECT_STREQ("Cb!!!CaD", output.Str().c_str());
}

TEST(TIntrusivePtrTest, EqualityOperator)
{
    TIntricateObject object, anotherObject;

    TIntricateObjectPtr emptyPointer;
    TIntricateObjectPtr somePointer(&object);
    TIntricateObjectPtr samePointer(&object);
    TIntricateObjectPtr anotherPointer(&anotherObject);

    EXPECT_FALSE(somePointer == emptyPointer);
    EXPECT_FALSE(samePointer == emptyPointer);

    EXPECT_TRUE(somePointer != emptyPointer);
    EXPECT_TRUE(samePointer != emptyPointer);

    EXPECT_TRUE(somePointer == samePointer);

    EXPECT_TRUE(&object == somePointer);
    EXPECT_TRUE(&object == samePointer);

    EXPECT_FALSE(somePointer == anotherPointer);
    EXPECT_TRUE(somePointer != anotherPointer);

    EXPECT_TRUE(&anotherObject == anotherPointer);
}

TEST(TIntrusivePtrTest, Reset)
{
    TIntricateObject object;
    TIntricateObjectPtr pointer(&object);
    EXPECT_THAT(object, HasRefCounts(1, 0, 0));
    EXPECT_EQ(&object, pointer.Release());
    EXPECT_THAT(object, HasRefCounts(1, 0, 0));
}

TEST(TIntrusivePtrTest, CompareWithNullptr)
{
    TIntricateObjectPtr pointer1;
    EXPECT_TRUE(nullptr == pointer1);
    EXPECT_FALSE(nullptr != pointer1);
    TIntricateObject object;
    TIntricateObjectPtr pointer2(&object);
    EXPECT_TRUE(pointer2 != nullptr);
    EXPECT_FALSE(pointer2 == nullptr);
}


template <class T>
void TestIntrusivePtrBehavior()
{
    using TMyPtr = TIntrusivePtr<T>;

    TStringStream output;
    {
        TMyPtr ptr(New<T>(&output));
        {
            TMyPtr anotherPtr(ptr);
            anotherPtr->DoSomething();
        }
        {
            TMyPtr anotherPtr(ptr);
            anotherPtr->DoSomething();
        }
        ptr->DoSomething();
    }

    // TObject... appends symbols to the output; see definitions.
    EXPECT_STREQ("C!!!D", output.Str().c_str());
}

TEST(TIntrusivePtrTest, SimpleRCBehaviour)
{
    TestIntrusivePtrBehavior<TObjectWithSimpleRC>();
}

TEST(TIntrusivePtrTest, FullRCBehaviour)
{
    TestIntrusivePtrBehavior<TObjectWithFullRC>();
}

TEST(TIntrusivePtrTest, ObjectAlignment)
{
    struct TObject
        : public TRefCounted
    {
        alignas(64) ui64 Data;
    };

    struct TPODObject final
    {
        alignas(64) ui64 Data;
    };

    auto foo = New<TObject>();
    auto bar = New<TPODObject>();

    EXPECT_TRUE(reinterpret_cast<uintptr_t>(foo.Get()) % 64 == 0);
    EXPECT_TRUE(reinterpret_cast<uintptr_t>(bar.Get()) % 64 == 0);
}

TEST(TIntrusivePtrTest, InitStruct)
{
    struct TObj1 final
    {
        const int A;
        const int B;
    };

    New<TObj1>(1, 2);

    struct TExplicitObj final
    {
        explicit TExplicitObj(int a = 0)
            : A(a)
        { }

        const int A;
    };

    New<TExplicitObj>();
    New<TExplicitObj>(1);

    struct TObj2 final
    {
        TObj2(i64 a = 0)
            : A(a)
        { }

        const i64 A;
    };

    New<TObj2>(123);

    struct TObj3 final
    {
        TObj3(ui64 a = 0)
            : A(a)
        { }

        const ui64 A;
    };

    New<TObj3>(123);

    struct TObj4 final
    {
        TObj4(int a, ui64 b = 0)
            : A(a)
            , B(b)
        { }

        int A;
        const ui64 B;
    };

    New<TObj4>(123);
    New<TObj4>(123, 123);

    struct TObj5 final
    {
        TExplicitObj E;
        int B;
    };

    New<TObj5>();

    struct TObj6 final
    {
        TObj2 O;
        int B;
    };

    New<TObj6>();
    New<TObj6>(1, 2);
}

TEST(TIntrusivePtrTest, Serialize)
{
    TBufferStream stream;

    struct TObject
        : public TRefCounted
    {
        ui64 Data;

        TObject()
            : Data(0)
        { }
        TObject(ui64 value)
            : Data(value)
        { }

        inline void Save(IOutputStream* out) const
        {
            ::Save(out, Data);
        }

        inline void Load(IInputStream* in)
        {
            ::Load(in, Data);
        }
    };

    ::Save(&stream, TIntrusivePtr<TObject>(nullptr));

    bool hasValue = true;
    ::Load(&stream, hasValue);
    EXPECT_FALSE(hasValue);
    EXPECT_EQ(stream.Buffer().Size(), 1ull);

    auto data = New<TObject>(42ull);
    ::Save(&stream, data);

    ::Load(&stream, hasValue);
    EXPECT_TRUE(hasValue);
    data->Data = 0;
    ::Load(&stream, data->Data);
    EXPECT_EQ(data->Data, 42ull);

    ::Save(&stream, data);
    TIntrusivePtr<TObject> ptr;
    ::Load(&stream, ptr);
    EXPECT_TRUE(ptr);
    EXPECT_EQ(data->Data, ptr->Data);

    ptr = nullptr;
    ::Save(&stream, TIntrusivePtr<TObject>(nullptr));
    ::Load(&stream, ptr);
    EXPECT_FALSE(ptr);
}

TEST(TIntrusivePtrTest, TestObjectConstructionFail)
{
    ASSERT_THROW(New<TObjectWithExceptionInConstructor>(), int);
}

////////////////////////////////////////////////////////////////////////////////

} // namespace
} // namespace NYT
