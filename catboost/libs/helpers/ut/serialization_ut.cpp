#include <catboost/libs/helpers/serialization.h>

#include <library/cpp/binsaver/util_stream_io.h>

#include <util/generic/string.h>
#include <util/stream/buffer.h>

#include <library/cpp/testing/unittest/registar.h>


Y_UNIT_TEST_SUITE(BinSaverSerialization) {
    template <class T>
    struct TValueWithRefCount : public TThrRefBase {
        T Value;

    public:
        TValueWithRefCount(const T& value = T())
            : Value(value)
        {}

        SAVELOAD(Value);
    };

    struct TWithPtrMembers1 {
        TIntrusivePtr<TValueWithRefCount<int>> IntrusiveInt;
        TVector<TIntrusivePtr<TValueWithRefCount<int>>> IntrusiveIntVector;
        THashMap<ui64, TAtomicSharedPtr<int>> SharedIntHashMap;
        THashMap<ui64, TIntrusivePtr<TValueWithRefCount<int>>> IntrusiveIntHashMap;

        TAtomicSharedPtr<int> SharedInt;

    public:
        SAVELOAD_WITH_SHARED(
            IntrusiveInt,
            IntrusiveIntVector,
            SharedIntHashMap,
            IntrusiveIntHashMap,
            SharedInt
        )

    };


    struct TWithPtrMembers2 {
        TIntrusivePtr<TValueWithRefCount<int>> IntrusiveInt;
        TIntrusivePtr<TValueWithRefCount<TString>> IntrusiveString;

        TVector<TIntrusivePtr<TValueWithRefCount<TString>>> IntrusiveStringVector;

        TAtomicSharedPtr<int> SharedInt;
        TAtomicSharedPtr<TString> SharedString;

        THolder<int> HolderInt;
        THolder<TString> HolderString;

    public:
        int operator&(IBinSaver& binSaver) {
            NCB::AddWithShared(&binSaver, &IntrusiveInt);
            NCB::AddWithShared(&binSaver, &IntrusiveString);
            NCB::AddWithShared(&binSaver, &IntrusiveStringVector);

            NCB::AddWithShared(&binSaver, &SharedInt);
            NCB::AddWithShared(&binSaver, &SharedString);

            binSaver.Add(0, &HolderInt);
            binSaver.Add(0, &HolderString);

            return 0;
        }
    };

    Y_UNIT_TEST(Test1) {
        TBuffer buffer;

        {
            TWithPtrMembers1 object;
            object.IntrusiveInt = MakeIntrusive<TValueWithRefCount<int>>(42);
            object.IntrusiveIntVector = {
                MakeIntrusive<TValueWithRefCount<int>>(1),
                MakeIntrusive<TValueWithRefCount<int>>(2)
            };
            object.SharedIntHashMap = {
                {30, MakeAtomicShared<int>(10)},
                {25, MakeAtomicShared<int>(9)}
            };
            object.IntrusiveIntHashMap = {
                {10, MakeIntrusive<TValueWithRefCount<int>>(7)},
                {12, MakeIntrusive<TValueWithRefCount<int>>(3)}
            };

            object.SharedInt = MakeAtomicShared<int>(42);

            TBufferOutput out(buffer);
            SerializeToArcadiaStream(out, object);
        }

        {
            TWithPtrMembers1 object;

            TBufferInput in(buffer);
            SerializeFromStream(in, object);

            UNIT_ASSERT(object.IntrusiveInt);
            UNIT_ASSERT_VALUES_EQUAL(object.IntrusiveInt->Value, 42);

            UNIT_ASSERT_VALUES_EQUAL(object.IntrusiveIntVector.size(), 2);
            UNIT_ASSERT_VALUES_EQUAL(object.IntrusiveIntVector[0]->Value, 1);
            UNIT_ASSERT_VALUES_EQUAL(object.IntrusiveIntVector[1]->Value, 2);

            UNIT_ASSERT_VALUES_EQUAL(object.SharedIntHashMap.size(), 2);
            UNIT_ASSERT_VALUES_EQUAL(*object.SharedIntHashMap[30], 10);
            UNIT_ASSERT_VALUES_EQUAL(*object.SharedIntHashMap[25], 9);

            UNIT_ASSERT_VALUES_EQUAL(object.IntrusiveIntHashMap.size(), 2);
            UNIT_ASSERT_VALUES_EQUAL(object.IntrusiveIntHashMap[10]->Value, 7);
            UNIT_ASSERT_VALUES_EQUAL(object.IntrusiveIntHashMap[12]->Value, 3);

            UNIT_ASSERT(object.SharedInt);
            UNIT_ASSERT_VALUES_EQUAL(*object.SharedInt, 42);
        }
    }

    Y_UNIT_TEST(Test2) {
        TBuffer buffer;

        {
            TWithPtrMembers2 object;
            object.IntrusiveInt = MakeIntrusive<TValueWithRefCount<int>>(42);
            object.IntrusiveString = MakeIntrusive<TValueWithRefCount<TString>>("Sample");
            object.IntrusiveStringVector = {
                MakeIntrusive<TValueWithRefCount<TString>>("Sample1"),
                MakeIntrusive<TValueWithRefCount<TString>>("Sample2")
            };

            object.SharedInt = MakeAtomicShared<int>(42);
            object.SharedString = MakeAtomicShared<TString>("Sample");

            object.HolderInt = MakeHolder<int>(42);
            object.HolderString = MakeHolder<TString>("Sample");

            TBufferOutput out(buffer);
            SerializeToArcadiaStream(out, object);
        }

        {
            TWithPtrMembers2 object;

            TBufferInput in(buffer);
            SerializeFromStream(in, object);

            UNIT_ASSERT(object.IntrusiveInt);
            UNIT_ASSERT_VALUES_EQUAL(object.IntrusiveInt->Value, 42);

            UNIT_ASSERT(object.IntrusiveString);
            UNIT_ASSERT_VALUES_EQUAL(object.IntrusiveString->Value, "Sample");

            UNIT_ASSERT_VALUES_EQUAL(object.IntrusiveStringVector.size(), 2);
            UNIT_ASSERT_VALUES_EQUAL(object.IntrusiveStringVector[0]->Value, "Sample1");
            UNIT_ASSERT_VALUES_EQUAL(object.IntrusiveStringVector[1]->Value, "Sample2");

            UNIT_ASSERT(object.SharedInt);
            UNIT_ASSERT_VALUES_EQUAL(*object.SharedInt, 42);

            UNIT_ASSERT(object.SharedString);
            UNIT_ASSERT_VALUES_EQUAL(*object.SharedString, "Sample");

            UNIT_ASSERT(object.HolderInt);
            UNIT_ASSERT_VALUES_EQUAL(*object.HolderInt, 42);

            UNIT_ASSERT(object.HolderString);
            UNIT_ASSERT_VALUES_EQUAL(*object.HolderString, "Sample");
        }
    }


    template <class T>
    void TestSaveAndLoadArrayData(TVector<T>&& data) {
        TBuffer buffer;

        {
            TBufferOutput out(buffer);
            TYaStreamOutput out2(out);
            IBinSaver binSaver(out2, false);
            NCB::SaveArrayData<T>(data, &binSaver);
        }
        {
            TBufferInput in(buffer);
            TYaStreamInput in2(in);
            IBinSaver binSaver(in2, true);

            TVector<T> loadedData(data.size());
            NCB::LoadArrayData<T>(loadedData, &binSaver);

            UNIT_ASSERT_VALUES_EQUAL(data, loadedData);
        }
    }

    Y_UNIT_TEST(TestSaveAndLoadArrayData) {
        TestSaveAndLoadArrayData(TVector<ui32>());
        TestSaveAndLoadArrayData(TVector<TString>());
        TestSaveAndLoadArrayData(TVector<i64>{12, 8, 7, 11});
        TestSaveAndLoadArrayData(TVector<TString>{"awk", "sed", "find", "ls"});
    }
}
