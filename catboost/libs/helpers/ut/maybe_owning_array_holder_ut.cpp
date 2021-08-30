#include <catboost/libs/helpers/maybe_owning_array_holder.h>

#include <library/cpp/binsaver/util_stream_io.h>

#include <util/stream/buffer.h>
#include <util/generic/string.h>

#include <library/cpp/testing/unittest/registar.h>

struct TStringHolder : public NCB::IResourceHolder {
    TString S;
};


Y_UNIT_TEST_SUITE(TMaybeOwningArrayHolder) {
    Y_UNIT_TEST(TestNonOwning) {
        TVector<int> v{1, 2, 3};

        auto arrayHolder =  NCB::TMaybeOwningArrayHolder<int>::CreateNonOwning(v);

        UNIT_ASSERT_EQUAL(*arrayHolder, TArrayRef<int>(v));
    }

    Y_UNIT_TEST(TestGenericOwning) {
        auto stringHolder = MakeIntrusive<TStringHolder>();
        stringHolder->S = "string";

        auto arrayHolder = NCB::TMaybeOwningArrayHolder<const char>::CreateOwning(
            TConstArrayRef<char>(stringHolder->S.cbegin(), stringHolder->S.cend()),
            stringHolder
        );

        UNIT_ASSERT_EQUAL(*arrayHolder, (TConstArrayRef<char>({'s', 't', 'r', 'i', 'n', 'g'})));
    }

    Y_UNIT_TEST(TestVectorOwning) {
        TVector<TString> v{"aa", "bbb", "cccc", "d"};

        auto arrayHolder =  NCB::TMaybeOwningArrayHolder<TString>::CreateOwning(std::move(v));

        UNIT_ASSERT_EQUAL(*arrayHolder, (TArrayRef<TString>)(TVector<TString>{"aa", "bbb", "cccc", "d"}));
    }

    template <class T>
    void TestSaveAndLoad(NCB::TMaybeOwningArrayHolder<T>&& data) {
        TBuffer buffer;

        {
            TBufferOutput out(buffer);
            SerializeToArcadiaStream(out, data);
        }

        NCB::TMaybeOwningArrayHolder<T> loadedData;

        {
            TBufferInput in(buffer);
            SerializeFromStream(in, loadedData);
        }

        UNIT_ASSERT_EQUAL(*data, *loadedData);
    }

    Y_UNIT_TEST(TestBinSaverSerialization) {
        TestSaveAndLoad(NCB::TMaybeOwningArrayHolder<ui32>());
        TestSaveAndLoad(NCB::TMaybeOwningArrayHolder<TString>());
        TestSaveAndLoad(NCB::TMaybeOwningArrayHolder<i64>::CreateOwning(TVector<i64>{12, 8, 7, 11}));
        TestSaveAndLoad(
            NCB::TMaybeOwningArrayHolder<TString>::CreateOwning(TVector<TString>{"awk", "sed", "find", "ls"})
        );
    }
}
