#include "enums.h"
#include "enums_with_header.h"
#include <tools/enum_parser/parse_enum/ut/enums_with_header.h_serialized.h>

#include "including_header.h"

// just to test that generated stuff works
#include <util/generic/serialized_enum.h>
#include <library/cpp/testing/unittest/registar.h>

#include <util/generic/ptr.h>
#include <util/generic/singleton.h>


void FunctionUsingEFwdEnum(EFwdEnum) {
}

class TEnumSerializationInitializer {
public:
    TEnumSerializationInitializer() {
        UNIT_ASSERT_VALUES_EQUAL(ToString(EDestructionPriorityTest::first), "first");
    }
    ~TEnumSerializationInitializer() {
        UNIT_ASSERT_VALUES_EQUAL(ToString(EDestructionPriorityTest::second), "second");
    }
};

class TEnumSerializationInitializerHolder {
public:
    TEnumSerializationInitializerHolder() {
    }

    ~TEnumSerializationInitializerHolder() {
    }

    void Init() { Ptr.Reset(new TEnumSerializationInitializer); }
private:
    THolder<TEnumSerializationInitializer> Ptr;
};


Y_UNIT_TEST_SUITE(TEnumGeneratorTest) {

    template<typename T>
    void CheckToString(const T& value, const TString& strValue) {
        UNIT_ASSERT_VALUES_EQUAL(ToString(value), strValue);
    }

    Y_UNIT_TEST(ToStringTest) {
        // ESimple
        CheckToString(Http, "Http");
        CheckToString(Https, "Https");
        CheckToString(ItemCount, "ItemCount");

        // ESimpleWithComma
        CheckToString(ESimpleWithComma::Http, "Http");
        CheckToString(ESimpleWithComma::Https, "Https");
        CheckToString(ESimpleWithComma::Http2, "Http"); // Http2 is an alias for Http
        CheckToString(ESimpleWithComma::ItemCount, "ItemCount");

        // ECustomAliases
        CheckToString(CAHttp, "http");
        CheckToString(CAHttps, "https");
        CheckToString(CAItemCount, "CAItemCount");

        // EMultipleAliases
        CheckToString(MAHttp, "http://");
        CheckToString(MAHttps, "https://");
        CheckToString(MAItemCount, "MAItemCount");

        // EDuplicateKeys
        CheckToString(Key0, "Key0");
        CheckToString(Key0Second, "Key0"); // obtain FIRST encountered value with such integer key
        CheckToString(Key1, "Key1");
        CheckToString(Key2, "k2");
        CheckToString(Key3, "k2"); // we CANNOT obtain "k3" here (as Key3 == Key2)
    }

    template<typename T>
    void CheckFromString(const TString& strValue, const T& value) {
        UNIT_ASSERT_VALUES_EQUAL(static_cast<int>(FromString<T>(TStringBuf(strValue))), static_cast<int>(value));
    }

    template<typename T>
    void CheckFromStringFail(const TString& strValue) {
        UNIT_ASSERT_EXCEPTION(FromString<T>(TStringBuf(strValue)), yexception);
    }

    template<typename T>
    void CheckTryFromString(const TString& strValue, const T& value) {
        T x;
        UNIT_ASSERT_VALUES_EQUAL(TryFromString(TStringBuf(strValue), x), true);
        UNIT_ASSERT_VALUES_EQUAL(x, value);
    }

    template<typename T>
    void CheckTryFromStringFail(const TString& strValue) {
        T x = T(-666);
        UNIT_ASSERT_VALUES_EQUAL(TryFromString(TStringBuf(strValue), x), false);
        UNIT_ASSERT_VALUES_EQUAL(int(x), -666);
    }

    Y_UNIT_TEST(TryFromStringTest) {
        // ESimple
        CheckFromString("Http", Http);
        CheckFromString("Https", Https);
        CheckFromString("ItemCount", ItemCount);
        CheckFromStringFail<ESimple>("ItemC0unt");

        CheckTryFromString("Http", Http);
        CheckTryFromString("Https", Https);
        CheckTryFromString("ItemCount", ItemCount);
        CheckTryFromStringFail<ESimple>("ItemC0unt");

        // ESimpleWithComma
        CheckTryFromString("Http", ESimpleWithComma::Http);
        CheckTryFromString("Https", ESimpleWithComma::Https);
        CheckTryFromString("ItemCount", ESimpleWithComma::ItemCount);
        CheckTryFromStringFail<ESimpleWithComma>("");

        // ECustomAliases
        CheckTryFromString("http", CAHttp);
        CheckTryFromString("https", CAHttps);
        CheckTryFromString("CAItemCount", CAItemCount);

        // EDuplicateKeys
        CheckTryFromString("Key0", Key0);
        CheckTryFromString("Key0Second", Key0Second);
        CheckTryFromString("Key1", Key1);
        CheckTryFromString("k2", Key2);
        CheckTryFromString("k2.1", Key2);
        CheckTryFromString("k3", Key3);
    }

    Y_UNIT_TEST(AllNamesValuesTest) {
        {
            auto allNames = GetEnumAllCppNames<EDuplicateKeys>();
            UNIT_ASSERT(!!allNames);
            UNIT_ASSERT_VALUES_EQUAL(allNames.size(), 5u);
            UNIT_ASSERT_VALUES_EQUAL(allNames[4], "Key3");
        }
        {
            auto allNames = GetEnumAllCppNames<ESimpleWithComma>();
            UNIT_ASSERT(!!allNames);
            UNIT_ASSERT_VALUES_EQUAL(allNames.size(), 4u);
            UNIT_ASSERT_VALUES_EQUAL(allNames[1], "ESimpleWithComma::Http2");
        }
    }

    Y_UNIT_TEST(EnumWithHeaderTest) {
        UNIT_ASSERT_VALUES_EQUAL(GetEnumItemsCount<EWithHeader>(), 3);
    }

    Y_UNIT_TEST(AllNamesValuesWithHeaderTest) {
        {
            auto allNames = GetEnumAllCppNames<EWithHeader>();
            UNIT_ASSERT_VALUES_EQUAL(allNames.size(), 3u);
            UNIT_ASSERT_VALUES_EQUAL(allNames.at(2), "HThree");
        }
        {
            UNIT_ASSERT_VALUES_EQUAL(GetEnumAllNames<EWithHeader>(), "'one', 'HTwo', 'HThree'");
        }
    }

    Y_UNIT_TEST(AllValuesTest) {
        const auto& allNames = GetEnumNames<EWithHeader>();
        const auto& allValues = GetEnumAllValues<EWithHeader>();
        UNIT_ASSERT_VALUES_EQUAL(allValues.size(), 3u);
        UNIT_ASSERT_VALUES_EQUAL(allValues[2], HThree);
        size_t size = 0;
        for (const EWithHeader value : GetEnumAllValues<EWithHeader>()) {
            size += 1;
            UNIT_ASSERT_VALUES_EQUAL(allNames.contains(value), true);
        }
        UNIT_ASSERT_VALUES_EQUAL(size, 3u);
    }

    Y_UNIT_TEST(EnumNamesTest) {
        const auto& names = GetEnumNames<EWithHeader>();
        UNIT_ASSERT_VALUES_EQUAL(names.size(), 3u);

        UNIT_ASSERT(names.contains(HOne));
        UNIT_ASSERT_VALUES_EQUAL(names.at(HOne), "one");

        UNIT_ASSERT(names.contains(HTwo));
        UNIT_ASSERT_VALUES_EQUAL(names.at(HTwo), "HTwo");

        UNIT_ASSERT(names.contains(HThree));
        UNIT_ASSERT_VALUES_EQUAL(names.at(HThree), "HThree");
    }

    Y_UNIT_TEST(ToStringBufHeaderTest) {
        UNIT_ASSERT_VALUES_EQUAL(ToStringBuf(HOne), "one"sv);
        UNIT_ASSERT_VALUES_EQUAL(ToStringBuf(HTwo), "HTwo"sv);
    }

    Y_UNIT_TEST(EnumSerializerDestructionPriority) {
        Singleton<TEnumSerializationInitializerHolder>()->Init();
    }
};
