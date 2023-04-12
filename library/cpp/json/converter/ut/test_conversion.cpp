#include "library/cpp/json/converter/converter.h"
#include <library/cpp/json/json_reader.h>
#include <library/cpp/json/json_writer.h>
#include "library/cpp/json/writer/json_value.h"
#include <library/cpp/testing/unittest/registar.h>

#include <util/generic/array_ref.h>
#include <util/generic/deque.h>
#include <util/generic/hash.h>
#include <util/generic/list.h>
#include <util/generic/map.h>
#include <util/generic/maybe.h>


namespace NJson {
    void AssertJsonsEqual(const TJsonValue& actualJson, const TJsonValue& expectedJson) {
        const auto actualString = WriteJson(actualJson, /*formatOutput*/ true, /*sortkeys*/ true, /*validateUtf8*/ true);
        const auto expectedString = WriteJson(expectedJson, /*formatOutput*/ true, /*sortkeys*/ true, /*validateUtf8*/ true);
        UNIT_ASSERT_NO_DIFF(actualString, expectedString);
    }

    void AssertJsonsEqual(const TJsonValue& actualJson, TStringBuf expectedString) {
        const auto expectedJson = ReadJsonFastTree(expectedString);
        AssertJsonsEqual(actualJson, expectedJson);
    }

    template<typename T>
    struct TConverterTester {
        using TValues = THashMap<TStringBuf, T>;

        static void TestEncoding(const TValues& values) {
            for (const auto& [serializedValue, value] : values) {
                const auto encodedValue = TConverter<T>::Encode(value);
                AssertJsonsEqual(encodedValue, serializedValue);
            }
        }

        static void TestDecoding(const TValues& values) {
            for (const auto& [serializedValue, value] : values) {
                const auto decodedValue = TConverter<T>::Decode(ReadJsonFastTree(serializedValue));
                UNIT_ASSERT_EQUAL(decodedValue, value);
            }
        }

        static void TestDecodingException(TStringBuf serializedValue) {
            try {
                TConverter<T>::Decode(ReadJsonFastTree(serializedValue));
                UNIT_ASSERT(false);
            } catch (...) {
            }
        }

        static void Test(const TValues& values) {
            TestEncoding(values);
            TestDecoding(values);

            for (const auto& [serializedValue, value] : values) {
                const auto encodedValue = TConverter<T>::Encode(value);
                const auto decodedValue = TConverter<T>::Decode(encodedValue);
                UNIT_ASSERT_EQUAL(value, decodedValue);
            }
        }
    };

    template<typename T>
    requires std::is_floating_point_v<T>
    struct TConverterTester<T> {
        using TValues = THashMap<TStringBuf, T>;

        static void TestDecoding(const TValues& values) {
            for (const auto& [serializedValue, value] : values) {
                {
                    const auto decodedValue = TConverter<T>::Decode(ReadJsonFastTree(serializedValue));
                    UNIT_ASSERT_DOUBLES_EQUAL(decodedValue, value, 0.000001);
                }
            }
        }

        static void Test(const TValues& values) {
            TestDecoding(values);

            for (const auto& [serializedValue, value] : values) {
                {
                    const auto encodedValue = TConverter<T>::Encode(value);
                    const auto decodedValue = TConverter<T>::Decode(encodedValue);
                    UNIT_ASSERT_DOUBLES_EQUAL(decodedValue, value, 0.000001);
                }
            }
        }
    };

    Y_UNIT_TEST_SUITE(ConversionTests) {
        Y_UNIT_TEST(PrimitivesTest) {
            TConverterTester<bool>::Test({{"true", true}, {"false", false}});

            TConverterTester<ui8>::Test({{"0", 0}, {"255", 255}});
            TConverterTester<i8>::Test({{"-128", -128}, {"127", 127}});
            TConverterTester<ui16>::Test({{"0", 0}, {"65535", 65535}});
            TConverterTester<i16>::Test({{"-32768", -32768}, {"32767", 32767}});
            TConverterTester<ui32>::Test({{"0", 0}, {"4294967295", 4294967295}});
            TConverterTester<i32>::Test({{"-2147483648", -2147483648}, {"2147483647", 2147483647}});
            TConverterTester<ui64>::Test({{"0", 0}, {"18446744073709551615", 18446744073709551615u}});
            TConverterTester<i64>::Test({
                {"-9223372036854775808", -9223372036854775808u},
                {"9223372036854775807", 9223372036854775807},
            });

            TConverterTester<i8>::TestDecodingException("128");
            TConverterTester<i8>::TestDecodingException("-129");
            TConverterTester<ui8>::TestDecodingException("256");
            TConverterTester<i16>::TestDecodingException("32768");
            TConverterTester<i16>::TestDecodingException("-32769");
            TConverterTester<ui16>::TestDecodingException("65536");
            TConverterTester<i32>::TestDecodingException("-2147483649");
            TConverterTester<i32>::TestDecodingException("2147483649");
            TConverterTester<ui32>::TestDecodingException("4294967296");

            TConverterTester<unsigned char>::Test({{"0", 0}, {"255", 255}});
            TConverterTester<signed char>::Test({{"-128", -128}, {"127", 127}});
            TConverterTester<unsigned short int>::Test({{"0", 0}, {"65535", 65535}});
            TConverterTester<signed short int>::Test({{"-32768", -32768}, {"32767", 32767}});
            TConverterTester<unsigned int>::Test({{"0", 0}, {"65535", 65535}});
            TConverterTester<signed int>::Test({{"-32768", -32768}, {"32767", 32767}});
            TConverterTester<unsigned long int>::Test({{"0", 0}, {"4294967295", 4294967295}});
            TConverterTester<signed long int>::Test({{"-2147483648", -2147483648}, {"2147483647", 2147483647}});
            TConverterTester<unsigned long long int>::Test({{"0", 0}, {"18446744073709551615", 18446744073709551615u}});
            TConverterTester<signed long long int>::Test({
                {"-9223372036854775808", -9223372036854775808u},
                {"9223372036854775807", 9223372036854775807},
            });

            TConverterTester<size_t>::Test({{"0", 0}, {"65535", 65535}});

            TConverterTester<float>::Test({{"-1.12312", -1.12312}, {"3434.25674", 3434.25674}});
            TConverterTester<double>::Test({{"-1.12312E+42", -1.12312E+42}, {"3.25E+120", 3.25E+120}});
        }

        Y_UNIT_TEST(StringsTest) {
            TConverterTester<TStringBuf>::TestEncoding({
                {R"("Let's permit using of Rust in Arcadia")", "Let's permit using of Rust in Arcadia"},
            });
            TConverterTester<TString>::Test({
                {
                    R"("Всякое непрерывное отображение замкнутого n-мерного шара в себя обладает неподвижной точкой")",
                    "Всякое непрерывное отображение замкнутого n-мерного шара в себя обладает неподвижной точкой",
                },
            });
        }

        Y_UNIT_TEST(MaybeTest) {
            TConverterTester<TMaybe<bool>>::Test({
                {"true", TMaybe<bool>(true)},
                {"null", Nothing()},
                {"false", TMaybe<bool>(false)},
            });
        }

        Y_UNIT_TEST(ArraysTest) {
            TConverterTester<TVector<bool>>::Test({{"[true, true, false]", {true, true, false}}});
            TConverterTester<TList<TString>>::Test({{R"(["a", "b"])", {"a", "b"}}});
            TConverterTester<TDeque<bool>>::Test({{"[false, true, false]", {false, true, false}}});
        }

        Y_UNIT_TEST(MapsTest) {
            TConverterTester<THashMap<TStringBuf, bool>>::TestEncoding({
                {R"({"a": true, "b": false})", {{"a", true}, {"b", false}}},
            });
            TConverterTester<THashMap<TString, bool>>::Test({
                {R"({"a": true, "b": false})", {{"a", true}, {"b", false}}},
            });
            TConverterTester<TMap<TStringBuf, TStringBuf>>::TestEncoding({
                {R"({"a": "A", "b": "B"})", {{"a", "A"}, {"b", "B"}}},
            });
            TConverterTester<TMap<TString, TString>>::Test({
                {R"({"a": "A", "b": "B"})", {{"a", "A"}, {"b", "B"}}},
            });
        }

        Y_UNIT_TEST(NestedTest) {
            TConverterTester<TVector<THashMap<TString, TVector<ui64>>>>::Test({
                {
                    R"([
                        {
                            "Three": [0, 1, 2],
                            "Five": [0, 1, 2, 3, 4]
                        },
                        {
                            "Four": [0, 1, 2, 3],
                            "Six": [0, 1, 2, 3, 4, 5]
                        },
                    ])",
                    {
                        {
                            {"Three", {0, 1, 2}},
                            {"Five", {0, 1, 2, 3, 4}},
                        },
                        {
                            {"Four", {0, 1, 2, 3}},
                            {"Six", {0, 1, 2, 3, 4, 5}},
                        },
                    },
                },
            });
        }
    }
}
