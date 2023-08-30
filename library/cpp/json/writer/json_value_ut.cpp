#include "json_value.h"

#include <library/cpp/testing/unittest/registar.h>

#include <util/stream/input.h>

using namespace NJson;

Y_UNIT_TEST_SUITE(TJsonValueTest) {
    Y_UNIT_TEST(Equal) {
         UNIT_ASSERT(1 == TJsonValue(1));
         UNIT_ASSERT(TJsonValue(1) == 1);
         UNIT_ASSERT(2 != TJsonValue(1));
         UNIT_ASSERT(TJsonValue(1) != 2);
    }

    Y_UNIT_TEST(UndefTest) {
        TJsonValue undef;
        TJsonValue null(JSON_NULL);
        TJsonValue _false(false);
        TJsonValue zeroInt(0);
        TJsonValue zeroDouble(0.0);
        TJsonValue emptyStr("");
        TJsonValue emptyArray(JSON_ARRAY);
        TJsonValue emptyMap(JSON_MAP);

        UNIT_ASSERT(!undef.IsDefined());
        UNIT_ASSERT(!null.IsDefined()); // json NULL is undefined too!
        UNIT_ASSERT(_false.IsDefined());
        UNIT_ASSERT(zeroInt.IsDefined());
        UNIT_ASSERT(zeroDouble.IsDefined());
        UNIT_ASSERT(emptyStr.IsDefined());
        UNIT_ASSERT(emptyArray.IsDefined());
        UNIT_ASSERT(emptyMap.IsDefined());

        UNIT_ASSERT(undef == TJsonValue());
        UNIT_ASSERT(undef != null);
        UNIT_ASSERT(undef != _false);
        UNIT_ASSERT(undef != zeroInt);
        UNIT_ASSERT(undef != zeroDouble);
        UNIT_ASSERT(undef != emptyStr);
        UNIT_ASSERT(undef != emptyArray);
        UNIT_ASSERT(undef != emptyMap);
    }

    Y_UNIT_TEST(DefaultCompareTest) {
        {
            TJsonValue lhs;
            TJsonValue rhs;
            UNIT_ASSERT(lhs == rhs);
            UNIT_ASSERT(rhs == lhs);
        }

        {
            TJsonValue lhs;
            TJsonValue rhs(JSON_NULL);
            UNIT_ASSERT(lhs != rhs);
            UNIT_ASSERT(rhs != lhs);
        }
    }

    Y_UNIT_TEST(NullCompareTest) {
        TJsonValue lhs(JSON_NULL);
        TJsonValue rhs(JSON_NULL);
        UNIT_ASSERT(lhs == rhs);
        UNIT_ASSERT(rhs == lhs);
    }

    Y_UNIT_TEST(StringCompareTest) {
        {
            TJsonValue lhs(JSON_STRING);
            TJsonValue rhs(JSON_STRING);
            UNIT_ASSERT(lhs == rhs);
            UNIT_ASSERT(rhs == lhs);
        }

        {
            TJsonValue lhs("");
            TJsonValue rhs("");
            UNIT_ASSERT(lhs == rhs);
            UNIT_ASSERT(rhs == lhs);
        }

        {
            TJsonValue lhs("abc");
            TJsonValue rhs("abc");
            UNIT_ASSERT(lhs == rhs);
            UNIT_ASSERT(rhs == lhs);
        }

        {
            TJsonValue lhs("1");
            TJsonValue rhs(1);
            UNIT_ASSERT(lhs != rhs);
            UNIT_ASSERT(rhs != lhs);
        }
    }

    Y_UNIT_TEST(ArrayCompareTest) {
        {
            TJsonValue lhs(JSON_ARRAY);
            TJsonValue rhs(JSON_ARRAY);
            UNIT_ASSERT(lhs == rhs);
            UNIT_ASSERT(rhs == lhs);
        }

        {
            TJsonValue lhs;
            TJsonValue rhs;

            lhs.AppendValue(TJsonValue());

            UNIT_ASSERT(lhs != rhs);
            UNIT_ASSERT(rhs != lhs);
        }

        {
            TJsonValue lhs;
            TJsonValue rhs;

            lhs.AppendValue(1);
            lhs.AppendValue("2");
            lhs.AppendValue(3.0);
            lhs.AppendValue(TJsonValue());
            lhs.AppendValue(TJsonValue(JSON_NULL));

            rhs.AppendValue(1);
            rhs.AppendValue("2");
            rhs.AppendValue(3.0);
            rhs.AppendValue(TJsonValue());
            rhs.AppendValue(TJsonValue(JSON_NULL));

            UNIT_ASSERT(lhs == rhs);
            UNIT_ASSERT(rhs == lhs);
        }

        {
            TJsonValue lhs;
            TJsonValue rhs;

            lhs.AppendValue(1);
            rhs.AppendValue("1");

            UNIT_ASSERT(lhs != rhs);
            UNIT_ASSERT(rhs != lhs);
        }
    }

    Y_UNIT_TEST(CompareTest) {
        {
            TJsonValue lhs;
            lhs.InsertValue("null value", TJsonValue(JSON_NULL));
            lhs.InsertValue("int key", TJsonValue(10));
            lhs.InsertValue("double key", TJsonValue(11.11));
            lhs.InsertValue("string key", TJsonValue("string"));

            TJsonValue array;
            array.AppendValue(1);
            array.AppendValue(2);
            array.AppendValue(3);
            array.AppendValue("string");
            lhs.InsertValue("array", array);

            lhs.InsertValue("bool key", TJsonValue(true));

            TJsonValue rhs;
            rhs = lhs;

            UNIT_ASSERT(lhs == rhs);
            UNIT_ASSERT(rhs == lhs);
        }

        {
            // Insert keys in different orders
            const int NUM_KEYS = 1000;

            TJsonValue lhs;
            for (int i = 0; i < NUM_KEYS; ++i)
                lhs.InsertValue(ToString(i), i);

            TJsonValue rhs;
            for (int i = 0; i < NUM_KEYS; i += 2)
                rhs.InsertValue(ToString(i), i);
            for (int i = 1; i < NUM_KEYS; i += 2)
                rhs.InsertValue(ToString(i), i);

            UNIT_ASSERT(lhs == rhs);
            UNIT_ASSERT(rhs == lhs);
        }

        {
            TJsonValue lhs;
            lhs.InsertValue("null value", TJsonValue(JSON_NULL));
            lhs.InsertValue("int key", TJsonValue(10));
            lhs.InsertValue("double key", TJsonValue(11.11));
            lhs.InsertValue("string key", TJsonValue("string"));

            TJsonValue array;
            array.AppendValue(1);
            array.AppendValue(2);
            array.AppendValue(3);
            array.AppendValue("string");
            lhs.InsertValue("array", array);

            lhs.InsertValue("bool key", TJsonValue(true));

            TJsonValue rhs;
            rhs.InsertValue("null value", TJsonValue(JSON_NULL));
            rhs.InsertValue("int key", TJsonValue(10));
            rhs.InsertValue("double key", TJsonValue(11.11));
            rhs.InsertValue("string key", TJsonValue("string"));
            rhs.InsertValue("bool key", TJsonValue(true));

            UNIT_ASSERT(lhs != rhs);
            UNIT_ASSERT(rhs != lhs);
        }
    }

    Y_UNIT_TEST(SwapTest) {
        {
            TJsonValue lhs;
            lhs.InsertValue("a", "b");
            TJsonValue lhsCopy = lhs;

            TJsonValue rhs(JSON_NULL);
            TJsonValue rhsCopy = rhs;

            UNIT_ASSERT(lhs == lhsCopy);
            UNIT_ASSERT(rhs == rhsCopy);

            lhs.Swap(rhs);

            UNIT_ASSERT(rhs == lhsCopy);
            UNIT_ASSERT(lhs == rhsCopy);

            lhs.Swap(rhs);

            UNIT_ASSERT(lhs == lhsCopy);
            UNIT_ASSERT(rhs == rhsCopy);
        }
    }

    Y_UNIT_TEST(GetValueByPathTest) {
        {
            TJsonValue lhs;
            TJsonValue first;
            TJsonValue second;
            TJsonValue last;
            first.InsertValue("e", "f");
            second.InsertValue("c", first);
            last.InsertValue("a", second);
            lhs.InsertValue("l", last);

            TJsonValue result;
            UNIT_ASSERT(lhs.GetValueByPath("l/a/c/e", result, '/'));
            UNIT_ASSERT(result.GetStringRobust() == "f");
            UNIT_ASSERT(!lhs.GetValueByPath("l/a/c/se", result, '/'));
            UNIT_ASSERT(lhs.GetValueByPath("l/a/c", result, '/'));
            UNIT_ASSERT(result.GetStringRobust() == "{\"e\":\"f\"}");

            // faster TStringBuf version
            UNIT_ASSERT_EQUAL(*lhs.GetValueByPath("l", '/'), last);
            UNIT_ASSERT_EQUAL(*lhs.GetValueByPath("l/a", '/'), second);
            UNIT_ASSERT_EQUAL(*lhs.GetValueByPath("l/a/c", '/'), first);
            UNIT_ASSERT_EQUAL(*lhs.GetValueByPath("l.a.c.e", '.'), "f");
            UNIT_ASSERT_EQUAL(lhs.GetValueByPath("l/a/c/e/x", '/'), NULL);
            UNIT_ASSERT_EQUAL(lhs.GetValueByPath("a/c/e/x", '/'), NULL);
            UNIT_ASSERT_EQUAL(lhs.GetValueByPath("nokey", '/'), NULL);
            UNIT_ASSERT_EQUAL(*lhs.GetValueByPath("", '/'), lhs); // itself

            TJsonValue array;
            TJsonValue third;
            array[0] = first;
            array[1] = second;
            third["t"] = array;

            UNIT_ASSERT(array.GetValueByPath("[0].e", result));
            UNIT_ASSERT(result.GetStringRobust() == "f");
            UNIT_ASSERT(third.GetValueByPath("t.[0].e", result));
            UNIT_ASSERT(result.GetStringRobust() == "f");
            UNIT_ASSERT(third.GetValueByPath("t.[1].c.e", result));
            UNIT_ASSERT(result.GetStringRobust() == "f");
            UNIT_ASSERT(!third.GetValueByPath("t.[2]", result));

            UNIT_ASSERT(third.SetValueByPath("t.[2]", "g"));
            UNIT_ASSERT(third.GetValueByPath("t.[2]", result));
            UNIT_ASSERT(result.GetStringRobust() == "g");

            UNIT_ASSERT(lhs.SetValueByPath("l/a/c/se", "h", '/'));
            UNIT_ASSERT(lhs.GetValueByPath("l/a/c/se", result, '/'));
            UNIT_ASSERT(result.GetStringRobust() == "h");
        }
    }

    Y_UNIT_TEST(GetValueByPathConstTest) {
        TJsonValue lhs;
        TJsonValue first;
        TJsonValue second;
        TJsonValue last;
        first.InsertValue("e", "f");
        second.InsertValue("c", first);
        last.InsertValue("a", second);
        lhs.InsertValue("l", last);

        {
            const TJsonValue* result = lhs.GetValueByPath("l", '/');
            UNIT_ASSERT_EQUAL(*result, last);
        }
        {
            const TJsonValue* result = lhs.GetValueByPath("l/a", '/');
            UNIT_ASSERT_EQUAL(*result, second);
        }
        {
            const TJsonValue* result = lhs.GetValueByPath("l/a/c", '/');
            UNIT_ASSERT_EQUAL(*result, first);
        }
        {
            const TJsonValue* result = lhs.GetValueByPath("l.a.c.e", '.');
            UNIT_ASSERT_EQUAL(*result, "f");
        }
        {
            const TJsonValue* result = lhs.GetValueByPath("l/a/c/e/x", '/');
            UNIT_ASSERT_EQUAL(result, nullptr);
        }
        {
            const TJsonValue* result = lhs.GetValueByPath("a/c/e/x", '/');
            UNIT_ASSERT_EQUAL(result, nullptr);
        }
        {
            const TJsonValue* result = lhs.GetValueByPath("nokey", '/');
            UNIT_ASSERT_EQUAL(result, nullptr);
        }
        {
            const TJsonValue* result = lhs.GetValueByPath("", '/');
            UNIT_ASSERT_EQUAL(*result, lhs); // itself
        }

        TJsonValue array;
        TJsonValue third;
        array[0] = first;
        array[1] = second;
        third["t"] = array;

        UNIT_ASSERT(array.GetValueByPath("[0].e", '.')->GetStringRobust() == "f");
        UNIT_ASSERT(third.GetValueByPath("t.[0].e", '.')->GetStringRobust() == "f");
        UNIT_ASSERT(third.GetValueByPath("t.[1].c.e", '.')->GetStringRobust() == "f");
    }

    Y_UNIT_TEST(EraseValueFromArray) {
        {
            TJsonValue vec;
            vec.AppendValue(TJsonValue(0));
            vec.AppendValue(TJsonValue(1));
            vec.AppendValue(TJsonValue("2"));
            vec.AppendValue(TJsonValue("3.14"));

            TJsonValue vec1;
            vec1.AppendValue(TJsonValue(0));
            vec1.AppendValue(TJsonValue("2"));
            vec1.AppendValue(TJsonValue("3.14"));

            TJsonValue vec2;
            vec2.AppendValue(TJsonValue(0));
            vec2.AppendValue(TJsonValue("2"));

            TJsonValue vec3;
            vec3.AppendValue(TJsonValue("2"));

            TJsonValue vec4(JSON_ARRAY);

            UNIT_ASSERT(vec.IsArray());
            UNIT_ASSERT(vec.GetArray().size() == 4);
            vec.EraseValue(1);
            UNIT_ASSERT(vec.GetArray().size() == 3);
            UNIT_ASSERT(vec == vec1);
            vec.EraseValue(2);
            UNIT_ASSERT(vec.GetArray().size() == 2);
            UNIT_ASSERT(vec == vec2);
            vec.EraseValue(0);
            UNIT_ASSERT(vec.GetArray().size() == 1);
            UNIT_ASSERT(vec == vec3);
            vec.EraseValue(0);
            UNIT_ASSERT(vec.GetArray().size() == 0);
            UNIT_ASSERT(vec == vec4);
        }
    }

    Y_UNIT_TEST(NonConstMethodsTest) {
        {
            TJsonValue src;
            TJsonValue value1;
            value1.AppendValue(1);
            value1.AppendValue(2);
            src.InsertValue("key", value1);
            src.InsertValue("key1", "HI!");

            TJsonValue dst;
            TJsonValue value2;
            value2.AppendValue(1);
            value2.AppendValue(2);
            value2.AppendValue(3);
            dst.InsertValue("key", value2);

            src.GetValueByPath("key", '.')->AppendValue(3);
            src.EraseValue("key1");
            UNIT_ASSERT(src == dst);

            dst.GetValueByPath("key", '.')->EraseValue(0);
            UNIT_ASSERT(src != dst);
            src.GetValueByPath("key", '.')->EraseValue(0);
            UNIT_ASSERT(src == dst);
        }

        {
            TJsonValue src;
            TJsonValue value1;
            TJsonValue arr1;
            value1.InsertValue("key", "value");
            arr1.AppendValue(value1);
            arr1.AppendValue(value1);
            arr1.AppendValue(value1);
            src.InsertValue("arr", arr1);

            TJsonValue dst;
            TJsonValue value2;
            TJsonValue arr2;
            value2.InsertValue("key", "value");
            value2.InsertValue("yek", "eulav");
            arr2.AppendValue(value2);
            arr2.AppendValue(value2);
            arr2.AppendValue(value2);
            arr2.AppendValue(value2);
            dst.InsertValue("arr", arr2);

            src["arr"].AppendValue(value1);
            for (auto& node : src["arr"].GetArraySafe()) {
                node.InsertValue("yek", "eulav");
            }
            UNIT_ASSERT(src == dst);
        }

        {
            TJsonValue src;
            TJsonValue value1;
            TJsonValue arr1;
            value1.InsertValue("key", "value");
            arr1.AppendValue(value1);
            arr1.AppendValue(value1);
            arr1.AppendValue(value1);
            src.InsertValue("arr", arr1);

            TJsonValue dst;
            TJsonValue value2;
            TJsonValue arr2;
            value2.InsertValue("key", "value");
            value2.InsertValue("yek", "eulav");
            arr2.AppendValue(value2);
            arr2.AppendValue(value2);
            arr2.AppendValue(value2);
            arr2.AppendValue(value2);
            dst.InsertValue("arr", arr2);

            src["arr"].AppendValue(value1);
            for (auto& node : src.GetValueByPath("arr", '.')->GetArraySafe()) {
                node.InsertValue("yek", "eulav");
            }
            UNIT_ASSERT(src == dst);
        }

        {
            TJsonValue json;
            json.InsertValue("key", "value");
            try {
                json.GetArraySafe();
                UNIT_ASSERT(false);
            } catch (const TJsonException&) {
            }

            const TJsonValue constJson(json);
            try {
                constJson.GetArray();
            } catch (...) {
                UNIT_ASSERT(false);
            }
        }

        {
            // Check non-const GetArraySafe()
            TJsonValue json{JSON_ARRAY};
            json.GetArraySafe().push_back(TJsonValue{"foo"});

            TJsonValue expectedJson;
            expectedJson.AppendValue(TJsonValue{"foo"});
            UNIT_ASSERT(json == expectedJson);

            TJsonValue::TArray jsonArray = std::move(json.GetArraySafe());
            TJsonValue::TArray expectedArray = {TJsonValue{"foo"}};
            UNIT_ASSERT(jsonArray == expectedArray);
        }

        {
            // Check non-const GetMap()
            TJsonValue json{JSON_MAP};
            json.GetMapSafe()["foo"] = "bar";

            TJsonValue expectedJson;
            expectedJson.InsertValue("foo", "bar");
            UNIT_ASSERT(json == expectedJson);

            TJsonValue::TMapType jsonMap = std::move(json.GetMapSafe());
            TJsonValue::TMapType expectedMap = {{"foo", TJsonValue{"bar"}}};
            UNIT_ASSERT(jsonMap == expectedMap);
        }
    }

    Y_UNIT_TEST(NonexistentFieldAccessTest) {
        {
            TJsonValue json;
            json.InsertValue("some", "key");

            UNIT_ASSERT(!json["some"]["weird"]["access"]["sequence"].Has("value"));
            UNIT_ASSERT(!json["some"]["weird"]["access"]["sequence"].IsDefined());

            UNIT_ASSERT(json["some"].GetType() == JSON_MAP);
        }
    }

    Y_UNIT_TEST(DefaultValuesTest) {
        {
            TJsonValue json;
            json.InsertValue("some", "key");
            json.InsertValue("existing", 1.2);

            UNIT_ASSERT_VALUES_EQUAL(json["existing"].GetDoubleSafe(), 1.2);
            UNIT_ASSERT_VALUES_EQUAL(json["existing"].GetDoubleSafe(15), 1.2);

            UNIT_ASSERT_EXCEPTION(json["some"].GetUIntegerSafe(), yexception);
            UNIT_ASSERT_EXCEPTION(json["some"].GetUIntegerSafe(12), yexception);

            UNIT_ASSERT_EXCEPTION(json["nonexistent"].GetUIntegerSafe(), yexception);
            UNIT_ASSERT_VALUES_EQUAL(json["nonexistent"].GetUIntegerSafe(12), 12);
            UNIT_ASSERT_VALUES_EQUAL(json["nonexistent"]["more_nonexistent"].GetUIntegerSafe(12), 12);

            json.InsertValue("map", TJsonValue(JSON_MAP));

            UNIT_ASSERT_VALUES_EQUAL(json["map"]["nonexistent"].GetUIntegerSafe(12), 12);
        }
    }

    Y_UNIT_TEST(GetArrayPointerInArrayTest) {
        TJsonValue outer;
        {
            TJsonValue json;
            json.AppendValue(1);
            json.AppendValue(2);
            json.AppendValue(3);

            outer.AppendValue(json);
        }
        const TJsonValue::TArray* array = nullptr;
        GetArrayPointer(outer, 0, &array);
        UNIT_ASSERT_VALUES_EQUAL((*array)[1], 2);
    }

    Y_UNIT_TEST(GetArrayPointerInMapTest) {
        TJsonValue outer;
        {
            TJsonValue json;
            json.AppendValue(1);
            json.AppendValue(2);
            json.AppendValue(3);

            outer.InsertValue("x", json);
        }
        const TJsonValue::TArray* array = nullptr;
        GetArrayPointer(outer, "x", &array);
        UNIT_ASSERT_VALUES_EQUAL((*array)[1], 2);
    }

    Y_UNIT_TEST(GetMapPointerInArrayTest) {
        TJsonValue outer;
        {
            TJsonValue json;
            json.InsertValue("a", 1);
            json.InsertValue("b", 2);
            json.InsertValue("c", 3);

            outer.AppendValue(json);
        }
        const TJsonValue::TMapType* map = nullptr;
        GetMapPointer(outer, 0, &map);
        UNIT_ASSERT_VALUES_EQUAL((*map).at("b"), 2);
    }

    Y_UNIT_TEST(GetMapPointerInMapTest) {
        TJsonValue outer;
        {
            TJsonValue json;
            json.InsertValue("a", 1);
            json.InsertValue("b", 2);
            json.InsertValue("c", 3);

            outer.InsertValue("x", json);
        }
        const TJsonValue::TMapType* map = nullptr;
        GetMapPointer(outer, "x", &map);
        UNIT_ASSERT_VALUES_EQUAL((*map).at("b"), 2);
    }

    Y_UNIT_TEST(GetIntegerRobustBignumStringTest) {
        TString value = "1626862681464633683";
        TJsonValue json(value);
        UNIT_ASSERT_VALUES_EQUAL(json.GetUIntegerRobust(), FromString<ui64>(value));
        UNIT_ASSERT_VALUES_EQUAL(json.GetIntegerRobust(), FromString<i64>(value));
    }

    Y_UNIT_TEST(MoveSubpartToSelf) {
        TJsonValue json;
        json[0] = "testing 0";
        json[1] = "testing 1";
        json[2] = "testing 2";
        json = std::move(json[1]);
        UNIT_ASSERT_VALUES_EQUAL(json.GetString(), "testing 1");

        const char* longTestString =
            "Testing TJsonValue& operator=(TJsonValue&&) subpart self moving "
            "after TJsonValue was constrcuted from TString&&.";

        json["hello"] = TString{longTestString};
        json = std::move(json["hello"]);
        UNIT_ASSERT_VALUES_EQUAL(json.GetString(), longTestString);
    }

    Y_UNIT_TEST(TJsonArrayMapConstructor) {
        TJsonMap emptyMap;
        UNIT_ASSERT_VALUES_EQUAL(emptyMap.GetType(), JSON_MAP);
        UNIT_ASSERT_VALUES_EQUAL(emptyMap.GetMapSafe().size(), 0);

        TJsonArray emptyArray;
        UNIT_ASSERT_VALUES_EQUAL(emptyArray.GetType(), JSON_ARRAY);
        UNIT_ASSERT_VALUES_EQUAL(emptyArray.GetArraySafe().size(), 0);

        TJsonMap filled = {
            {"1", 1},
            {"2", "2"},
            {"3", TJsonArray{3}},
            {"4", TJsonMap{{"5", 5}}},
        };
        UNIT_ASSERT_VALUES_EQUAL(filled.GetType(), JSON_MAP);
        UNIT_ASSERT_VALUES_EQUAL(filled["1"], TJsonValue{1});
        UNIT_ASSERT_VALUES_EQUAL(filled["2"], TJsonValue{"2"});
        UNIT_ASSERT_VALUES_EQUAL(filled["3"].GetArraySafe().size(), 1);
        UNIT_ASSERT_VALUES_EQUAL(filled["3"][0], TJsonValue{3});
        UNIT_ASSERT_VALUES_EQUAL(filled["4"].GetMapSafe().size(), 1);
        UNIT_ASSERT_VALUES_EQUAL(filled["4"]["5"], TJsonValue{5});
    }
} // TJsonValueTest
