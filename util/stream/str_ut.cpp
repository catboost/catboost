#include "str.h"

#include <library/cpp/testing/unittest/registar.h>
#include <util/generic/typetraits.h>

template <typename T>
const T ReturnConstTemp();

Y_UNIT_TEST_SUITE(TStringInputOutputTest) {
    Y_UNIT_TEST(Lvalue) {
        TString str = "Hello, World!";
        TStringInput input(str);

        TString result = input.ReadAll();

        UNIT_ASSERT_VALUES_EQUAL(result, str);
    }

    Y_UNIT_TEST(ConstRef) {
        TString str = "Hello, World!";
        const TString& r = str;
        TStringInput input(r);

        TString result = input.ReadAll();

        UNIT_ASSERT_VALUES_EQUAL(result, str);
    }

    Y_UNIT_TEST(NonConstRef) {
        TString str = "Hello, World!";
        TString& r = str;
        TStringInput input(r);

        TString result = input.ReadAll();

        UNIT_ASSERT_VALUES_EQUAL(result, str);
    }

    Y_UNIT_TEST(Transfer) {
        TString inputString = "some_string";
        TStringInput input(inputString);

        TString outputString;
        TStringOutput output(outputString);

        TransferData(&input, &output);

        UNIT_ASSERT_VALUES_EQUAL(inputString, outputString);
    }

    Y_UNIT_TEST(SkipReadAll) {
        TString string0 = "All animals are equal, but some animals are more equal than others.";

        TString string1;
        for (size_t i = 1; i <= string0.size(); i++) {
            string1 += string0.substr(0, i);
        }

        TStringInput input0(string1);

        size_t left = 5;
        while (left > 0) {
            left -= input0.Skip(left);
        }

        TString string2 = input0.ReadAll();

        UNIT_ASSERT_VALUES_EQUAL(string2, string1.substr(5));
    }

    Y_UNIT_TEST(OperatorBool) {
        TStringStream str;
        UNIT_ASSERT(!str);
        str << "data";
        UNIT_ASSERT(str);
        str.Clear();
        UNIT_ASSERT(!str);
    }

    Y_UNIT_TEST(TestReadTo) {
        TString s("0123456789abc");
        TString t;

        TStringInput in0(s);
        UNIT_ASSERT_VALUES_EQUAL(in0.ReadTo(t, '7'), 8);
        UNIT_ASSERT_VALUES_EQUAL(t, "0123456");
        UNIT_ASSERT_VALUES_EQUAL(in0.ReadTo(t, 'z'), 5);
        UNIT_ASSERT_VALUES_EQUAL(t, "89abc");
    }

    Y_UNIT_TEST(WriteViaNextAndUndo) {
        TString str1;
        TStringOutput output(str1);
        TString str2;

        for (size_t i = 0; i < 10000; ++i) {
            str2.push_back('a' + (i % 20));
        }

        size_t written = 0;
        void* ptr = nullptr;
        while (written < str2.size()) {
            size_t bufferSize = output.Next(&ptr);
            UNIT_ASSERT(ptr && bufferSize > 0);
            size_t toWrite = Min(bufferSize, str2.size() - written);
            memcpy(ptr, str2.begin() + written, toWrite);
            written += toWrite;
            if (toWrite < bufferSize) {
                output.Undo(bufferSize - toWrite);
            }
        }

        UNIT_ASSERT_STRINGS_EQUAL(str1, str2);
    }

    Y_UNIT_TEST(Write) {
        TString str;
        TStringOutput output(str);
        output << "1"
               << "22"
               << "333"
               << "4444"
               << "55555";

        UNIT_ASSERT_STRINGS_EQUAL(str, "1"
                                       "22"
                                       "333"
                                       "4444"
                                       "55555");
    }

    Y_UNIT_TEST(WriteChars) {
        TString str;
        TStringOutput output(str);
        output << '1' << '2' << '3' << '4' << '5' << '6' << '7' << '8' << '9' << '0';

        UNIT_ASSERT_STRINGS_EQUAL(str, "1234567890");
    }

    Y_UNIT_TEST(MoveConstructor) {
        TString str;
        TStringOutput output1(str);
        output1 << "foo";

        TStringOutput output2 = std::move(output1);
        output2 << "bar";
        UNIT_ASSERT_STRINGS_EQUAL(str, "foobar");

        // Check old stream is in a valid state
        output1 << "baz";
    }
}
