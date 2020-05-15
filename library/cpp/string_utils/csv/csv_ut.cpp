#include "csv.h"
#include <library/cpp/unittest/registar.h>

namespace NCsvFormat {
    Y_UNIT_TEST_SUITE(CSVSplit) {
        Y_UNIT_TEST(CSVSplitSimple) {
            TString s = "1,2,3";
            TVector<TString> expected = {"1", "2", "3"};
            TVector<TString> actual = (TVector<TString>)CsvSplitter(s);
            UNIT_ASSERT_VALUES_EQUAL(expected, actual);
        }
        Y_UNIT_TEST(CSVSplitSimpleWithoutEscaping) {
            TString s = "1,2,3";
            TVector<TString> expected = {"1", "2", "3"};
            TVector<TString> actual = (TVector<TString>)CsvSplitter(s, ',', '\0');
            UNIT_ASSERT_VALUES_EQUAL(expected, actual);
        }
        Y_UNIT_TEST(CSVSplitDoubleQuote) {
            TString s = "Samsumg,1000$,\"19 \"\" display\"";
            TVector<TString> expected = {"Samsumg", "1000$", "19 \" display"};
            TVector<TString> actual = (TVector<TString>)CsvSplitter(s);
            UNIT_ASSERT_VALUES_EQUAL(expected, actual);
        }
        Y_UNIT_TEST(CSVSplitTripleQuote) {
            TString s = "\"\"\"something here\"\"\",\"and \"\"maybe\"\" even\",here,\"and \"\"here\"\"\"";
            TVector<TString> expected = {"\"something here\"", "and \"maybe\" even", "here", "and \"here\""};
            TVector<TString> actual = (TVector<TString>)CsvSplitter(s);
            UNIT_ASSERT_VALUES_EQUAL(expected, actual);
        }
        Y_UNIT_TEST(CSVSplitDanglingComma) {
            TString s = ",";
            TVector<TString> expected = {"", ""};
            TVector<TString> actual = (TVector<TString>)CsvSplitter(s);
            UNIT_ASSERT_VALUES_EQUAL(expected, actual);
        }
        Y_UNIT_TEST(CSVSplitDelimeterInsideEscapedString) {
            TString s = "Yandex,\"Moscow, Russia\",1997";
            TVector<TString> expected = {"Yandex", "Moscow, Russia", "1997"};
            TVector<TString> actual = (TVector<TString>)CsvSplitter(s);
            UNIT_ASSERT_VALUES_EQUAL(expected, actual);
        }
        Y_UNIT_TEST(CSVSplitCyrillic) {
            TString s = "NIN,\"Cleveland, Ohio\",\"9\"\" Nails\",\"Trent Reznor\",1988,Девятидюймовые Гвозди";
            TVector<TString> expected = {"NIN", "Cleveland, Ohio", "9\" Nails", "Trent Reznor", "1988", "Девятидюймовые Гвозди"};
            TVector<TString> actual = (TVector<TString>)CsvSplitter(s);
            UNIT_ASSERT_VALUES_EQUAL(expected, actual);
        }
        Y_UNIT_TEST(CSVSplitCustomDelimeter) {
            TString s = "1\t\'2\t3,4\'\t5";
            TVector<TString> expected = {"1", "2\t3,4", "5"};
            TVector<TString> actual = (TVector<TString>)CsvSplitter(s, '\t', '\'');
            UNIT_ASSERT_VALUES_EQUAL(expected, actual);
        }
    }
}
