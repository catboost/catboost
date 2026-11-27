#include <contrib/libs/rapidjson/include/rapidjson/document.h>

#include <library/cpp/testing/unittest/registar.h>

#include <util/generic/string.h>
#include <util/generic/vector.h>

#include <cstdlib>

Y_UNIT_TEST_SUITE(TRapidJsonStrtodTest) {
    Y_UNIT_TEST(FullPrecisionMatchesStrtod) {
        const TString halfwayAfterOne = "1.00000000000000011102230246251565404236316680908203125";
        const TVector<TString> numbers = {
            "1794478992.2227573",
            "1.8e308",
            "9e308",
            "0." + TString(329, '0') + "1",
            "0." + TString(400, '0') + "1",
            "0.184467440737095516159",
            "2.2250738585072011e-308",
            "2.2250738585072012e-308",
            "4.9406564584124654e-324",
            "2.4703282292062328e-324",
            "2.47032822920623272088e-324",
            "1.7976931348623157e308",
            "1.797693134862315807937e308",
            "9.0154242660317648163e308",
            "433917.0255669425691104980e303",
            "179769313486231580793728971405303415079934132710037826936173778980444968292764750946649017977587207096330286416692887910946555547851940402630657488671505820681908902000708383676273854845817711531764475730270069855571366959622842914819860834936475292719074168444365510704342711559699508093042880177904174497792",
            "179769313486231580793728971405303415079934132710037826936173778980444968292764750946649017977587207096330286416692887910946555547851940402630657488671505820681908902000708383676273854845817711531764475730270069855571366959622842914819860834936475292719074168444365510704342711559699508093042880177904174497791",
            "1.111111111111111111111111111111e-2147483620",
            "-84388369429940575913119257981509021684334291816379935715286",
            "0." + TString(330, '0') + "123456789",
            "1234567890" + TString(290, '7') + "e-200",
            halfwayAfterOne + TString(713, '0') + "1",
            halfwayAfterOne + TString(714, '0') + "1",
            "-" + halfwayAfterOne + TString(714, '0') + "1",
        };
        for (const TString& number : numbers) {
            const TString json = "[" + number + "]";
            rapidjson::Document document;
            document.Parse<rapidjson::kParseFullPrecisionFlag>(json.c_str());
            UNIT_ASSERT_C(!document.HasParseError(), number);
            UNIT_ASSERT_VALUES_EQUAL_C(document[0].GetDouble(), std::strtod(number.c_str(), nullptr), number);
        }
    }
}
