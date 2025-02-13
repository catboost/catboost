#include <library/cpp/testing/unittest/registar.h>

#include <library/cpp/regex/pire/regexp.h>

#include <util/stream/output.h>
#include <utility>

#include <util/generic/buffer.h>
#include <util/generic/map.h>
#include <util/generic/vector.h>
#include <util/generic/ptr.h>
#include <util/generic/ymath.h>
#include <util/generic/ylimits.h>

#include <util/random/random.h>

#include "expression.h"

Y_UNIT_TEST_SUITE(TCalcExpressionTest) {
    Y_UNIT_TEST(TestCalcExpression) {
        THashMap<TString, double> m;
        m["mv"] = 32768;
        m["big"] = 68719476736;
        m["small"] = 1;
        m["neg"] = -1000.;

        UNIT_ASSERT_EQUAL(CalcExpression("1 == 1", m), 1);
        UNIT_ASSERT_EQUAL(CalcExpression("1 == 0", m), 0);
        UNIT_ASSERT_EQUAL(CalcExpression("1 > 0", m), 1);
        UNIT_ASSERT_EQUAL(CalcExpression("(2 - 1) > 0", m), 1);
        UNIT_ASSERT_EQUAL(CalcExpression("(2 - 1) > 1", m), 0);
        UNIT_ASSERT_EQUAL(CalcExpression("2 - 2 - 2", m), -2);
        UNIT_ASSERT_VALUES_EQUAL(CalcExpression("0*neg", m), 0);
        UNIT_ASSERT_VALUES_EQUAL(CalcExpression("1 * -1000", m), -1000);
        UNIT_ASSERT_VALUES_EQUAL(CalcExpression("1 * +1000", m), 1000);
        UNIT_ASSERT_VALUES_EQUAL(CalcExpression("small * +1000", m), 1000);
        UNIT_ASSERT_VALUES_EQUAL(CalcExpression("small * -1000", m), -1000);
        UNIT_ASSERT_VALUES_EQUAL(CalcExpression("small * -small", m), -1);
        UNIT_ASSERT_VALUES_EQUAL(CalcExpression("(small + small) * -small", m), -2);
        UNIT_ASSERT_VALUES_EQUAL(CalcExpression("0*-1000", m), 0);

        UNIT_ASSERT_EQUAL(CalcExpression("-2 + 2", m), 0);
        UNIT_ASSERT_EQUAL(CalcExpression("mv&32768==32768", m), 1);

        UNIT_ASSERT_EQUAL(CalcExpression("big&34359738368==big&68719476736", m), 0); //big & (1<<35) != big & 1 << 36
        UNIT_ASSERT_EQUAL(CalcExpression("small&34359738369==1", m), 1);

        UNIT_ASSERT_EQUAL(CalcExpression("(mv&32768==32768)||(mv&32768==32768)||(2 - 2 - 2)||(2 - 2 - 2)&&(mv&32768==32768)", m), 1);

        m["a"] = 0.1;
        m["size"] = 18;
        UNIT_ASSERT_EQUAL(CalcExpression("(a*11) > 1", m), 1);
        UNIT_ASSERT_EQUAL(CalcExpression("(9*a) > 1 && size > 20", m), 0);
        UNIT_ASSERT_EQUAL(CalcExpression("2+2*2", m), 6);
        UNIT_ASSERT_EQUAL(CalcExpression("2+2^3", m), 10);
        UNIT_ASSERT_EQUAL(CalcExpression("2 - 20*2", m), -38);
        UNIT_ASSERT_EQUAL(CalcExpression("1 #MIN# 2", m), 1);
        UNIT_ASSERT_EQUAL(CalcExpression("2 #MIN# 1", m), 1);
        UNIT_ASSERT_EQUAL(CalcExpression("1 #MAX# 2", m), 2);
        UNIT_ASSERT_EQUAL(CalcExpression("2 #MAX# 1", m), 2);
        UNIT_ASSERT_EQUAL(CalcExpression("2 #MAX# 1 #MIN# 3", m), 2);
        UNIT_ASSERT_EQUAL(CalcExpression("2+2*2 #MAX# 2+2^3", m), 10);
        UNIT_ASSERT_EQUAL(CalcExpression("2+2*2 #MIN# 2+2^3", m), 6);
        UNIT_ASSERT_EQUAL(CalcExpression("1 < 0 #MAX# a + 100", m), 100.1);
        UNIT_ASSERT_EQUAL(CalcExpression("(2*a^(-3)-2000)^2<=10^(-5)", m), 1); // comparison adds EPS to lhs
        UNIT_ASSERT_EQUAL(CalcExpression("2*a^(-3)==2000", m), 1);
        UNIT_ASSERT_EQUAL(IsNan(CalcExpression("0^0", m)), 1);
        UNIT_ASSERT_EQUAL(IsNan(CalcExpression("0.0^0", m)), 1);
        UNIT_ASSERT_EQUAL(CalcExpression("0.00001^0", m), 1);
        UNIT_ASSERT_EQUAL(CalcExpression("0.0^0.00001", m), 0);

        // Sum with boolean
        UNIT_ASSERT_EQUAL(CalcExpression("2+(2==2)", m), 3);
        UNIT_ASSERT_EQUAL(CalcExpression("2+(2==0)", m), 2);
        UNIT_ASSERT_EQUAL(CalcExpression("~big+~small+~azaza", m), 2);

        THashMap<TString, THistogramPointsAndBins> histogramDataMap;
        TVector<double> randomFilledPoints = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        TVector<double> randomFilledBins = {10, 0, 5, 17, 13, 105, 6, 100, 9, 10, 0};
        auto randomFilledHistogramData = THistogramPointsAndBins(randomFilledPoints, randomFilledBins);
        TVector<double> equalPartsPoints = {1, 2, 3};
        TVector<double> equalPartsBins = {100, 100, 100, 100};
        auto equalPartsHistogramData = THistogramPointsAndBins(equalPartsPoints, equalPartsBins);
        TVector<double> firstZeroPoints = {0, 2, 3};
        TVector<double> firstZeroBins = {100, 100, 100, 100};
        auto firstZeroHistogramData = THistogramPointsAndBins(firstZeroPoints, firstZeroBins);
        TVector<double> emptyPartsPoints = {1, 2, 3};
        TVector<double> emptyPartsBins = {0, 0, 0, 0};
        auto emptyPartsHistogramData = THistogramPointsAndBins(emptyPartsPoints, emptyPartsBins);
        TVector<double> maxIntPoints = {1, 2, std::numeric_limits<int>::max()};
        TVector<double> maxIntBins = {100, 100, 100, 0};
        auto maxIntHistogramData = THistogramPointsAndBins(maxIntPoints, maxIntBins);

        histogramDataMap["random.feature"] = randomFilledHistogramData;
        histogramDataMap["equal.parts.feature%"] = equalPartsHistogramData;
        histogramDataMap["equal.parts.feature-with-dash%"] = equalPartsHistogramData;
        histogramDataMap["equal.parts.feature/with/divide%"] = equalPartsHistogramData;
        histogramDataMap["empty.parts.feature%"] = emptyPartsHistogramData;
        histogramDataMap["first.zero.feature%"] = firstZeroHistogramData;
        histogramDataMap["max.int.feature%"] = maxIntHistogramData;

        UNIT_ASSERT_EQUAL(CalcExpression("#HISTOGRAM_PERCENTILE# random.feature, 95", histogramDataMap), 8 + (1.0 - (265 - 261.25) / 9));

        UNIT_ASSERT_EQUAL(CalcExpression("#HISTOGRAM_PERCENTILE# max.int.feature%, 99", histogramDataMap), 2 * 1.1);

        // (-inf, 0]
        UNIT_ASSERT_EQUAL(CalcExpression("#HISTOGRAM_PERCENTILE# first.zero.feature%, 5", histogramDataMap), 0);
        UNIT_ASSERT_EQUAL(CalcExpression("#HISTOGRAM_PERCENTILE# first.zero.feature%, 20", histogramDataMap), 0);
        // (0, 2]
        UNIT_ASSERT_EQUAL(CalcExpression("#HISTOGRAM_PERCENTILE# first.zero.feature%, 30", histogramDataMap), 2 * (1.0 - (200 - 400 * 0.3) / 100));
        UNIT_ASSERT_EQUAL(CalcExpression("#HISTOGRAM_PERCENTILE# first.zero.feature%, 50", histogramDataMap), 2);

        UNIT_ASSERT_EQUAL(CalcExpression("#HISTOGRAM_PERCENTILE# equal.parts.feature%, 5", histogramDataMap), 0 + 1.0 - (100 - 400 * 0.05) / 100);
        UNIT_ASSERT_EQUAL(CalcExpression("#HISTOGRAM_PERCENTILE# equal.parts.feature%, 20", histogramDataMap), 0 + 1.0 - (100 - 400 * 0.2) / 100);
        UNIT_ASSERT_EQUAL(CalcExpression("#HISTOGRAM_PERCENTILE# equal.parts.feature%, 25", histogramDataMap), 1);
        UNIT_ASSERT_EQUAL(CalcExpression("#HISTOGRAM_PERCENTILE# equal.parts.feature%, 50", histogramDataMap), 2);
        UNIT_ASSERT_EQUAL(CalcExpression("#HISTOGRAM_PERCENTILE# equal.parts.feature%, 75", histogramDataMap), 3);
        UNIT_ASSERT_EQUAL(CalcExpression("#HISTOGRAM_PERCENTILE# equal.parts.feature%, 99", histogramDataMap), 3 * 1.1);

        UNIT_ASSERT_EQUAL(CalcExpression("#HISTOGRAM_PERCENTILE# 'equal.parts.feature-with-dash%', 99", histogramDataMap), 3 * 1.1);

        UNIT_ASSERT_EQUAL(CalcExpression("5 + #HISTOGRAM_PERCENTILE# equal.parts.feature%, 50 - 3", histogramDataMap), 4);
        UNIT_ASSERT_EQUAL(CalcExpression("5 + #HISTOGRAM_PERCENTILE# equal.parts.feature%, 50 / 0.5", histogramDataMap), 9);
        UNIT_ASSERT_EQUAL(CalcExpression("(5 + #HISTOGRAM_PERCENTILE# equal.parts.feature%, 50) / 7", histogramDataMap), 1);
        UNIT_ASSERT_EQUAL(CalcExpression("#HISTOGRAM_PERCENTILE# equal.parts.feature%, 50 + 5", histogramDataMap), 7);
        UNIT_ASSERT_EQUAL(CalcExpression("#HISTOGRAM_PERCENTILE# equal.parts.feature%, 50 + #HISTOGRAM_PERCENTILE# equal.parts.feature%, 75", histogramDataMap), 5);
        UNIT_ASSERT_EQUAL(CalcExpression("#HISTOGRAM_PERCENTILE# equal.parts.feature%, #HISTOGRAM_PERCENTILE# equal.parts.feature%, 99", histogramDataMap), 0 + 1.0 - (100 - 4 * 3 * 1.1) / 100);

        // Invalid #HISTOGRAM_PERCENTILE#
        UNIT_ASSERT_EQUAL(CalcExpression("#HISTOGRAM_PERCENTILE# #HISTOGRAM_PERCENTILE# equal.parts.feature%, 5, 5", histogramDataMap), 0);
        UNIT_ASSERT_EQUAL(CalcExpression("#HISTOGRAM_PERCENTILE# equal.parts.feature%, ", histogramDataMap), 0);
        UNIT_ASSERT_EQUAL(CalcExpression("#HISTOGRAM_PERCENTILE# equal.parts.feature%, 0", histogramDataMap), 0);
        UNIT_ASSERT_EQUAL(CalcExpression("#HISTOGRAM_PERCENTILE# equal.parts.feature%, 100", histogramDataMap), 0);
        UNIT_ASSERT_EQUAL(CalcExpression("#HISTOGRAM_PERCENTILE#", histogramDataMap), 0);
        UNIT_ASSERT_EQUAL(CalcExpression("#HISTOGRAM_PERCENTILE# 1,", histogramDataMap), 0);
        UNIT_ASSERT_EQUAL(CalcExpression("#HISTOGRAM_PERCENTILE# 1,,", histogramDataMap), 0);
        UNIT_ASSERT_EQUAL(CalcExpression("#HISTOGRAM_PERCENTILE# ,5", histogramDataMap), 0);
        UNIT_ASSERT_EQUAL(CalcExpression("#HISTOGRAM_PERCENTILE# ,105", histogramDataMap), 0);
        UNIT_ASSERT_EQUAL(CalcExpression("#HISTOGRAM_PERCENTILE# ,random.feature", histogramDataMap), 0);
        UNIT_ASSERT_EQUAL(CalcExpression("#HISTOGRAM_PERCENTILE# equal.parts.feature%, 20, 20", histogramDataMap), 0);
        UNIT_ASSERT_EQUAL(CalcExpression("#HISTOGRAM_PERCENTILE# empty.parts.feature%, 75", histogramDataMap), 0);
        UNIT_ASSERT_EQUAL(CalcExpression("#HISTOGRAM_PERCENTILE# equal.parts.feature-with-dash%, 99", histogramDataMap), 0);
        UNIT_ASSERT_EQUAL(CalcExpression("#HISTOGRAM_PERCENTILE# \"equal.parts.feature-with-dash%\", 99", histogramDataMap), 0);

        UNIT_CHECK_GENERATED_EXCEPTION(CalcExpression("#HISTOGRAM_PERCENTILE# (equal.parts.feature%, 20)", histogramDataMap), yexception);
        UNIT_CHECK_GENERATED_EXCEPTION(CalcExpression("#HISTOGRAM_PERCENTILE# equal.parts.feature/with/divide%, 99'", histogramDataMap), yexception);
        UNIT_CHECK_GENERATED_EXCEPTION(CalcExpression("#HISTOGRAM_PERCENTILE# 'equal.parts.feature/with/divide%', 99'", histogramDataMap), yexception);
        UNIT_CHECK_GENERATED_EXCEPTION(CalcExpression("#HISTOGRAM_PERCENTILE# \"equal.parts.feature/with/divide%\", 99\"", histogramDataMap), yexception);
    }

    Y_UNIT_TEST(TestStringExpression) {
        THashMap<TString, TString> m;
        m["a"] = "b";
        m["vm"] = "32768";
        m["qc"] = "1";
        m["l"] = "2.";
        m["version"] = "15.9.3145";
        const TString key = "UPPER.ApplyBlender_MarketExp_exp.#insrule_Vertical/market";
        const TString strVal = "UPPER.ApplyBlender_MarketExp_exp.#insrule_Vertical";
        m[key] = "1";
        UNIT_ASSERT_EQUAL(CalcExpression("'" + key + "'==1", m), 1);
        UNIT_ASSERT_EQUAL(CalcExpression("\"" + strVal + "\"==" + strVal, m), 1);
        UNIT_ASSERT_EQUAL(CalcExpression("vm&32768==32768", m), 1);
        UNIT_ASSERT_EQUAL(CalcExpression("(qc&1)==1", m), 1);
        UNIT_ASSERT_EQUAL(CalcExpression("a == b", m), 1);
        UNIT_ASSERT_EQUAL(CalcExpression("a == a", m), 1);
        UNIT_ASSERT_EQUAL(CalcExpression("a == c", m), 0);
        UNIT_ASSERT_EQUAL(CalcExpression("#EXP#l", m), exp(2.));
        UNIT_ASSERT_EQUAL(CalcExpression("#EXP# l + 1", m), exp(2.) + 1);
        UNIT_ASSERT_EQUAL(CalcExpression("#EXP#(l + 1)", m), exp(3.));
        UNIT_ASSERT_EQUAL(CalcExpression("-#EXP#(l + 1)", m), -exp(3.));
        UNIT_ASSERT_EQUAL(CalcExpression("#LOG#l", m), log(2.));
        UNIT_ASSERT_EQUAL(CalcExpression("#LOG# l + 1", m), log(2.) + 1);
        UNIT_ASSERT_EQUAL(CalcExpression("#LOG#(l + 1)", m), log(3.));
        UNIT_ASSERT_EQUAL(CalcExpression("-#LOG#(l + 1)", m), -log(3.));
        UNIT_ASSERT_EQUAL(CalcExpression("#SIGMOID#l", m), (1.0 / (1.0 + exp(-2))));
        UNIT_ASSERT_EQUAL(CalcExpression("#SIGMOID# l + 1", m), (1.0 + 1.0 / (1.0 + exp(-2))));
        UNIT_ASSERT_EQUAL(CalcExpression("#SIGMOID#(l + 1)", m), (1.0 / (1.0 + exp(-3))));
        UNIT_ASSERT_EQUAL(CalcExpression("-#SIGMOID#(l + 1)", m), -(1.0 / (1.0 + exp(-3))));
        UNIT_ASSERT_EQUAL(CalcExpression("#EXP#(-#LOG#(l + 1) + #EXP#(l + 3))", m), exp(-log(3.) + exp(5.)));
        UNIT_ASSERT_EQUAL(CalcExpression("~a", m), 1);
        UNIT_ASSERT_EQUAL(CalcExpression("~b", m), 0);
        UNIT_ASSERT_EQUAL(CalcExpression("~a && a == \"b\"", m), 1);
        UNIT_ASSERT_EQUAL(CalcExpression("~b && b == \"b\"", m), 0);
        UNIT_ASSERT_EQUAL(CalcExpressionStr("a", m), TString("b"));
        UNIT_ASSERT_EQUAL(CalcExpressionStr("\"c\"", m), TString("c"));

        // alphabetical comparsion
        UNIT_ASSERT_EQUAL(CalcExpressionStr("version", m), TString("15.9.3145"));
        UNIT_ASSERT_EQUAL(CalcExpression("version >@ \"1\"", m), 1);
        UNIT_ASSERT_EQUAL(CalcExpression("version >@ \"15\"", m), 1);
        UNIT_ASSERT_EQUAL(CalcExpression("version >@ \"20\"", m), 0);
        UNIT_ASSERT_EQUAL(CalcExpression("version <@ \"16.0.00\"", m), 1);
        UNIT_ASSERT_EQUAL(CalcExpression("version >=@ \"15.9\"", m), 1);
        UNIT_ASSERT_EQUAL(CalcExpression("version >=@ \"1.0.0.1\"", m), 1);
        UNIT_ASSERT_EQUAL(CalcExpression("version >=@ \"16.0.00\"", m), 0);
        UNIT_ASSERT_EQUAL(CalcExpression("version <=@ \"16.2.4.5\"", m), 1);
        UNIT_ASSERT_EQUAL(CalcExpression("version <=@ \"15.9.3145\"", m), 1);
        UNIT_ASSERT_EQUAL(CalcExpression("version <=@ \"20\"", m), 1);
        UNIT_ASSERT_EQUAL(CalcExpression("version <=@ \"10\"", m), 0);
        UNIT_ASSERT_EQUAL(CalcExpression("version <=@ \"10.0\"", m), 0);

        // versional comparsion
        /* Сравнение версий по стандарту uatraits:
        1) Версию мы всегда считаем из первых четырех чисел, остальные игнорируем.
        Если меньше четырех, добавляем недостающие нули.
        2) Если в каком-то из операндов мусор, какие-то префиксы или постфиксы, то
        операторы <#, <=#, >#, >=# и ==# вернут false, !=# вернёт true. */
        UNIT_ASSERT_EQUAL(CalcExpression("\"15.9.3145\" ==# \"15.9.3145\"", m), 1);
        UNIT_ASSERT_EQUAL(CalcExpression("\"15.9.3145\" ==# \"15.9.3245\"", m), 0);
        UNIT_ASSERT_EQUAL(CalcExpression("\"15.9\" ==# \"15.9.0.0\"", m), 1);
        UNIT_ASSERT_EQUAL(CalcExpression("\"15.9\" ==# \"15.9.0\"", m), 1);
        UNIT_ASSERT_EQUAL(CalcExpression("\"15.9.beta.3145\" ==# \"15.9.0.3145\"", m), 0);
        UNIT_ASSERT_EQUAL(CalcExpression("\"prestable-15.9.3145.2\" ==# \"0.9.3145.2\"", m), 0);
        UNIT_ASSERT_EQUAL(CalcExpression("\"!..chat.2\" ==# \"0.0.0.2\"", m), 0);
        UNIT_ASSERT_EQUAL(CalcExpression("\"alpha\" ==# \"beta\"", m), 0);

        UNIT_ASSERT_EQUAL(CalcExpression("\"15.9.3145\" !=# \"15.9.3145\"", m), 0);
        UNIT_ASSERT_EQUAL(CalcExpression("\"15.9.3145\" !=# \"15.9.3245\"", m), 1);
        UNIT_ASSERT_EQUAL(CalcExpression("\"15.9\" !=# \"15.9.0.0\"", m), 0);
        UNIT_ASSERT_EQUAL(CalcExpression("\"15.9\" !=# \"15.9.0\"", m), 0);
        UNIT_ASSERT_EQUAL(CalcExpression("\"15.9.beta.3145\" !=# \"15.9.0.3145\"", m), 1);
        UNIT_ASSERT_EQUAL(CalcExpression("\"prestable-15.9.3145.2\" !=# \"0.9.3145.2\"", m), 1);
        UNIT_ASSERT_EQUAL(CalcExpression("\"!..chat.2\" !=# \"0.0.0.2\"", m), 1);
        UNIT_ASSERT_EQUAL(CalcExpression("\"alpha\" !=# \"beta\"", m), 1);

        UNIT_ASSERT_EQUAL(CalcExpression("\"15.9.3145\" ># \"1\"", m), 1);
        UNIT_ASSERT_EQUAL(CalcExpression("\"15.9.3145\" ># \"15\"", m), 1);
        UNIT_ASSERT_EQUAL(CalcExpression("\"15.9.3145\" ># \"15.8\"", m), 1);
        UNIT_ASSERT_EQUAL(CalcExpression("\"15.9.3145\" ># \"15.8.20\"", m), 1);
        UNIT_ASSERT_EQUAL(CalcExpression("\"15.9.3145\" ># \"15.9.3144\"", m), 1);
        UNIT_ASSERT_EQUAL(CalcExpression("\"15.9.3145\" ># \"15.9.3145\"", m), 0);
        UNIT_ASSERT_EQUAL(CalcExpression("\"15.9.3145\" ># \"15.9.3145.1\"", m), 0);
        UNIT_ASSERT_EQUAL(CalcExpression("\"15.9.3145\" ># \"15.9.3146\"", m), 0);
        UNIT_ASSERT_EQUAL(CalcExpression("\"15.9.3145\" ># \"20\"", m), 0);
        UNIT_ASSERT_EQUAL(CalcExpression("\"15.9.3145\" ># \"20.0.0.0\"", m), 0);
        UNIT_ASSERT_EQUAL(CalcExpression("\"15.9.3145\" ># \"15.0-stable\"", m), 0);
        UNIT_ASSERT_EQUAL(CalcExpression("\"15.9.3145\" ># \"15.10-beta\"", m), 0);
        UNIT_ASSERT_EQUAL(CalcExpression("\"15.0.2\" ># \"15.0-stable.beta\"", m), 0);
        UNIT_ASSERT_EQUAL(CalcExpression("\"15.0.2\" ># \"15.10-beta.3\"", m), 0);

        UNIT_ASSERT_EQUAL(CalcExpression("\"15.9.3145\" <# \"16.0.00\"", m), 1);
        UNIT_ASSERT_EQUAL(CalcExpression("\"15.9.3145\" <# \"15.10.0.0\"", m), 1);
        UNIT_ASSERT_EQUAL(CalcExpression("\"15.9.3145\" <# \"15.9.3146.0\"", m), 1);
        UNIT_ASSERT_EQUAL(CalcExpression("\"15.9.3145\" <# \"15.9.3145.2\"", m), 1);
        UNIT_ASSERT_EQUAL(CalcExpression("\"15.9.3145\" <# \"15.9.314\"", m), 0);
        UNIT_ASSERT_EQUAL(CalcExpression("\"15.9.3145\" <# \"15.0-stable\"", m), 0);
        UNIT_ASSERT_EQUAL(CalcExpression("\"15.9.3145\" <# \"15.10-beta\"", m), 0);

        UNIT_ASSERT_EQUAL(CalcExpression("\"15.9.3145\" >=# \"15.9\"", m), 1);
        UNIT_ASSERT_EQUAL(CalcExpression("\"15.9.3145\" >=# \"1.0.0.1\"", m), 1);
        UNIT_ASSERT_EQUAL(CalcExpression("\"15.9.3145\" >=# \"15.9.3145\"", m), 1);
        UNIT_ASSERT_EQUAL(CalcExpression("\"15.9.3145\" >=# \"15.9.3145.alpha\"", m), 0);
        UNIT_ASSERT_EQUAL(CalcExpression("\"15.9.3145\" >=# \"16.0.00\"", m), 0);

        UNIT_ASSERT_EQUAL(CalcExpression("\"15.9.3145\" <=# \"16.2.4.5\"", m), 1);
        UNIT_ASSERT_EQUAL(CalcExpression("\"15.9.3145\" <=# \"15.9.3145\"", m), 1);
        UNIT_ASSERT_EQUAL(CalcExpression("\"15.9.3145\" <=# \"20\"", m), 1);
        UNIT_ASSERT_EQUAL(CalcExpression("\"15.9.3145\" <=# \"10\"", m), 0);
        UNIT_ASSERT_EQUAL(CalcExpression("\"15.9.3145\" <=# \"10.0\"", m), 0);


        UNIT_ASSERT_EQUAL(CalcExpression("version >? \"15.9\"", m), 1);
        UNIT_ASSERT_EQUAL(CalcExpression("version >? \"1\"", m), 1);
        UNIT_ASSERT_EQUAL(CalcExpression("version >? \"16\"", m), 0);

        UNIT_ASSERT_EQUAL(CalcExpression("#SQR#l", m), pow(2., 2.));
        UNIT_ASSERT_EQUAL(CalcExpression("#SQR# l + 1", m), pow(2., 2) + 1);
        UNIT_ASSERT_EQUAL(CalcExpression("#SQR#(l + 1)", m), pow(3., 2));
        UNIT_ASSERT_EQUAL(CalcExpression("-#SQRT#(l + 1)", m), -pow(3., 0.5));

        UNIT_ASSERT_EXCEPTION(CalcExpressionStr("\"", m), yexception);
        UNIT_ASSERT_EXCEPTION(CalcExpressionStr("qqqqq\"", m), yexception);
        UNIT_ASSERT_EXCEPTION(CalcExpressionStr("\"qqqqq", m), yexception);
        UNIT_ASSERT_EXCEPTION(CalcExpressionStr("\'qqqqq", m), yexception);
        UNIT_ASSERT_EQUAL(CalcExpression("''", m), 0);
        m[""] = "1";
        UNIT_ASSERT_EQUAL(CalcExpression("''", m), 1);
    }

    Y_UNIT_TEST(TestDoubleExpression) {
        THashMap<TString, double> m;
        m["rand"] = 0.0001;
        m["IntentProbability"] = 0.23;
        m["filter"] = 1;
        UNIT_ASSERT_EQUAL(CalcExpression("rand < 0.001", m), 1);
        UNIT_ASSERT_EQUAL(CalcExpression("filter > 0.5 && rand < 0.001", m), 1);
        UNIT_ASSERT_EQUAL(CalcExpression("filter > 0.5 && rand < 0.001 && IntentProbability > 0.2", m), 1);
    }

    Y_UNIT_TEST(TestPushBackToVectorUB) {
        THashMap<TString, double> m;
        TString expr = "0";
        const size_t n = 100;
        for (size_t i = 1; i <= n; ++i)
            expr += " + " + ToString(i);
        expr = "(" + expr + ") * (" + expr + ")";
        const size_t sum = n * (n + 1) / 2;
        UNIT_ASSERT_EQUAL(CalcExpression(expr, m), sum * sum);
    }

    Y_UNIT_TEST(TestGetTokens) {
        TExpression expression{"(ab + ac - bc > 0) || c"};
        TVector<TString> tokens;
        expression.GetTokensWithPrefix("a", tokens);
        UNIT_ASSERT_VALUES_EQUAL(tokens.size(), 2);
        expression.GetTokensWithSuffix("c", tokens);
        UNIT_ASSERT_VALUES_EQUAL(tokens.size(), 3);
    }

    Y_UNIT_TEST(TestUnknownExpression) {
        THashMap<TString, double> m;
        m["mv"] = 32768;
        m["big"] = 68719476736;
        m["small"] = 1;

        // test unknown parameter, that is not present in dictionary "m"
        UNIT_ASSERT_EQUAL(CalcExpression("unknown == 1", m), 0);
        UNIT_ASSERT_EQUAL(CalcExpression("unknown == 0", m), 0);
        // but in sum it has value 0
        UNIT_ASSERT_EQUAL(CalcExpression("unknown + small + mv", m), 32769);
        // in comparisons < or > too
        UNIT_ASSERT_EQUAL(CalcExpression("unknown > 1", m), 0);
        UNIT_ASSERT_EQUAL(CalcExpression("unknown > 0", m), 0);
        UNIT_ASSERT_EQUAL(CalcExpression("unknown > (-1)", m), 1);
        UNIT_ASSERT_EQUAL(CalcExpression("unknown < 1", m), 1);
        UNIT_ASSERT_EQUAL(CalcExpression("unknown < 0", m), 0);
        UNIT_ASSERT_EQUAL(CalcExpression("unknown < (-1)", m), 0);
    }

    Y_UNIT_TEST(TestDivZero) {
        THashMap<TString, double> m;
        m["mv"] = 32768.2;
        m["zero"] = 0;

        UNIT_ASSERT_EQUAL(CalcExpressionStr("zero / zero", m), "nan");
        UNIT_ASSERT_EQUAL(CalcExpressionStr("mv / zero", m), "inf");
    }

    Y_UNIT_TEST(TestCond) {
        THashMap<TString, double> m;
        THashMap<TString, TString> n;
        n["A"] = "a_str";
        n["B"] = "b_str";
        UNIT_ASSERT_EQUAL(CalcExpression("1 ? 2 : 3", m), 2);
        UNIT_ASSERT_EQUAL(CalcExpression("1 ? 1+1 : 3 + 100", m), 2);
        UNIT_ASSERT_EQUAL(CalcExpression("0 ? 0 : 3", m), 3);
        UNIT_ASSERT_EQUAL(CalcExpression("1 ? \"abc.def\" >? \"abc\" : 0", m), 1);
        UNIT_ASSERT_EQUAL(CalcExpression("1 ? 1 : 0 ? 2 : 3", m), CalcExpression("1 ? 1 : (0 ? 2 : 3)", m));
        UNIT_ASSERT_EQUAL(CalcExpression("1 ? 77 : 2 && 1", m), 77);
        UNIT_ASSERT_EQUAL(CalcExpressionStr("1 ?@ \"one\" : \"two\"", n), TString("one"));
        UNIT_ASSERT_EQUAL(CalcExpressionStr("0 ?@ \"one\" : \"two\"", n), TString("two"));
        UNIT_ASSERT_EQUAL(CalcExpressionStr("1 ?@ A : B", n), TString("a_str"));
        UNIT_ASSERT_EQUAL(CalcExpressionStr("1 > 0 ?@ \"one\" : \"two\"", n), TString("one"));
        UNIT_ASSERT_EQUAL(CalcExpressionStr("1 ?@ 2 : 3", n), TString("2"));
        UNIT_ASSERT_EQUAL(CalcExpressionStr("1 ?@ 2:3", n), TString("2"));
        UNIT_ASSERT_EQUAL(CalcExpressionStr("1?@2:3", n), TString("2"));
    }

    Y_UNIT_TEST(TestText) {
        THashMap<TString, TString> m;
        m["A"] = "a_str";
        m["B"] = "b_str";
        m["C"] = "a_str";
        m["D"] = "";
        UNIT_ASSERT_EQUAL(CalcExpression("A == \"a_str\"", m), 1);
        UNIT_ASSERT_EQUAL(CalcExpression("A != \"a_str\"", m), 0);
        UNIT_ASSERT_EQUAL(CalcExpression("A == a_str", m), 1);
        UNIT_ASSERT_EQUAL(CalcExpression("A != a_str", m), 0);
        UNIT_ASSERT_EQUAL(CalcExpression("A == A", m), 1);
        UNIT_ASSERT_EQUAL(CalcExpression("A != A", m), 0);
        UNIT_ASSERT_EQUAL(CalcExpression("A == B", m), 0);
        UNIT_ASSERT_EQUAL(CalcExpression("A != B", m), 1);
        UNIT_ASSERT_EQUAL(CalcExpression("A == C", m), 1);
        UNIT_ASSERT_EQUAL(CalcExpression("A != C", m), 0);
        UNIT_ASSERT_EQUAL(CalcExpression("~D", m), 1);
        UNIT_ASSERT_EQUAL(CalcExpression("D == ''", m), 1);
        UNIT_ASSERT_EQUAL(CalcExpression("D != ''", m), 0);
    }

    Y_UNIT_TEST(TestInvalidCond) {
        THashMap<TString, double> m;
        UNIT_CHECK_GENERATED_EXCEPTION(CalcExpression("(2 > 3 ? 41+1) : 666 / 2", m), yexception);
    }

    Y_UNIT_TEST(TestRegex) {
        THashMap<TString, TString> m;
        m["A"] = "a_str";
        m["cyr"] = "кириллица";
        m["RX"] = "._s[a-z]+";
        auto calcExpression = [&m](TStringBuf expr) {
            return TExpression{expr}.SetRegexMatcher([](TStringBuf str, TStringBuf rx) {
                const auto opts = NRegExp::TFsm::TOptions{}.SetCharset(CODES_UTF8);
                return NRegExp::TMatcher{NRegExp::TFsm{rx, opts}}.Match(str).Final();
            }).CalcExpression(m);
        };
        UNIT_ASSERT_EQUAL(calcExpression("A =~ RX"), 1);
        UNIT_ASSERT_EQUAL(calcExpression("A =~ \"b\""), 0);
        UNIT_ASSERT_EQUAL(calcExpression("A =~ \"..s.*\""), 1);
        UNIT_ASSERT_EQUAL(calcExpression("A =~ \".*r\""), 1);
        UNIT_ASSERT_EQUAL(calcExpression("A =~ A"), 1);
        UNIT_ASSERT_EQUAL(calcExpression("cyr =~ \"к.р[и]л*ица\""), 1);
    }

} // TCalcExpressionTest
