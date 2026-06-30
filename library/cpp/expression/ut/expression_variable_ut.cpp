#include <library/cpp/testing/unittest/registar.h>

#include "expression_variable.h"


Y_UNIT_TEST_SUITE(TExpressionVariableTest) {
    Y_UNIT_TEST(TestConstructors) {
        auto emptyExpressionVariable = TExpressionVariable();
        UNIT_ASSERT_EQUAL(emptyExpressionVariable.ToDouble(), 0.0);
        UNIT_ASSERT_EQUAL(emptyExpressionVariable.ToStr(), "");
        UNIT_ASSERT_EQUAL(ToString(emptyExpressionVariable.ToHistogramPointsAndBins()), ";");

        TString source = "test";
        auto stringExpressionVariable = TExpressionVariable(source);
        UNIT_ASSERT_EQUAL(stringExpressionVariable.ToDouble(), 0.0);
        UNIT_ASSERT_EQUAL(stringExpressionVariable.ToStr(), "test");
        UNIT_ASSERT_EQUAL(ToString(stringExpressionVariable.ToHistogramPointsAndBins()), ";");

        auto doubleExpressionVariable = TExpressionVariable(13.5);
        UNIT_ASSERT_EQUAL(doubleExpressionVariable.ToDouble(), 13.5);
        UNIT_ASSERT_EQUAL(doubleExpressionVariable.ToStr(), "13.5");
        UNIT_ASSERT_EQUAL(ToString(doubleExpressionVariable.ToHistogramPointsAndBins()), ";");

        TVector<double> points = {1, 2,};
        TVector<double> bins = {10, 20, 30};
        THistogramPointsAndBins histogramData = THistogramPointsAndBins(points, bins);
        auto histogramExpressionVariable = TExpressionVariable(histogramData);
        UNIT_ASSERT_EQUAL(histogramExpressionVariable.ToDouble(), 0.0);
        UNIT_ASSERT_EQUAL(histogramExpressionVariable.ToStr(), "1.000000,2.000000,;10.000000,20.000000,30.000000,");
        UNIT_ASSERT_EQUAL(ToString(histogramExpressionVariable.ToHistogramPointsAndBins()), "1.000000,2.000000,;10.000000,20.000000,30.000000,");

        auto variableExpressionVariable = TExpressionVariable(histogramExpressionVariable);
        UNIT_ASSERT_EQUAL(variableExpressionVariable.ToDouble(), 0.0);
        UNIT_ASSERT_EQUAL(variableExpressionVariable.ToStr(), "1.000000,2.000000,;10.000000,20.000000,30.000000,");
        UNIT_ASSERT_EQUAL(ToString(variableExpressionVariable.ToHistogramPointsAndBins()), "1.000000,2.000000,;10.000000,20.000000,30.000000,");

        auto boolExpressionVariable = TExpressionVariable(true);
        UNIT_ASSERT_EQUAL(boolExpressionVariable.ToDouble(), 1.0);
        UNIT_ASSERT_EQUAL_C(boolExpressionVariable.ToStr(), "1", boolExpressionVariable.ToStr());
        UNIT_ASSERT_EQUAL(ToString(boolExpressionVariable.ToHistogramPointsAndBins()), ";");
    }
    Y_UNIT_TEST(TestAssignmentOperator) {
        TExpressionVariable variable = 13.5;
        UNIT_ASSERT_EQUAL(variable.ToDouble(), 13.5);
        UNIT_ASSERT_EQUAL(variable.ToStr(), "13.5");
        UNIT_ASSERT_EQUAL(ToString(variable.ToHistogramPointsAndBins()), ";");

        variable = "test";
        UNIT_ASSERT_EQUAL(variable.ToDouble(), 0.0);
        UNIT_ASSERT_EQUAL(variable.ToStr(), "test");
        UNIT_ASSERT_EQUAL(ToString(variable.ToHistogramPointsAndBins()), ";");

        TVector<double> points = {1, 2,};
        TVector<double> bins = {10, 20, 30};
        THistogramPointsAndBins histogramData = THistogramPointsAndBins(points, bins);
        variable = TExpressionVariable(histogramData);
        UNIT_ASSERT_EQUAL(variable.ToDouble(), 0.0);
        UNIT_ASSERT_EQUAL(variable.ToStr(), "1.000000,2.000000,;10.000000,20.000000,30.000000,");
        UNIT_ASSERT_EQUAL(ToString(variable.ToHistogramPointsAndBins()), "1.000000,2.000000,;10.000000,20.000000,30.000000,");

        auto histogramExpressionVariable = TExpressionVariable(histogramData);
        variable = TExpressionVariable(histogramData);
        UNIT_ASSERT_EQUAL(variable.ToDouble(), 0.0);
        UNIT_ASSERT_EQUAL(variable.ToStr(), "1.000000,2.000000,;10.000000,20.000000,30.000000,");
        UNIT_ASSERT_EQUAL(ToString(variable.ToHistogramPointsAndBins()), "1.000000,2.000000,;10.000000,20.000000,30.000000,");
    }
    Y_UNIT_TEST(TestStringOperations) {
        TExpressionVariable firstOperand = TString("first");
        TExpressionVariable secondOperand = TString("second");
        TExpressionVariable partOfFirstOperand = TString("fir");
        TExpressionVariable numberOperand = TString("1");

        UNIT_ASSERT_EQUAL(firstOperand.StrLe(secondOperand), 1.0);
        UNIT_ASSERT_EQUAL(firstOperand.StrL(secondOperand), 1.0);
        UNIT_ASSERT_EQUAL(firstOperand.StrGe(secondOperand), 0.0);
        UNIT_ASSERT_EQUAL(firstOperand.StrG(secondOperand), 0.0);

        UNIT_ASSERT_EQUAL(firstOperand.StrLe(firstOperand), 1.0);
        UNIT_ASSERT_EQUAL(firstOperand.StrL(firstOperand), 0.0);
        UNIT_ASSERT_EQUAL(firstOperand.StrGe(firstOperand), 1.0);
        UNIT_ASSERT_EQUAL(firstOperand.StrG(firstOperand), 0.0);

        UNIT_ASSERT_EQUAL(firstOperand.StrStartsWith(partOfFirstOperand), 1.0);
        UNIT_ASSERT_EQUAL(firstOperand.StrStartsWith(secondOperand), 0.0);
        UNIT_ASSERT_EQUAL(secondOperand.StrStartsWith(partOfFirstOperand), 0.0);

        UNIT_ASSERT_EQUAL(firstOperand.StrCond(secondOperand, partOfFirstOperand), "fir");
        UNIT_ASSERT_EQUAL(numberOperand.StrCond(secondOperand, partOfFirstOperand), "second");
    }
    Y_UNIT_TEST(TestArithmeticOperations) {
        TExpressionVariable firstOperand = 5.2;
        TExpressionVariable secondOperand = 13.7;
        TExpressionVariable zeroOperand = 0.0;

        UNIT_ASSERT_EQUAL(firstOperand.Add(secondOperand), 18.9);
        UNIT_ASSERT_EQUAL(firstOperand.Sub(secondOperand), -8.5);
        UNIT_ASSERT_EQUAL(firstOperand.Mult(secondOperand), 71.24);

        UNIT_ASSERT_EQUAL(firstOperand.Div(secondOperand), 5.2 / 13.7);
        UNIT_ASSERT_EQUAL(secondOperand.Div(zeroOperand), std::numeric_limits<double>::infinity());
        UNIT_ASSERT(std::isnan(zeroOperand.Div(zeroOperand)));

        UNIT_ASSERT_EQUAL(firstOperand.Pow(secondOperand), std::pow(5.2, 13.7));
        UNIT_ASSERT(std::isnan(zeroOperand.Pow(zeroOperand)));

        UNIT_ASSERT_EQUAL(firstOperand.Exp(), std::exp(5.2));
        UNIT_ASSERT_EQUAL(firstOperand.Log(), std::log(5.2));
        UNIT_ASSERT_EQUAL(firstOperand.Sqr(), std::pow(5.2, 2.0));
        UNIT_ASSERT_EQUAL(firstOperand.Sqrt(), std::pow(5.2, 0.5));
        UNIT_ASSERT_EQUAL(firstOperand.Sigmoid(), 1.0 / (1.0 + std::exp(-5.2)));
        UNIT_ASSERT_EQUAL(firstOperand.Minus(), -5.2);
    }
    Y_UNIT_TEST(TestBitOperations) {
        TExpressionVariable firstOperand = 5.2;
        TExpressionVariable secondOperand = 13.7;

        UNIT_ASSERT_EQUAL(firstOperand.BitsOr(secondOperand), 13.0);
        UNIT_ASSERT_EQUAL(firstOperand.BitsAnd(secondOperand), 5.0);
    }
    Y_UNIT_TEST(TestLogicalOperations) {
        TExpressionVariable firstOperand = 5.2;
        TExpressionVariable secondOperand = 13.7;
        TExpressionVariable zeroOperand = 0.0;

        UNIT_ASSERT_EQUAL(firstOperand.Not(), 0.0);
        UNIT_ASSERT_EQUAL(zeroOperand.Not(), 1.0);

        UNIT_ASSERT_EQUAL(firstOperand.Le(secondOperand), 1.0);
        UNIT_ASSERT_EQUAL(firstOperand.L(secondOperand), 1.0);
        UNIT_ASSERT_EQUAL(firstOperand.Ge(secondOperand), 0.0);
        UNIT_ASSERT_EQUAL(firstOperand.G(secondOperand), 0.0);

        UNIT_ASSERT_EQUAL(firstOperand.Le(firstOperand), 1.0);
        UNIT_ASSERT_EQUAL(firstOperand.L(firstOperand), 0.0);
        UNIT_ASSERT_EQUAL(firstOperand.Ge(firstOperand), 1.0);
        UNIT_ASSERT_EQUAL(firstOperand.G(firstOperand), 0.0);

        UNIT_ASSERT_EQUAL(zeroOperand.Cond(firstOperand, secondOperand), 13.7);
        UNIT_ASSERT_EQUAL(secondOperand.Cond(firstOperand, secondOperand), 5.2);
    }
    Y_UNIT_TEST(TestEqualOperations) {
        TExpressionVariable firstOperand;
        TExpressionVariable secondOperand;

        UNIT_ASSERT_EQUAL(firstOperand.E(secondOperand), 0.0);
        UNIT_ASSERT_EQUAL(firstOperand.Ne(secondOperand), 1.0);

        firstOperand = 5.2;

        UNIT_ASSERT_EQUAL(firstOperand.E(secondOperand), 0.0);
        UNIT_ASSERT_EQUAL(firstOperand.Ne(secondOperand), 1.0);

        secondOperand = 13.7;

        UNIT_ASSERT_EQUAL(firstOperand.E(secondOperand), 0.0);
        UNIT_ASSERT_EQUAL(firstOperand.Ne(secondOperand), 1.0);

        firstOperand = TString("test");

        UNIT_ASSERT_EQUAL(firstOperand.E(secondOperand), 0.0);
        UNIT_ASSERT_EQUAL(firstOperand.Ne(secondOperand), 1.0);

        secondOperand = TString("test");

        UNIT_ASSERT_EQUAL(firstOperand.E(secondOperand), 1.0);
        UNIT_ASSERT_EQUAL(firstOperand.Ne(secondOperand), 0.0);

        TVector<double> points = {1, 2, 3};
        TVector<double> bins = {0, 0, 0, 0};
        firstOperand = THistogramPointsAndBins(points, bins);
        secondOperand = THistogramPointsAndBins(points, bins);

        UNIT_ASSERT_EQUAL(firstOperand.E(secondOperand), 1.0);
        UNIT_ASSERT_EQUAL(firstOperand.Ne(secondOperand), 0.0);
    }
    Y_UNIT_TEST(TestCustomOperations) {
        TExpressionVariable firstOperand = 5.2;
        TExpressionVariable secondOperand = 13.7;

        UNIT_ASSERT_EQUAL(firstOperand.Min(secondOperand), 5.2);
        UNIT_ASSERT_EQUAL(firstOperand.Min(firstOperand), 5.2);
        UNIT_ASSERT_EQUAL(secondOperand.Min(firstOperand), 5.2);

        UNIT_ASSERT_EQUAL(firstOperand.Max(secondOperand), 13.7);
        UNIT_ASSERT_EQUAL(firstOperand.Max(firstOperand), 5.2);
        UNIT_ASSERT_EQUAL(secondOperand.Max(firstOperand), 13.7);

        TVector<double> emptyPointsPoints = {0, 0, 0};
        TVector<double> emptyPointsBins = {10, 20, 10, 40};
        auto emptyPointsHistogramData = THistogramPointsAndBins(emptyPointsPoints, emptyPointsBins);
        TVector<double> emptyBinsPoints = {1, 2, 3};
        TVector<double> emptyBinsBins = {0, 0, 0, 0};
        auto emptyBinsHistogramData = THistogramPointsAndBins(emptyBinsPoints, emptyBinsBins);
        auto emptyHistogramData = THistogramPointsAndBins();
        TVector<double> randomFilledPoints = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        TVector<double> randomFilledBins = {10, 0, 5, 17, 13, 105, 6, 100, 9, 10, 0};
        auto randomFilledHistogramData = THistogramPointsAndBins(randomFilledPoints, randomFilledBins);
        TVector<double> equalPartsPoints = {1, 2, 3};
        TVector<double> equalPartsBins = {100, 100, 100, 100};
        auto equalPartsHistogramData = THistogramPointsAndBins(equalPartsPoints, equalPartsBins);
        TVector<double> firstZeroPoints = {0, 2, 3};
        TVector<double> firstZeroBins = {100, 100, 100, 100};
        auto firstZeroHistogramData = THistogramPointsAndBins(firstZeroPoints, firstZeroBins);
        TVector<double> maxIntPoints = {1, 2, std::numeric_limits<int>::max()};
        TVector<double> maxIntBins = {100, 100, 100, 0};
        auto maxIntHistogramData = THistogramPointsAndBins(maxIntPoints, maxIntBins);

        TExpressionVariable histogramOperand = emptyPointsHistogramData;
        UNIT_ASSERT_EQUAL(histogramOperand.HistogramPercentile(15), 0);

        histogramOperand = emptyBinsHistogramData;
        UNIT_ASSERT_EQUAL(histogramOperand.HistogramPercentile(25), 0);

        histogramOperand = emptyHistogramData;
        UNIT_ASSERT_EQUAL(histogramOperand.HistogramPercentile(90), 0);

        histogramOperand = randomFilledHistogramData;
        UNIT_ASSERT_EQUAL(histogramOperand.HistogramPercentile(95), 8 + (1.0 - (265 - 261.25) / 9));
        UNIT_ASSERT_EQUAL(histogramOperand.HistogramPercentile(105), 0);
        UNIT_ASSERT_EQUAL(histogramOperand.HistogramPercentile(-5), 0);

        histogramOperand = equalPartsHistogramData;
        UNIT_ASSERT_EQUAL(histogramOperand.HistogramPercentile(5), 0 + 1.0 - (100 - 400 * 0.05) / 100);
        UNIT_ASSERT_EQUAL(histogramOperand.HistogramPercentile(25), 1);

        histogramOperand = firstZeroHistogramData;
        UNIT_ASSERT_EQUAL(histogramOperand.HistogramPercentile(20), 0);
        UNIT_ASSERT_EQUAL(histogramOperand.HistogramPercentile(50), 2);

        histogramOperand = maxIntHistogramData;
        UNIT_ASSERT_EQUAL(histogramOperand.HistogramPercentile(99), 2 * 1.1);
    }
    // Здесь проверяется лишь базовая работоспособность функций сравнения версий, всевозможные corner-кейсы протестированы в expression_ut
    Y_UNIT_TEST(TestVersionOperations) {
        TExpressionVariable firstOperand = TString("15.9.3145");
        TExpressionVariable secondOperand = TString("15.9.3245");

        UNIT_ASSERT_EQUAL(firstOperand.VerE(secondOperand), 0.0);
        UNIT_ASSERT_EQUAL(firstOperand.VerNe(secondOperand), 1.0);
        UNIT_ASSERT_EQUAL(firstOperand.VerE(firstOperand), 1.0);
        UNIT_ASSERT_EQUAL(firstOperand.VerLe(secondOperand), 1.0);
        UNIT_ASSERT_EQUAL(firstOperand.VerLe(firstOperand), 1.0);
        UNIT_ASSERT_EQUAL(firstOperand.VerL(secondOperand), 1.0);
        UNIT_ASSERT_EQUAL(firstOperand.VerGe(secondOperand), 0.0);
        UNIT_ASSERT_EQUAL(firstOperand.VerGe(firstOperand), 1.0);
        UNIT_ASSERT_EQUAL(firstOperand.VerG(secondOperand), 0.0);
    }

} // TExpressionVariableTest
