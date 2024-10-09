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
}
