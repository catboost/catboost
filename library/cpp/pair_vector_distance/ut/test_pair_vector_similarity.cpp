#include <library/cpp/pair_vector_distance/pair_vector_similarity.h>
#include <library/cpp/testing/unittest/registar.h>


namespace NPairVectorSimilarity {

    Y_UNIT_TEST_SUITE(UrlMetricTests) {
        Y_UNIT_TEST(PairVectorSimilarityTest1) {
            float lhs[] {1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0};
            float rhs[] {4.0, 3.0, 2.0, 1.0, 3.0, 4.0, 1.0, 2.0};
            UNIT_ASSERT_DOUBLES_EQUAL(PairVectorSimilarityMetric(lhs, rhs, 8), 0.8496732026143791, 1e-7);
        }
        Y_UNIT_TEST(PairVectorSimilarityTest2) {
            float lhs[] {5.0, 2.0, -2.0, 4.0, 1.0, 2.0, 3.0, 4.0};
            float rhs[] {4.0, 3.0, 2.0, 1.0, -2.0, -2.0, -3.0, 2.0};
            UNIT_ASSERT_DOUBLES_EQUAL(PairVectorSimilarityMetric(lhs, rhs, 8), 0.5043767232556493, 1e-7);
        }
        Y_UNIT_TEST(PairVectorSimilarityTest3) {
            float lhs[] {5.0, 2.0, -2.0, 4.0, 1.0, 2.0, 3.0, 4.0};
            float rhs[] {5.0, 2.0, -2.0, 4.0, -1.0, -2.0, -3.0, -4.0};
            UNIT_ASSERT_DOUBLES_EQUAL(PairVectorSimilarityMetric(lhs, rhs, 8), 0.0, 1e-7);
        }
        Y_UNIT_TEST(PairVectorSimilarityTest4) {
            float lhs[] {5.0, 2.0, -2.0, 4.0, 1.0, 2.0, 3.0, 4.0};
            float rhs[] {5.0, 2.0, -2.0, 4.0, 1.0, 2.0, 3.0, 4.0};
            UNIT_ASSERT_DOUBLES_EQUAL(PairVectorSimilarityMetric(lhs, rhs, 8), 1.0, 1e-7);
        }
        Y_UNIT_TEST(PairVectorSimilarityTest5) {
            float lhs[] {0.809017,0.587785};
            float rhs[] {0.309017,0.951057};
            UNIT_ASSERT_DOUBLES_EQUAL(PairVectorSimilarityMetric(lhs, rhs, 2), 1.0, 1e-7);
        }
    }

}
