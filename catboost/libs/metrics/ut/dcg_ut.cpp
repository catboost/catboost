#include <catboost/libs/metrics/dcg.h>
#include <catboost/libs/metrics/metric.h>
#include <catboost/libs/metrics/sample.h>

#include <library/cpp/testing/unittest/registar.h>

#include <util/stream/labeled.h>
#include <util/generic/algorithm.h>
#include <util/generic/array_ref.h>
#include <util/generic/vector.h>
#include <util/random/fast.h>

Y_UNIT_TEST_SUITE(NdcgTests) {
    Y_UNIT_TEST(TestNdcgDetails) {
        {
            TVector<double> approx{1.0, 0.0, 2.0};
            TVector<double> target{1.0, 0.0, 2.0};
            const auto samples = NMetrics::TSample::FromVectors(target, approx);
            TVector<double> decay(samples.size());
            FillDcgDecay(ENdcgDenominatorType::LogPosition, Nothing(), decay);
            UNIT_ASSERT_VALUES_EQUAL(CalcNdcg(samples, decay, ENdcgMetricType::Base), 1);
            UNIT_ASSERT_VALUES_EQUAL(CalcNdcg(samples, decay, ENdcgMetricType::Exp), 1);
        }
        {
            TVector<double> approx{1.0, 1.0, 2.0};
            TVector<double> target{1.0, 0.0, 2.0};
            const auto samples = NMetrics::TSample::FromVectors(target, approx);
            TVector<double> decay(samples.size());
            FillDcgDecay(ENdcgDenominatorType::LogPosition, Nothing(), decay);
            UNIT_ASSERT_DOUBLES_EQUAL(CalcNdcg(samples, decay, ENdcgMetricType::Base), 0.9502344168, 1e-5);
            UNIT_ASSERT_DOUBLES_EQUAL(CalcNdcg(samples, decay, ENdcgMetricType::Exp), 0.9639404333, 1e-5);
        }
    }

    Y_UNIT_TEST(TestDcgDetails) {
        {
            TVector<double> approx{1.0, 0.0, 2.0};
            TVector<double> target{1.0, 0.0, 2.0};
            const auto samples = NMetrics::TSample::FromVectors(target, approx);
            UNIT_ASSERT_DOUBLES_EQUAL(CalcDcg(samples, ENdcgMetricType::Base), 2.6309297535714573, 1e-5);
            UNIT_ASSERT_DOUBLES_EQUAL(CalcDcg(samples, ENdcgMetricType::Exp), 3.6309297535714573, 1e-5);
        }
        {
            TVector<double> approx{1.0, 1.0, 2.0};
            TVector<double> target{1.0, 0.0, 2.0};
            const auto samples = NMetrics::TSample::FromVectors(target, approx);
            UNIT_ASSERT_DOUBLES_EQUAL(CalcDcg(samples, ENdcgMetricType::Base), 2.5, 1e-5);
            UNIT_ASSERT_DOUBLES_EQUAL(CalcDcg(samples, ENdcgMetricType::Exp), 3.5, 1e-5);
        }
    }

    static void TestNdcg(
        const TString metricDescription,
        const size_t totalDocumentCount,
        const size_t maxPerQueryDocumentCount,
        const float targetScale,
        const float weightScale,
        const ui64 seed,
        const double eps,
        const TMetricHolder expected)
    {
        TVector<TQueryInfo> queryInfos;
        TVector<float> targets;
        targets.yresize(totalDocumentCount);
        TVector<TVector<double>> approxes(1);
        approxes.front().yresize(totalDocumentCount);
        TVector<float> dummyWeights;

        TFastRng<ui64> prng(seed);
        queryInfos.reserve(totalDocumentCount);
        for (size_t documentCount = 0; documentCount < totalDocumentCount; documentCount += (queryInfos.back().End - queryInfos.back().Begin)) {
            const auto size = prng.Uniform(Min<ui32>(maxPerQueryDocumentCount, totalDocumentCount - documentCount)) + 1;

            TQueryInfo info(documentCount, documentCount + size);
            info.Weight = prng.GenRandReal1() * weightScale;
            queryInfos.emplace_back(std::move(info));
        }

        ForEach(targets.begin(), targets.end(), [&](auto& v) { v = prng.GenRandReal1() * targetScale; });
        ForEach(
            approxes.front().begin(), approxes.front().end(),
            [&](auto& v) { v = prng.GenRandReal1() * targetScale; });

        NPar::TLocalExecutor executor;
        const auto ndcg = std::move(CreateMetricsFromDescription({metricDescription}, 1).front());
        const auto metric = dynamic_cast<const ISingleTargetEval*>(ndcg.Get())->Eval(
            approxes,
            targets,
            dummyWeights,
            queryInfos,
            0,
            queryInfos.size(),
            executor);

        UNIT_ASSERT_VALUES_EQUAL_C(
            expected.Stats.size(), metric.Stats.size(),
            LabeledOutput(
                metricDescription,
                totalDocumentCount,
                targetScale,
                weightScale,
                seed));
        UNIT_ASSERT_VALUES_EQUAL_C(
            expected.Stats.size(), 2,
            LabeledOutput(
                metricDescription,
                totalDocumentCount,
                targetScale,
                weightScale,
                seed));
        UNIT_ASSERT_DOUBLES_EQUAL_C(
            expected.Stats[0], metric.Stats[0], eps,
            LabeledOutput(
                metricDescription,
                totalDocumentCount,
                targetScale,
                weightScale,
                seed));
        UNIT_ASSERT_DOUBLES_EQUAL_C(
            expected.Stats[1], metric.Stats[1], eps,
            LabeledOutput(
                metricDescription,
                totalDocumentCount,
                targetScale,
                weightScale,
                seed));
    }

    Y_UNIT_TEST(TestDcg1) {
        TMetricHolder expected(2);
        expected.Stats[0] = 11.869085255821183;
        expected.Stats[1] = 2.5384541749954224;
        TestNdcg("DCG:top=3;type=Base", 10, 3, 5, 1, 20181129, 1e-6, expected);
    }

    Y_UNIT_TEST(TestDcg2) {
        TMetricHolder expected(2);
        expected.Stats[0] = 18.669697535999937;
        expected.Stats[1] = 4;
        TestNdcg("DCG:top=3;type=Base;use_weights=false", 10, 3, 5, 1, 20181129, 1e-6, expected);
    }

    Y_UNIT_TEST(TestDcg3) {
        TMetricHolder expected(2);
        expected.Stats[0] = 42.705168163665576;
        expected.Stats[1] = 2.5384541749954224;
        TestNdcg("DCG:top=3;type=Exp", 10, 3, 5, 1, 20181129, 1e-6, expected);
    }

    Y_UNIT_TEST(TestDcg4) {
        TMetricHolder expected(2);
        expected.Stats[0] = 68.80169093445322;
        expected.Stats[1] = 4;
        TestNdcg("DCG:top=3;type=Exp;use_weights=false", 10, 3, 5, 1, 20181129, 1e-6, expected);
    }

    Y_UNIT_TEST(TestDcgPos1) {
        TMetricHolder expected(2);
        expected.Stats[0] = 10.82153538353361;
        expected.Stats[1] = 2.5384541749954224;
        TestNdcg("DCG:top=3;type=Base;denominator=Position", 10, 3, 5, 1, 20181129, 1e-6, expected);
    }

    Y_UNIT_TEST(TestDcgPos2) {
        TMetricHolder expected(2);
        expected.Stats[0] = 16.924527952273333;
        expected.Stats[1] = 4;
        TestNdcg("DCG:top=3;type=Base;use_weights=false;denominator=Position", 10, 3, 5, 1, 20181129, 1e-6, expected);
    }

    Y_UNIT_TEST(TestDcgPos3) {
        TMetricHolder expected(2);
        expected.Stats[0] = 39.65690359032617;
        expected.Stats[1] = 2.5384541749954224;
        TestNdcg("DCG:top=3;type=Exp;denominator=Position", 10, 3, 5, 1, 20181129, 1e-6, expected);
    }

    Y_UNIT_TEST(TestDcgPos4) {
        TMetricHolder expected(2);
        expected.Stats[0] = 63.43172290133928;
        expected.Stats[1] = 4;
        TestNdcg("DCG:top=3;type=Exp;use_weights=false;denominator=Position", 10, 3, 5, 1, 20181129, 1e-6, expected);
    }

    Y_UNIT_TEST(TestNdcg1) {
        TMetricHolder expected(2);
        expected.Stats[0] = 2.3522257049940407;
        expected.Stats[1] = 2.5384541749954224;
        TestNdcg("NDCG:top=3;type=Base", 10, 3, 5, 1, 20181129, 1e-6, expected);
    }

    Y_UNIT_TEST(TestNdcg2) {
        TMetricHolder expected(2);
        expected.Stats[0] = 3.662129720738318;
        expected.Stats[1] = 4;
        TestNdcg("NDCG:top=3;type=Base;use_weights=false", 10, 3, 5, 1, 20181129, 1e-6, expected);
    }

    Y_UNIT_TEST(TestNdcg3) {
        TMetricHolder expected(2);
        expected.Stats[0] = 2.318911199842912;
        expected.Stats[1] = 2.5384541749954224;
        TestNdcg("NDCG:top=3;type=Exp", 10, 3, 5, 1, 20181129, 1e-6, expected);
    }

    Y_UNIT_TEST(TestNdcg4) {
        TMetricHolder expected(2);
        expected.Stats[0] = 3.6016879356622065;
        expected.Stats[1] = 4;
        TestNdcg("NDCG:top=3;type=Exp;use_weights=false", 10, 3, 5, 1, 20181129, 1e-6, expected);
    }

    Y_UNIT_TEST(TestNdcgPos1) {
        TMetricHolder expected(2);
        expected.Stats[0] = 2.272529107367333;
        expected.Stats[1] = 2.5384541749954224;
        TestNdcg("NDCG:top=3;type=Base;denominator=Position", 10, 3, 5, 1, 20181129, 1e-6, expected);
    }

    Y_UNIT_TEST(TestNdcgPos2) {
        TMetricHolder expected(2);
        expected.Stats[0] = 3.5175379099202693;
        expected.Stats[1] = 4;
        TestNdcg("NDCG:top=3;type=Base;use_weights=false;denominator=Position", 10, 3, 5, 1, 20181129, 1e-6, expected);
    }

    Y_UNIT_TEST(TestNdcgPos3) {
        TMetricHolder expected(2);
        expected.Stats[0] = 2.23288740341198;
        expected.Stats[1] = 2.5384541749954224;
        TestNdcg("NDCG:top=3;type=Exp;denominator=Position", 10, 3, 5, 1, 20181129, 1e-6, expected);
    }

    Y_UNIT_TEST(TestNdcgPos4) {
        TMetricHolder expected(2);
        expected.Stats[0] = 3.4456168251015917;
        expected.Stats[1] = 4;
        TestNdcg("NDCG:top=3;type=Exp;use_weights=false;denominator=Position", 10, 3, 5, 1, 20181129, 1e-6, expected);
    }

    Y_UNIT_TEST(TestNdcg5) {
        TMetricHolder expected(2);
        expected.Stats[0] = 19738.908943893228;
        expected.Stats[1] = 32236.35508507835;
        TestNdcg("NDCG:top=3;type=Base", 1000000, 30, 5, 1, 20181129, 1e-6, expected);
    }

    Y_UNIT_TEST(TestNdcg6) {
        TMetricHolder expected(2);
        expected.Stats[0] = 39451.05458691906;
        expected.Stats[1] = 64428;
        TestNdcg("NDCG:top=3;type=Base;use_weights=false", 1000000, 30, 5, 1, 20181129, 1e-6, expected);
    }

    Y_UNIT_TEST(TestNdcg7) {
        TMetricHolder expected(2);
        expected.Stats[0] = 14114.655296797971;
        expected.Stats[1] = 32236.35508507835;
        TestNdcg("NDCG:top=3;type=Exp", 1000000, 30, 5, 1, 20181129, 1e-6, expected);
    }

    Y_UNIT_TEST(TestNdcg8) {
        TMetricHolder expected(2);
        expected.Stats[0] = 28194.27048448048;
        expected.Stats[1] = 64428;
        TestNdcg("NDCG:top=3;type=Exp;use_weights=false", 1000000, 30, 5, 1, 20181129, 1e-6, expected);
    }

    Y_UNIT_TEST(TestNdcg9) {
        TMetricHolder expected(2);
        expected.Stats[0] = 79170.37766674075;
        expected.Stats[1] = 90833.32908555106;
        TestNdcg("NDCG:top=15;type=Base", 1000000, 10, 5, 1, 20181129, 1e-6, expected);
    }

    Y_UNIT_TEST(TestNdcg10) {
        TMetricHolder expected(2);
        expected.Stats[0] = 158283.11128760662;
        expected.Stats[1] = 181600;
        TestNdcg("NDCG:top=15;type=Base;use_weights=false", 1000000, 10, 5, 1, 20181129, 1e-6, expected);
    }

    Y_UNIT_TEST(TestNdcg11) {
        TMetricHolder expected(2);
        expected.Stats[0] = 71849.62196691956;
        expected.Stats[1] = 90833.32908555106;
        TestNdcg("NDCG:top=15;type=Exp", 1000000, 10, 5, 1, 20181129, 1e-6, expected);
    }

    Y_UNIT_TEST(TestNdcg12) {
        TMetricHolder expected(2);
        expected.Stats[0] = 143625.41865727352;
        expected.Stats[1] = 181600;
        TestNdcg("NDCG:top=15;type=Exp;use_weights=false", 1000000, 10, 5, 1, 20181129, 1e-6, expected);
    }
}
