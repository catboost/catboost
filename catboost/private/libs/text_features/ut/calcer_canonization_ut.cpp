#include <catboost/private/libs/text_features/bm25.h>
#include <catboost/private/libs/text_features/naive_bayesian.h>

#include <library/cpp/testing/unittest/registar.h>
#include <util/random/fast.h>
#include <util/generic/xrange.h>

using namespace NCB;

Y_UNIT_TEST_SUITE(TestFeatureCalcerCanonization) {
    TText CreateRandomText(TFastRng<ui64>& rng, ui32 dictionarySize, ui32 scale=100) {
        TMap<TTokenId, TText::TCountType> tokenToCount;
        for (ui32 tokenId: xrange(dictionarySize)) {
            double prob = rng.GenRandReal1();
            if (prob > 0.5) {
                tokenToCount[tokenId] = static_cast<ui32>(prob * scale);
            }
        }
        return TText(std::move(tokenToCount));
    }

    TVector<TText> CreateRandomTextVector(TFastRng<ui64>& rng, ui32 dictionarySize, ui32 size, ui32 scale=100) {
        TVector<TText> result;
        for (ui32 sampleId : xrange(size)) {
            Y_UNUSED(sampleId);
            result.push_back(CreateRandomText(rng, dictionarySize, scale));
        }
        return result;
    }

    Y_UNIT_TEST(TestBm25Canonization) {
        const ui32 numSamples = 10;
        const ui32 dictionarySize = 30;
        TFastRng<ui64> rng(42);

        TVector<TText> texts = CreateRandomTextVector(rng, dictionarySize, numSamples);
        TText testText = CreateRandomText(rng, dictionarySize);

        const double truncateBorder = 1e-3;
        THashMap<ui32, TVector<float>> scores;

        for (ui32 numClasses : {2, 6, 10}) {
            TBM25 bm25(CreateGuid(), numClasses, truncateBorder);
            TBM25Visitor visitor;
            for (ui32 sampleId: xrange(numSamples)) {
                visitor.Update(sampleId % numClasses, texts[sampleId], &bm25);
            }

            scores[numClasses] = bm25.TTextFeatureCalcer::Compute(testText);
        }

        const double epsilon = 1e-7;
        {
            const ui32 numClasses = 2;
            const auto& caseScores = scores[numClasses];

            UNIT_ASSERT_DOUBLES_EQUAL(0.04216079265, caseScores[0], epsilon);
            UNIT_ASSERT_DOUBLES_EQUAL(0.03969458635, caseScores[1], epsilon);
        }

        {
            const ui32 numClasses = 6;
            const auto& caseScores = scores[numClasses];

            UNIT_ASSERT_DOUBLES_EQUAL(0.03936469741, caseScores[0], epsilon);
            UNIT_ASSERT_DOUBLES_EQUAL(0.03447869110, caseScores[1], epsilon);
            UNIT_ASSERT_DOUBLES_EQUAL(0.03701182930, caseScores[2], epsilon);
            UNIT_ASSERT_DOUBLES_EQUAL(0.03216326646, caseScores[3], epsilon);
            UNIT_ASSERT_DOUBLES_EQUAL(0.01930596789, caseScores[4], epsilon);
            UNIT_ASSERT_DOUBLES_EQUAL(0.02180865015, caseScores[5], epsilon);
        }

        {
            const ui32 numClasses = 10;
            const auto& caseScores = scores[numClasses];

            UNIT_ASSERT_DOUBLES_EQUAL(1.8240569360, caseScores[0], epsilon);
            UNIT_ASSERT_DOUBLES_EQUAL(0.9259216262, caseScores[1], epsilon);
            UNIT_ASSERT_DOUBLES_EQUAL(1.8366250600, caseScores[2], epsilon);
            UNIT_ASSERT_DOUBLES_EQUAL(1.8302790500, caseScores[3], epsilon);
            UNIT_ASSERT_DOUBLES_EQUAL(1.8101622080, caseScores[4], epsilon);
            UNIT_ASSERT_DOUBLES_EQUAL(2.7758485690, caseScores[5], epsilon);
            UNIT_ASSERT_DOUBLES_EQUAL(2.7796200730, caseScores[6], epsilon);
            UNIT_ASSERT_DOUBLES_EQUAL(0.0195988582, caseScores[7], epsilon);
            UNIT_ASSERT_DOUBLES_EQUAL(1.8829803690, caseScores[8], epsilon);
            UNIT_ASSERT_DOUBLES_EQUAL(0.9249947509, caseScores[9], epsilon);
        }
    }

    Y_UNIT_TEST(TestNaiveBayesCanonization) {
        const ui32 numSamples = 100;
        const ui32 dictionarySize = 30;
        TFastRng<ui64> rng(42);

        TVector<TText> texts = CreateRandomTextVector(rng, dictionarySize, numSamples);
        TText testText = CreateRandomText(rng, dictionarySize, 10);

        THashMap<ui32, TVector<float>> scores;

        for (ui32 numClasses : {2, 6, 10}) {
            TMultinomialNaiveBayes naiveBayes(CreateGuid(), numClasses);
            TNaiveBayesVisitor visitor;
            for (ui32 sampleId: xrange(numSamples)) {
                visitor.Update(sampleId % numClasses, texts[sampleId], &naiveBayes);
            }

            scores[numClasses] = naiveBayes.TTextFeatureCalcer::Compute(testText);
        }

        const double epsilon = 1e-7;
        {
            const ui32 numClasses = 2;
            const auto& caseScores = scores[numClasses];

            UNIT_ASSERT_DOUBLES_EQUAL(0.08221943676, caseScores[0], epsilon);
        }

        {
            const ui32 numClasses = 6;
            const auto& caseScores = scores[numClasses];

            UNIT_ASSERT_DOUBLES_EQUAL(0.115891076600, caseScores[0], epsilon);
            UNIT_ASSERT_DOUBLES_EQUAL(1.81478697e-06, caseScores[1], epsilon);
            UNIT_ASSERT_DOUBLES_EQUAL(9.96617018e-06, caseScores[2], epsilon);
            UNIT_ASSERT_DOUBLES_EQUAL(0.883523583400, caseScores[3], epsilon);
            UNIT_ASSERT_DOUBLES_EQUAL(4.33121471e-07, caseScores[4], epsilon);
            UNIT_ASSERT_DOUBLES_EQUAL(0.000573132420, caseScores[5], epsilon);
        }

        {
            const ui32 numClasses = 10;
            const auto& caseScores = scores[numClasses];

            UNIT_ASSERT_DOUBLES_EQUAL(9.481456800e-06, caseScores[0], epsilon);
            UNIT_ASSERT_DOUBLES_EQUAL(1.108779179e-05, caseScores[1], epsilon);
            UNIT_ASSERT_DOUBLES_EQUAL(4.613123394e-07, caseScores[2], epsilon);
            UNIT_ASSERT_DOUBLES_EQUAL(0.2804053724000, caseScores[3], epsilon);
            UNIT_ASSERT_DOUBLES_EQUAL(2.468921734e-10, caseScores[4], epsilon);
            UNIT_ASSERT_DOUBLES_EQUAL(5.559921235e-08, caseScores[5], epsilon);
            UNIT_ASSERT_DOUBLES_EQUAL(2.756301480e-12, caseScores[6], epsilon);
            UNIT_ASSERT_DOUBLES_EQUAL(7.357047252e-06, caseScores[7], epsilon);
            UNIT_ASSERT_DOUBLES_EQUAL(0.7195571065000, caseScores[8], epsilon);
            UNIT_ASSERT_DOUBLES_EQUAL(9.068413419e-06, caseScores[9], epsilon);
        }
    }

}
