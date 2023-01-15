#include <catboost/private/libs/text_features/bm25.h>
#include <catboost/private/libs/text_features/naive_bayesian.h>
#include <catboost/private/libs/text_features/embedding_online_features.h>

#include <library/cpp/unittest/registar.h>
#include <util/random/fast.h>

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

    Y_UNIT_TEST(TestEmbeddingFeaturesCanonization) {
        const ui32 numSamples = 10;
        const ui32 dictionarySize = 30;
        TFastRng<ui64> rng(42);

        TVector<TText> texts = CreateRandomTextVector(rng, dictionarySize, numSamples);
        TText testText = CreateRandomText(rng, dictionarySize, 10);

        TEmbeddingPtr embeddingPtr;
        const ui32 embeddingDim = dictionarySize;
        {
            TDenseHash<TTokenId, TVector<float>> hashVectors;
            for (ui32 tokenId: xrange(dictionarySize)) {
                TVector<float> vector(embeddingDim);
                vector[tokenId] = 1.0;
                hashVectors[TTokenId(tokenId)] = vector;
            }
            embeddingPtr = CreateEmbedding(std::move(hashVectors));
        }

        THashMap<ui32, TVector<float>> scores;

        for (ui32 numClasses : {2, 6, 10}) {
            TEmbeddingOnlineFeatures embeddingOnlineFeatures(CreateGuid(), numClasses, embeddingPtr);
            TEmbeddingFeaturesVisitor visitor(numClasses, embeddingDim);
            for (ui32 sampleId: xrange(numSamples)) {
                visitor.Update(sampleId % numClasses, texts[sampleId], &embeddingOnlineFeatures);
            }

            scores[numClasses] = embeddingOnlineFeatures.TTextFeatureCalcer::Compute(testText);
        }

        const double epsilon = 1e-6;
        Y_UNUSED(epsilon);
        {
            const ui32 numClasses = 2;
            const auto& caseScores = scores[numClasses];

            // CosDistance
            UNIT_ASSERT_DOUBLES_EQUAL(0.7962321650, caseScores[0], epsilon);
            UNIT_ASSERT_DOUBLES_EQUAL(0.7259065113, caseScores[1], epsilon);

            // LDA features
            UNIT_ASSERT_DOUBLES_EQUAL(0.4930330234, caseScores[2], epsilon);
            UNIT_ASSERT_DOUBLES_EQUAL(0.4998014690, caseScores[3], epsilon);

            UNIT_ASSERT_DOUBLES_EQUAL(0.5069669766, caseScores[4], epsilon);
            UNIT_ASSERT_DOUBLES_EQUAL(0.5001985310, caseScores[5], epsilon);
        }

        {
            const ui32 numClasses = 6;
            const auto& caseScores = scores[numClasses];

            // CosDistance
            UNIT_ASSERT_DOUBLES_EQUAL(0.8384363095, caseScores[0], epsilon);
            UNIT_ASSERT_DOUBLES_EQUAL(0.6407144896, caseScores[2], epsilon);
            UNIT_ASSERT_DOUBLES_EQUAL(0.5457051607, caseScores[5], epsilon);

            // LDA features
            UNIT_ASSERT_DOUBLES_EQUAL(0.1754046068, caseScores[6], epsilon);
            UNIT_ASSERT_DOUBLES_EQUAL(0.1865056705, caseScores[7], epsilon);
            UNIT_ASSERT_DOUBLES_EQUAL(0.1892145501, caseScores[10], epsilon);

            UNIT_ASSERT_DOUBLES_EQUAL(0.1856485121, caseScores[12], epsilon);
            UNIT_ASSERT_DOUBLES_EQUAL(0.1315803611, caseScores[14], epsilon);
            UNIT_ASSERT_DOUBLES_EQUAL(0.1252915611, caseScores[17], epsilon);
        }

        {
            const ui32 numClasses = 10;
            const auto& caseScores = scores[numClasses];

            // CosDistance
            UNIT_ASSERT_DOUBLES_EQUAL(0.764705887, caseScores[0], epsilon);
            UNIT_ASSERT_DOUBLES_EQUAL(0.6507913778, caseScores[2], epsilon);
            UNIT_ASSERT_DOUBLES_EQUAL(0.7399400782, caseScores[6], epsilon);
            UNIT_ASSERT_DOUBLES_EQUAL(0.5009794371, caseScores[7], epsilon);
            UNIT_ASSERT_DOUBLES_EQUAL(0.7058823575, caseScores[9], epsilon);

            // LDA features
            UNIT_ASSERT_DOUBLES_EQUAL(0.09541972041, caseScores[10], epsilon);
            UNIT_ASSERT_DOUBLES_EQUAL(0.09987460737, caseScores[13], epsilon);
            UNIT_ASSERT_DOUBLES_EQUAL(0.09977643617, caseScores[15], epsilon);
            UNIT_ASSERT_DOUBLES_EQUAL(0.10143683600, caseScores[16], epsilon);
            UNIT_ASSERT_DOUBLES_EQUAL(0.10034430050, caseScores[19], epsilon);

            UNIT_ASSERT_DOUBLES_EQUAL(0.10158981290, caseScores[20], epsilon);
            UNIT_ASSERT_DOUBLES_EQUAL(0.10304506590, caseScores[24], epsilon);
            UNIT_ASSERT_DOUBLES_EQUAL(0.10493913550, caseScores[26], epsilon);
            UNIT_ASSERT_DOUBLES_EQUAL(0.10052911200, caseScores[27], epsilon);
            UNIT_ASSERT_DOUBLES_EQUAL(0.09968623802, caseScores[29], epsilon);
        }
    }
}
