#include <catboost/private/libs/text_features/feature_calcer.h>
#include <catboost/private/libs/text_features/text_feature_calcers.h>

#include <catboost/private/libs/text_features/helpers.h>

#include <library/unittest/registar.h>
#include <util/generic/ylimits.h>

using namespace NCB;

Y_UNIT_TEST_SUITE(TestFeatureCalcer) {
    Y_UNIT_TEST(TestOutputFeaturesIterator) {
        {
            const ui32 arraySize = 9;
            const ui32 step = 2;

            TArrayHolder<float> array = new float[arraySize];
            Iota(array.Get(), array.Get() + arraySize, 0);

            TOutputFloatIterator iterator(array.Get() + 1, step, arraySize - 1);

            UNIT_ASSERT(iterator.IsValid());
            UNIT_ASSERT_EQUAL(1, *iterator);
            ++iterator;

            UNIT_ASSERT(iterator.IsValid());
            UNIT_ASSERT_EQUAL(3, *iterator);
            ++iterator;

            UNIT_ASSERT(iterator.IsValid());
            UNIT_ASSERT_EQUAL(5, *iterator);
            ++iterator;

            UNIT_ASSERT(iterator.IsValid());
            UNIT_ASSERT_EQUAL(7, *iterator);
            ++iterator;

            UNIT_ASSERT(!iterator.IsValid());
        }

        {
            const ui32 arraySize = 100;
            TArrayHolder<float> array = new float[arraySize];
            Iota(array.Get(), array.Get() + arraySize, 0);
            TOutputFloatIterator iterator(array.Get(), arraySize);

            for (ui32 i = 0; i < arraySize; i++, ++iterator) {
                UNIT_ASSERT(iterator.IsValid());
                UNIT_ASSERT_EQUAL(*iterator, i);
            }
            UNIT_ASSERT(!iterator.IsValid());
        }
    }

    Y_UNIT_TEST(TestBagOfWordsCalcer) {
        const ui32 numTokens = 27;
        TTextFeatureCalcerPtr calcer = MakeIntrusive<TBagOfWordsCalcer>(numTokens);

        const ui32 numSamples = 327;
        const ui32 numTokensPerText = 11;

        TVector<TText> texts;
        TVector<float> features(numSamples * numTokens);

        for (ui32 docId : xrange(numSamples)) {
            TText text;
            for (ui32 idx : xrange(numTokensPerText)) {
                ui32 tokenId = (docId + idx) % numTokens;
                text[tokenId] = 1;
            }
            texts.push_back(text);
            calcer->Compute(text, TOutputFloatIterator(features.data() + docId, numSamples, features.size()));
        }

        for (ui32 docId : xrange(numSamples)) {
            for (ui32 idx : xrange(numTokens)) {
                UNIT_ASSERT(
                    features[docId + idx * numSamples] ==
                    static_cast<float>(texts[docId].Has(idx))
                );
            }
        }
    }

    static void Log(TArrayRef<double> values) {
        for (double& value: values) {
            value = log(value);
        }
    }

    Y_UNIT_TEST(TestNaiveBayes) {
        const float epsilon = 1e-6;

        { // TestNoPriorInformation
            TText text;
            text.insert({/*tokenId*/ 0, /*count*/ 1});

            for (const ui32 numClasses: {2, 5, 10, 200}) {
                TMultinomialNaiveBayes naiveBayes(numClasses);

                TVector<float> features = naiveBayes.TTextFeatureCalcer::Compute(text);
                UNIT_ASSERT_EQUAL(features.size(), numClasses == 2 ? 1 : numClasses);

                for (ui32 classIdx : xrange(features.size())) {
                    UNIT_ASSERT_DOUBLES_EQUAL(1. / float(numClasses), features[classIdx], epsilon);
                }
            }
        }

        { // TestZeroPrior
            const float classPrior = 0;
            const float tokenPrior = 0;
            const ui32 numClasses = 4;
            const ui32 dictionarySize = 7;

            TMultinomialNaiveBayes naiveBayes(numClasses, classPrior, tokenPrior);
            TNaiveBayesVisitor bayesVisitor;
            for (ui32 classIdx: xrange(numClasses)) {
                TText text;
                for (const ui32 tokenId : xrange(dictionarySize)) {
                    text.insert({tokenId, /*count*/ 1});
                }
                bayesVisitor.Update(classIdx, text, &naiveBayes);
            }

            const TVector<THashMap<ui32, ui32>> classTokenToCount = {
                {{/*tokenId*/ 0, /*count*/ 7}},
                {{/*tokenId*/ 1, /*count*/ 3}},
                {{/*tokenId*/ 2, /*count*/ 4}, {/*tokenId*/ 6, /*count*/ 2}},
                {{/*tokenId*/ 3, /*count*/ 9}, {/*tokenId*/ 5, /*count*/ 1}},
            };

            for (ui32 classIdx : xrange(numClasses)) {
                TText text;
                for (const auto& [tokenId, count]: classTokenToCount[classIdx]) {
                    text.insert({tokenId, count});
                }
                bayesVisitor.Update(classIdx, text, &naiveBayes);
            }

            for (ui32 tokenId : xrange(dictionarySize)) {
                TText text;
                text.insert({tokenId, /*count*/ 1});

                TVector<float> features = naiveBayes.TTextFeatureCalcer::Compute(text);
                TVector<double> classLogProbs(numClasses);

                for (ui32 classIdx: xrange(numClasses)) {
                    ui32 classTokensCount = dictionarySize;
                    for (const auto& [tokId, count]: classTokenToCount[classIdx]) {
                        Y_UNUSED(tokId);
                        classTokensCount += count;
                    }

                    if (classIdx < features.size()) {
                        const ui32 classSamples = 2;
                        const ui32 tokenCount = classTokenToCount[classIdx].Value(tokenId, 0) + 1;
                        classLogProbs[classIdx] = log(classSamples * tokenCount / float(classTokensCount));
                    }
                }

                Softmax(classLogProbs);

                for (ui32 featureIdx : xrange(naiveBayes.FeatureCount())) {
                    UNIT_ASSERT_DOUBLES_EQUAL(classLogProbs[featureIdx], features[featureIdx], epsilon);
                }
            }
        }

        { // TestRepeatingTokens
            const float classPrior = 0;
            const float tokenPrior = 0;
            const ui32 numClasses = 2;

            TMultinomialNaiveBayes naiveBayes(numClasses, classPrior, tokenPrior);
            TNaiveBayesVisitor bayesVisitor;

            {
                TText text1;
                text1.insert({/*tokenId*/ 0, /*count*/ 2});
                text1.insert({/*tokenId*/ 1, /*count*/ 3});

                TText text2;
                text2.insert({/*tokenId*/ 0, /*count*/ 4});
                text2.insert({/*tokenId*/ 1, /*count*/ 1});

                bayesVisitor.Update(0, text1, &naiveBayes);
                bayesVisitor.Update(1, text2, &naiveBayes);
            }

            TVector<double> token0Prob = {2./5., 4./5.};
            TVector<double> token1Prob = {3./5., 1./5.};

            Log(token0Prob);
            Log(token1Prob);

            TVector<double> norm0Prob(token0Prob);
            TVector<double> norm1Prob(token1Prob);
            Softmax(norm0Prob);
            Softmax(norm1Prob);

            TText text0;
            text0.insert({0, 1});

            TText text1;
            text1.insert({1, 1});

            TVector<float> features = naiveBayes.TTextFeatureCalcer::Compute(text0);
            UNIT_ASSERT_DOUBLES_EQUAL(norm0Prob[0], features[0], epsilon);

            features = naiveBayes.TTextFeatureCalcer::Compute(text1);
            UNIT_ASSERT_DOUBLES_EQUAL(norm1Prob[0], features[0], epsilon);

            TText textRepeated0;
            textRepeated0.insert({0, 4});

            TVector<double> repeatedToken0Probs = {
                token0Prob[0] * 4,
                token0Prob[1] * 4
            };
            Softmax(repeatedToken0Probs);

            features = naiveBayes.TTextFeatureCalcer::Compute(textRepeated0);
            UNIT_ASSERT_DOUBLES_EQUAL(
                repeatedToken0Probs[0],
                features[0],
                epsilon
            );

            TText textRepeated1;
            textRepeated1.insert({1, 3});

            TVector<double> repeatedToken1Probs = {
                token1Prob[0] * 3,
                token1Prob[1] * 3
            };
            Softmax(repeatedToken1Probs);

            features = naiveBayes.TTextFeatureCalcer::Compute(textRepeated1);
            UNIT_ASSERT_DOUBLES_EQUAL(
                repeatedToken1Probs[0],
                features[0],
                epsilon
            );
        }
    }
}
