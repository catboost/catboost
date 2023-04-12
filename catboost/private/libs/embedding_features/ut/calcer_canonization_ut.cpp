#include <catboost/private/libs/embedding_features/lda.h>

#include <library/cpp/testing/unittest/registar.h>
#include <util/random/fast.h>
#include <util/random/normal.h>
#include <util/random/random.h>
#include <util/generic/ymath.h>

using namespace NCB;

Y_UNIT_TEST_SUITE(TestCalcerCanonization) {

    TEmbeddingsArray NormalEmbedding(TVector<float> mean) {
        for (auto& cord : mean) {
            cord += StdNormalRandom<float>();
        }
        return TMaybeOwningArrayHolder<const float>::CreateOwning(std::move(mean));
    }

    TVector<TEmbeddingsArray> DataSetGenerator(TVector<TVector<float>> means, TVector<ui32> target) {
        TFastRng<ui32> rng(42);
        TVector<TEmbeddingsArray> result;
        for (auto idx : target) {
            result.push_back(NormalEmbedding(means[idx]));
        }
        return result;
    }

    Y_UNIT_TEST(TestEmbeddingsUtilities) {
        TVector<float> matrix{5, 2, 2, 1};
        TVector<float> res{1, -2, -2, 5};
        InverseMatrix(&matrix, 2);
        float norm = 0;
        for (int i = 0; i < 4; ++i) {
            norm += (res[i] - matrix[i]) * (res[i] - matrix[i]);
        }
        UNIT_ASSERT_DOUBLES_EQUAL(norm, 0.0, 1e-7);

        TVector<float> mean{1, 1};
        auto emb1 = TMaybeOwningArrayHolder<const float>::CreateOwning(TVector<float>{0,0});
        auto emb2 = TMaybeOwningArrayHolder<const float>::CreateOwning(TVector<float>{0,1});
        float ratio = CalculateGaussianLikehood(emb1, mean, matrix)/CalculateGaussianLikehood(emb2, mean, matrix);
        UNIT_ASSERT_DOUBLES_EQUAL(ratio, 0.6065, 1e-2);
    }

    Y_UNIT_TEST(TestLDACanonization) {
        const ui32 numSamples = 50000;

        TVector<TVector<float>> means{{13, 15, 2}, {-28, -13, 15}, {15, -5, -20}};
        TVector<ui32> target;
        for (ui32 id = 0; id < numSamples; ++id) {
            target.push_back(RandomNumber<ui32>(means.size() - 1));
        }

        auto dataSet = DataSetGenerator(means, target);
        ui32 numClasses = means.size() - 1;
        ui32 dim = means[0].size();

        TLinearDACalcer testLDA(dim, means.size(), numClasses);
        TLinearDACalcerVisitor visitor;

        for (ui32 idx = 0; idx < target.size(); ++idx) {
            visitor.Update(target[idx], dataSet[idx], &testLDA);
        }

        TVector<float> proj(dim * numClasses);
        auto it = proj.begin();
        for (ui32 id = 0; id < dim; ++id) {
            TVector<float> emb(dim, 0.0);
            emb[id] = 1.0;
            auto embedding = TMaybeOwningArrayHolder<const float>::CreateOwning(std::move(emb));
            TOutputFloatIterator out(it, numClasses);
            testLDA.Compute(embedding, out);
            it += numClasses;
        }
        const float eps = 0.05;
        float norm1 = sqrt(proj[0] * proj[0] + proj[2] * proj[2] + proj[4] * proj[4]);
        float norm2 = sqrt(proj[1] * proj[1] + proj[3] * proj[3] + proj[5] * proj[5]);

        UNIT_ASSERT_DOUBLES_EQUAL(proj[0] - proj[2] + proj[4], 0.0, eps * norm1);
        UNIT_ASSERT_DOUBLES_EQUAL(proj[1] - proj[3] + proj[5], 0.0, eps * norm2);
    }
}
