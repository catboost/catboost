#include "embedding_online_features.h"
#include "helpers.h"

#include <cmath>
#include <contrib/libs/clapack/clapack.h>

static inline void SolveLinearSystemCholesky(TArrayRef<double> matrix,
                                             TArrayRef<double> vec) {

    char matrixStorageType[] = {'U', '\0'};
    int systemSize = static_cast<int>(vec.size());
    int numberOfRightHandSides = 1;

    int info = 0;
    dposv_(matrixStorageType, &systemSize, &numberOfRightHandSides, matrix.data(), &systemSize,
           vec.data(), &systemSize, &info);

    Y_VERIFY(info >= 0);
}


inline double LogProbNormal(TConstArrayRef<float> x, TConstArrayRef<double> mu, TConstArrayRef<double> sigma) {
    const ui32 dim = x.size();

    TVector<double> target(x.begin(), x.end());
    for (ui32 i = 0; i < dim; ++i) {
        target[i] -= mu[i];
    }
    TVector<double> sigmaCopy(sigma.begin(), sigma.end());
    SolveLinearSystemCholesky(sigmaCopy, target);
    double logDet = 0;
    for (ui32 i = 0; i < dim; ++i) {
        logDet += log(sigmaCopy[i * dim + i]);
    }
    Y_ASSERT(std::isfinite(logDet));
    double result = 0;
    for (ui32 i = 0; i < dim; ++i) {
        result += (x[i] - mu[i]) * target[i];
    }
    result *= 0.5;

    result += 0.5 * log(2 * PI) + logDet;
    return result;
}

void NCB::TEmbeddingOnlineFeatures::Compute(
    TConstArrayRef<float> embedding,
    TOutputFloatIterator outputFeaturesIterator) const {

    TVector<double> classProbsHomoscedastic(NumClasses);
    TVector<double> classProbsHeteroscedastic(NumClasses);

    for (ui32 i = 0; i < NumClasses; ++i) {
        const double weight = ClassSizes[i] + Prior;
        const double classPrior = log(weight) - log(TotalWeight);

        if (ComputeHomoscedasticModel) {
            classProbsHomoscedastic[i] += classPrior;
            classProbsHomoscedastic[i] += LogProbNormal(embedding, Means[i], TotalSigma);
        }
        if (ComputeHeteroscedasticModel) {
            classProbsHeteroscedastic[i] += classPrior;
            classProbsHeteroscedastic[i] += LogProbNormal(embedding, Means[i], PerClassSigma[i]);
        }

        if (ComputeCosDistance) {
            *outputFeaturesIterator = CosDistance(
                MakeConstArrayRef(Means[i]),
                embedding
            );
            ++outputFeaturesIterator;
        }
    }

    if (ComputeHomoscedasticModel) {
        Softmax(classProbsHomoscedastic);
    }
    if (ComputeHeteroscedasticModel) {
        Softmax(classProbsHeteroscedastic);
    }
    for (ui32 i = 0; i < NumClasses; ++i) {
        if (ComputeHomoscedasticModel) {
            *outputFeaturesIterator = classProbsHomoscedastic[i];
            ++outputFeaturesIterator;
        }
        if (ComputeHeteroscedasticModel) {
            *outputFeaturesIterator = classProbsHeteroscedastic[i];
            ++outputFeaturesIterator;
        }
    }
}

void NCB::TEmbeddingFeaturesVisitor::UpdateEmbedding(
    ui32 classId,
    TConstArrayRef<float> embedding,
    TEmbeddingOnlineFeatures* embeddingCalcer) {

    Y_ASSERT(Dim == embedding.size());

    const double prior = embeddingCalcer->Prior;

    {
        embeddingCalcer->ClassSizes[classId]++;
        auto& sums = Sums[classId];
        auto& sums2 = Sums2[classId];

        for (ui64 i = 0; i < Dim; ++i) {
            sums[i] += embedding[i];
            for (ui32 j = 0; j <= i; ++j) {
                sums2[i * (i + 1) / 2 + j] += embedding[i] * embedding[j];
            }
        }
    }

    auto& totalSigma = embeddingCalcer->TotalSigma;
    Fill(totalSigma.begin(), totalSigma.end(), 0);
    for (ui64 x = 0; x < Dim; ++x) {
        totalSigma[x * Dim + x] = prior;
    }

    double totalWeight = prior;

    const bool needMatrices =
        embeddingCalcer->ComputeHomoscedasticModel ||
        embeddingCalcer->ComputeHeteroscedasticModel;

    const auto& classSizes = embeddingCalcer->ClassSizes;
    for (ui32 i = 0; i < NumClasses; ++i) {
        const auto& sum = Sums[i];
        const auto& sum2 = Sums2[i];

        auto& sigma = embeddingCalcer->PerClassSigma[i];
        auto& mean = embeddingCalcer->Means[i];
        const double weight = classSizes[i] + prior;
        totalWeight += classSizes[i];

        if (needMatrices) {
            for (ui64 x = 0; x < Dim; ++x) {
                mean[x] = sum[x] / weight;
                sigma[x * Dim + x] = prior;

                for (ui64 y = 0; y < x; ++y) {
                    sigma[x * Dim + y] = sum2[x * (x + 1) / 2 + y] / weight - mean[x] * mean[y];
                    sigma[y * Dim + x] = sum2[x * (x + 1) / 2 + y] / weight - mean[x] * mean[y];

                    totalSigma[x * Dim + y] += sum2[x * (x + 1) / 2 + y] - classSizes[i] * mean[x] * mean[y];
                    totalSigma[y * Dim + x] += sum2[x * (x + 1) / 2 + y] - classSizes[i] * mean[x] * mean[y];
                }

                sigma[x * Dim + x] += sum2[x * (x + 1) / 2 + x] / weight - mean[x] * mean[x];
                totalSigma[x * Dim + x] += sum2[x * (x + 1) / 2 + x] - classSizes[i] * mean[x] * mean[x];
            }
        }
    }

    for (auto& val : totalSigma) {
        val /= totalWeight;
    }
    embeddingCalcer->TotalWeight = totalWeight;
}
