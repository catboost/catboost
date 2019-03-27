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


inline double LogProbNormal(TConstArrayRef<double> x, TConstArrayRef<double> mu, TConstArrayRef<double> sigma) {
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

TVector<double> NCB::TEmbeddingOnlineFeatures::CalcFeatures(TConstArrayRef<float> embedding) const {
    TVector<double> features;

    TVector<double> classProbsHomoscedastic(NumClasses);
    TVector<double> classProbsHeteroscedastic(NumClasses);

    const ui32 dim = Embedding->Dim();
    Y_ASSERT(dim && dim == embedding.size());

    TVector<double> totalSigma(dim * dim);
    for (ui64 x = 0; x < dim; ++x) {
        totalSigma[x * dim + x] = Prior;
    }
    TVector<TVector<double>> means;
    TVector<TVector<double>> perClassSigma;
    for (ui32 i = 0; i < NumClasses; ++i) {
        means.push_back(TVector<double>(dim));
        perClassSigma.push_back(TVector<double>(dim * dim));
    }

    double totalWeight = Prior;

    bool needMatrices = ComputeHomoscedasticModel || ComputeHeteroscedasticModel;

    //TODO(noxoomo): this could be cached for test application
    for (ui32 i = 0; i < NumClasses; ++i) {
        auto& sum = Sums[i];
        auto& sum2 = Sums2[i];

        auto& sigma = perClassSigma[i];
        auto& mean = means[i];
        const auto weight = ClassSizes[i] + Prior;
        totalWeight += ClassSizes[i];

        if (needMatrices) {
            for (ui64 x = 0; x < dim; ++x) {
                mean[x] = sum[x] / weight;
                sigma[x * dim + x] = Prior;

                for (ui64 y = 0; y < x; ++y) {
                    sigma[x * dim + y] = sum2[x * (x + 1) / 2 + y] / weight - mean[x] * mean[y];
                    sigma[y * dim + x] = sum2[x * (x + 1) / 2 + y] / weight - mean[x] * mean[y];

                    totalSigma[x * dim + y] += sum2[x * (x + 1) / 2 + y] - ClassSizes[i] * mean[x] * mean[y];
                    totalSigma[y * dim + x] += sum2[x * (x + 1) / 2 + y] - ClassSizes[i] * mean[x] * mean[y];
                }

                sigma[x * dim + x] += sum2[x * (x + 1) / 2 + x] / weight - mean[x] * mean[x];
                totalSigma[x * dim + x] += sum2[x * (x + 1) / 2 + x] - ClassSizes[i] * mean[x] * mean[x];
            }
        }

        if (UseCos) {
            TVector<double> center(Sums[i].begin(), Sums[i].end());
            for (auto& val : center) {
                val /= weight;
            }
            features.push_back(CosDistance(MakeConstArrayRef(center), MakeConstArrayRef(embedding)));
        }
    }
    for (auto& val : totalSigma) {
        val /= totalWeight;
    }

    TVector<double> x(dim);
    for (ui32 i = 0; i < dim; ++i) {
        x[i] = embedding[i];
    }
    for (ui32 i = 0; i < NumClasses; ++i) {
        const auto weight = ClassSizes[i] + Prior;
        double classPrior = log(weight) - log(totalWeight);

        if (ComputeHomoscedasticModel) {
            classProbsHomoscedastic[i] += classPrior;
            classProbsHomoscedastic[i] += LogProbNormal(x, means[i], totalSigma);
        }
        if (ComputeHeteroscedasticModel) {
            classProbsHeteroscedastic[i] += classPrior;
            classProbsHeteroscedastic[i] += LogProbNormal(x, means[i], perClassSigma[i]);
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
            features.push_back(classProbsHomoscedastic[i]);
        }
        if (ComputeHeteroscedasticModel) {
            features.push_back(classProbsHeteroscedastic[i]);
        }
    }
    return features;
}

TVector<double> NCB::TEmbeddingOnlineFeatures::CalcFeaturesAndAddEmbedding(ui32 classId, TConstArrayRef<float> embedding) {
    auto result = CalcFeatures(embedding);
    AddEmbedding(classId, embedding);
    return result;
}

void NCB::TEmbeddingOnlineFeatures::AddEmbedding(ui32 classId, TConstArrayRef<float> embedding) {
    ClassSizes[classId]++;
    auto& sum = Sums[classId];
    auto& sum2 = Sums2[classId];

    for (ui64 i = 0; i < sum.size(); ++i) {
        sum[i] += embedding[i];
        for (ui32 j = 0; j <= i; ++j) {
            sum2[i * (i + 1) / 2 + j] += embedding[i] * embedding[j];
        }
    }
}
