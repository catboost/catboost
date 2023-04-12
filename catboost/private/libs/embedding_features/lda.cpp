#include "lda.h"
#include <contrib/libs/clapack/clapack.h>

#include <catboost/private/libs/embedding_features/flatbuffers/embedding_feature_calcers.fbs.h>

#include <util/generic/ymath.h>


namespace NCB {
    void CalculateProjection(TVector<float>* scatterInner,
                                           TVector<float>* scatterTotal,
                                           TVector<float>* projectionMatrix,
                                           TVector<float>* eigenValues,
                                           TVector<float>* cache) {

        int reductionType = 1;
        char triangle = 'L';
        char workType = 'V';
        int dim = eigenValues->size();
        int lenght = cache->size();
        int info;

        ssygst_(&reductionType, &triangle, &dim,
                &scatterTotal->at(0), &dim,
                &scatterInner->at(0), &dim, &info);

        Y_ASSERT(info == 0);

        ssyev_(&workType, &triangle, &dim,
              &scatterTotal->at(0), &dim,
              &eigenValues->at(0),
              &cache->at(0), &lenght, &info);

        Y_ASSERT(info == 0);

        std::copy(scatterTotal->end() - projectionMatrix->size(), scatterTotal->end(), projectionMatrix->begin());
    }

    float CalculateGaussianLikehood(const TEmbeddingsArray& embed,
                                                  const TVector<float>& mean,
                                                  const TVector<float>& scatter) {
        TVector<float> diff(mean);
        TVector<float> askew(mean.size());
        for (ui32 id = 0; id < diff.size(); ++id) {
            diff[id] -= embed[id];
        }
        cblas_sgemv(CblasRowMajor, CblasNoTrans,
                    mean.size(), mean.size(),
                    1.0,
                    scatter.data(), mean.size(),
                    diff.data(), 1,
                    0.0,
                    &askew[0], 1);
        float deg = 0;
        for (ui32 id = 0; id < diff.size(); ++id) {
            deg += askew[id] * diff[id];
        }
        return Exp2f(- M_LN2_INV * 0.5 * deg);
    }

    void InverseMatrix(TVector<float>* matrix, int dim) {
        int info;
        TVector<int> pivot(dim);
        TVector<float> cache(dim);
        sgetrf_(&dim, &dim, matrix->data(), &dim, pivot.data(), &info);
        Y_ASSERT(info == 0);
        sgetri_(&dim, matrix->data(), &dim, pivot.data(), cache.data(), &dim, &info);
        Y_ASSERT(info == 0);
    }

    void IncrementalCloud::AddVector(const TEmbeddingsArray& embed) {
        ++AdditionalSize;
        for (int idx = 0; idx < Dimension; ++idx) {
            Buffer.push_back(embed[idx] - BaseCenter[idx]);
            NewShift[idx] += Buffer.back();
        }
        if (BaseSize < 128 || AdditionalSize >= 32) {
            Update();
        }
    }

    void IncrementalCloud::Update() {
        if (AdditionalSize == 0) {
            return;
        }
        for (int idx = 0; idx < Dimension; ++idx) {
            NewShift[idx] /= TotalSize();
            BaseCenter[idx] += NewShift[idx];
        }
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    Dimension, Dimension, AdditionalSize,
                    1.0 / TotalSize(),
                    &Buffer[0], Dimension,
                    &Buffer[0], Dimension,
                    BaseSize / TotalSize(),
                    &ScatterMatrix[0], Dimension);
        Buffer.clear();
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
               Dimension, Dimension, 1,
               -1.0,
               &NewShift[0], 1,
               &NewShift[0], 1,
               1.0, &ScatterMatrix[0], Dimension);
        BaseSize += AdditionalSize;
        AdditionalSize = 0;
        NewShift.assign(Dimension, 0);
    }

    void TLinearDACalcer::Compute(const TEmbeddingsArray& embed,
                                  TOutputFloatIterator iterator) const {
        TVector<float> proj(ProjectionDimension, 0.0);
        cblas_sgemv(CblasRowMajor, CblasNoTrans,
                    ProjectionDimension, TotalDimension,
                    1.0,
                    &ProjectionMatrix[0], TotalDimension,
                    &embed[0], 1,
                    0.0,
                    &proj[0], 1);
        if (IsClassification && ComputeProbabilities) {
            std::vector<float> likehoods(NumClasses);
            float likehood = 0;
            for (int classId = 0; classId < NumClasses; ++classId) {
                likehoods[classId] = CalculateGaussianLikehood(embed,
                                                               ClassesDist[classId].BaseCenter,
                                                               BetweenMatrix);
                likehood += likehoods[classId];
            }
            for (auto like : likehoods) {
                proj.push_back((likehood > 1e-6 ? like/likehood : 1.0/NumClasses));
            }
        }
        ForEachActiveFeature(
            [&proj, &iterator](ui32 featureId){
                *iterator = proj[featureId];
                ++iterator;
            }
        );
    }

    void TLinearDACalcer::TotalScatterCalculation(TVector<float>* result) {
        TVector<float> totalMean(TotalDimension, 0);
        for (auto& dist : ClassesDist) {
            float weight = dist.TotalSize()/Size;
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        TotalDimension, TotalDimension, 1,
                        weight,
                        &dist.BaseCenter[0], 1,
                        &dist.BaseCenter[0], 1,
                        1.0, result->data(), TotalDimension);
            std::transform(totalMean.begin(), totalMean.end(), dist.BaseCenter.begin(),
                           totalMean.begin(), [weight](float x, float y) {
                               return x + weight * y;
                           });
        }
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    TotalDimension, TotalDimension, 1,
                    -1.0,
                    &totalMean[0], 1,
                    &totalMean[0], 1,
                    1.0, result->data(), TotalDimension);
    }

    void TLinearDACalcerVisitor::Update(float target, const TEmbeddingsArray& embed,
                                        TEmbeddingFeatureCalcer* featureCalcer) {
        auto lda = dynamic_cast<TLinearDACalcer*>(featureCalcer);
        Y_ASSERT(lda);
        lda->ClassesDist[lda->IsClassification ? (size_t)target : 0].AddVector(embed);
        ++lda->Size;
        if (2 * LastFlush <= lda->Size) {
            Flush(featureCalcer);
            LastFlush = lda->Size;
        }
    }

    void TLinearDACalcerVisitor::Flush(TEmbeddingFeatureCalcer* featureCalcer) {
        auto lda = dynamic_cast<TLinearDACalcer*>(featureCalcer);
        Y_ASSERT(lda);
        ui32 dim = lda->TotalDimension;
        TVector<float> totalScatter(dim * dim, 0);
        lda->TotalScatterCalculation(&totalScatter);
        if (lda->IsClassification) {
            lda->BetweenMatrix.assign(dim * dim, 0);
            for (size_t classIdx = 0; classIdx < lda->ClassesDist.size(); ++classIdx) {
                float weight = lda->ClassesDist[classIdx].TotalSize() / lda->Size;
                std::transform(lda->BetweenMatrix.begin(), lda->BetweenMatrix.end(),
                               lda->ClassesDist[classIdx].ScatterMatrix.begin(),
                               lda->BetweenMatrix.begin(), [weight](float x, float y) {
                                   return x + weight * y;
                               });
            }
        } else {
            lda->BetweenMatrix = lda->ClassesDist[0].ScatterMatrix;
        }
        for (ui32 idx = 0; idx < lda->BetweenMatrix.size(); idx+= dim + 1) {
            lda->BetweenMatrix[idx] += lda->RegParam;
        }
        CalculateProjection(&lda->BetweenMatrix,
                            &totalScatter,
                            &lda->ProjectionMatrix,
                            &lda->EigenValues,
                            &lda->ProjectionCalculationCache);
        if (lda->IsClassification && lda->ComputeProbabilities) {
            InverseMatrix(&lda->BetweenMatrix, lda->TotalDimension);
        }
    }

    TEmbeddingFeatureCalcer::TEmbeddingCalcerFbs TLinearDACalcer::SaveParametersToFB(flatbuffers::FlatBufferBuilder& builder) const {
        using namespace NCatBoostFbs::NEmbeddings;

        auto fbProjectionMatrix = builder.CreateVector(
            ProjectionMatrix.data(),
            ProjectionMatrix.size()
        );
        const auto& fbLDA = CreateTLDA(
            builder,
            TotalDimension,
            NumClasses,
            ProjectionDimension,
            ComputeProbabilities,
            fbProjectionMatrix,
            IsClassification
        );
        return TEmbeddingCalcerFbs(TAnyEmbeddingCalcer_TLDA, fbLDA.Union());
    }

    void TLinearDACalcer::LoadParametersFromFB(const NCatBoostFbs::NEmbeddings::TEmbeddingCalcer* calcer) {
        auto fbLDA = calcer->FeatureCalcerImpl_as_TLDA();
        TotalDimension = fbLDA->TotalDimension();
        NumClasses = fbLDA->NumClasses();
        ProjectionDimension = fbLDA->ProjectionDimension();
        ComputeProbabilities = fbLDA->ComputeProbabilities();
        auto projection = fbLDA->ProjectionMatrix();

        Y_ASSERT(static_cast<int>(projection->size()) == ProjectionDimension * TotalDimension);

        ProjectionMatrix.yresize(projection->size());
        Copy(projection->begin(), projection->end(), ProjectionMatrix.begin());

        IsClassification = fbLDA->IsClassification();
    }

    void TLinearDACalcer::SaveLargeParameters(IOutputStream*) const {
    }

    void TLinearDACalcer::LoadLargeParameters(IInputStream*) {
    }

    TEmbeddingFeatureCalcerFactory::TRegistrator<TLinearDACalcer> LDARegistrator(EFeatureCalcerType::LDA);

};
