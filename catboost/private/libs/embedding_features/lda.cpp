#include "lda.h"
#include <contrib/libs/clapack/clapack.h>

namespace NCB {
    static inline void CalculateProjection(TVector<float>* scatterInner,
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
        ForEachActiveFeature(
            [&proj, &iterator](ui32 featureId){
                *iterator = proj[featureId];
                ++iterator;
            }
        );
    }

    void TLinearDACalcer::BetweenScatterCalculation(TVector<float>* result) {
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

    void TLinearDACalcerVisitor::Update(ui32 classId, const TEmbeddingsArray& embed,
                                        TEmbeddingFeatureCalcer* featureCalcer) {
        auto lda = dynamic_cast<TLinearDACalcer*>(featureCalcer);
        Y_ASSERT(lda);
        lda->ClassesDist[classId].AddVector(embed);
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
        TVector<float> meanScatter(dim * dim, 0);
        TVector<float> totalScatter(dim * dim, 0);
        lda->BetweenScatterCalculation(&totalScatter);
        for (int classIdx = 0; classIdx < lda->NumClasses; ++classIdx) {
            float weight = lda->ClassesDist[classIdx].TotalSize() / lda->Size;
            std::transform(meanScatter.begin(), meanScatter.end(),
                           lda->ClassesDist[classIdx].ScatterMatrix.begin(),
                           meanScatter.begin(), [weight](float x, float y) {
                               return x + weight * y;
                           });
        }
        for (ui32 idx = 0; idx < meanScatter.size(); idx+= dim + 1) {
            meanScatter[idx] += lda->RegParam;
        }
        CalculateProjection(&meanScatter,
                            &totalScatter,
                            &lda->ProjectionMatrix,
                            &lda->EigenValues,
                            &lda->ProjectionCalculationCache);
    }
};
