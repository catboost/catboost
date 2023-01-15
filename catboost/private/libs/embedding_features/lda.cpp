#include "lda.h"
#include <contrib/libs/clapack/clapack.h>

namespace NCB {
    static inline void CalculateProjection(TVector<float>* scatterInner,
                                           TVector<float>* scatterTotal,
                                           TVector<float>* projectionMatrix,
                                           TVector<float>* eigenValues,
                                           TVector<float>* cache) {

        int reductionType = 1;
        char triangle = 'U';
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

        ui32 shift = scatterInner->size() - projectionMatrix->size();
        for (ui32 idx = 0; idx < projectionMatrix->size(); ++idx) {
            projectionMatrix->operator[](idx) = scatterInner->at(shift + idx);
        }
    }

    void IncrementalCloud::AddVector(const TEmbeddingsArray& embed) {
        ++AdditionalSize;
        for (int idx = 0; idx < Dimension; ++idx) {
            Buffer.push_back(embed[idx] - BaseCenter[idx]);
            NewShift[idx] += Buffer.back();
        }
        if (AdditionalSize > 31 || BaseSize < 128) {
            Update();
        }
    }

    void IncrementalCloud::Update() {
        if (AdditionalSize == 0) {
            return;
        }
        BaseSize += AdditionalSize;
        float alpha = 1.0 / BaseSize;
        float beta = 1.0 - alpha;
        for (int idx = 0; idx < Dimension; ++idx) {
            NewShift[idx] /= BaseSize;
            BaseCenter[idx] += NewShift[idx];
        }
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    Dimension, Dimension, AdditionalSize,
                    alpha,
                    &Buffer[0], Dimension,
                    &Buffer[0], Dimension,
                    beta,
                    &ScatterMatrix[0], Dimension);
        Buffer.clear();
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
               Dimension, Dimension, 1,
               -1.0,
               &NewShift[0], Dimension,
               &NewShift[0], Dimension,
               1.0, &ScatterMatrix[0], Dimension);
        AdditionalSize = 0;
        NewShift.assign(Dimension, 0);
    }

    void TLinearDACalcer::Compute(const TEmbeddingsArray& embed,
                                  TOutputFloatIterator iterator) const {
        TVector<float> proj(0.0, ProjectionDimension);
        cblas_sgemv(CblasRowMajor, CblasNoTrans,
                    TotalDimension, ProjectionDimension,
                    1.0,
                    &ProjectionMatrix[0], TotalDimension,
                    &embed[0], TotalDimension,
                    0.0,
                    &proj[0], ProjectionDimension);
        ForEachActiveFeature(
            [&proj, &iterator](ui32 featureId){
                *iterator = proj[featureId];
                ++iterator;
            }
        );
    }

    void TLinearDACalcerVisitor::Update(ui32 classId, const TEmbeddingsArray& embed,
                                        TEmbeddingFeatureCalcer* featureCalcer) {
        auto lda = dynamic_cast<TLinearDACalcer*>(featureCalcer);
        Y_ASSERT(lda);
        lda->ClassesDist[classId].AddVector(embed);
        lda->TotalDist.AddVector(embed);
    }

    void TLinearDACalcerVisitor::Flush(TEmbeddingFeatureCalcer* featureCalcer) {
        auto lda = dynamic_cast<TLinearDACalcer*>(featureCalcer);
        Y_ASSERT(lda);
        ui32 dim = lda->TotalDimension;
        TVector<float> meanScatter(dim * dim, 0);
        TVector<float> totalScatter(lda->TotalDist.ScatterMatrix);
        for (ui32 idx = 0; idx < meanScatter.size(); ++idx) {
            for (int classIdx = 0; classIdx < lda->NumClasses; ++classIdx) {
                meanScatter[idx] += lda->ClassesDist[classIdx].BaseSize *
                                    lda->ClassesDist[classIdx].ScatterMatrix[idx];
            }
            meanScatter[idx] *= (1 - lda->RegParam);
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
