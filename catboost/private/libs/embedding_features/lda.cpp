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

        ui32 shift = scatterInner->size() - projectionMatrix->size();
        std::copy(scatterTotal->begin() + shift, scatterTotal->end(), projectionMatrix->begin());
    }

    void IncrementalCloud::AddVector(const TEmbeddingsArray& embed) {
        ++AdditionalSize;
        for (int idx = 0; idx < Dimension; ++idx) {
            Buffer.push_back(embed[idx] - BaseCenter[idx]);
            NewShift[idx] += Buffer.back();
        }
        if (AdditionalSize > 0) {
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
                    &Buffer[0], AdditionalSize,
                    &Buffer[0], AdditionalSize,
                    beta,
                    &ScatterMatrix[0], Dimension);
        Buffer.clear();
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
               Dimension, Dimension, 1,
               -1.0,
               &NewShift[0], 1,
               &NewShift[0], 1,
               1.0, &ScatterMatrix[0], Dimension);
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

    void TLinearDACalcerVisitor::Update(ui32 classId, const TEmbeddingsArray& embed,
                                        TEmbeddingFeatureCalcer* featureCalcer) {
        auto lda = dynamic_cast<TLinearDACalcer*>(featureCalcer);
        Y_ASSERT(lda);
        lda->ClassesDist[classId].AddVector(embed);
        lda->TotalDist.AddVector(embed);
        if (2 * LastFlush <= lda->TotalDist.BaseSize) {
            Flush(featureCalcer);
            LastFlush = lda->TotalDist.BaseSize;
        }
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
