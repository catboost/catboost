#pragma once

#include <catboost/cuda/cuda_util/kernel/instructions.cuh>
#include <catboost/cuda/cuda_util/kernel/random_gen.cuh>
#include <catboost/cuda/cuda_util/kernel/kernel_helpers.cuh>
#include <catboost/cuda/gpu_data/gpu_structures.h>
#include <cmath>

namespace NKernel {

    class TSolarScoreCalcer {
    public:
        __host__ __device__ TSolarScoreCalcer(float) {
        }

        __host__ __device__ TSolarScoreCalcer(const TSolarScoreCalcer& other)  = default;

        __forceinline__ __host__ __device__ void NextFeature(TCBinFeature) {
            Score = 0;
        }

        __forceinline__ __host__ __device__ void AddLeaf(double sum, double weight) {
            Score += (weight > 1e-20f ? (-sum * sum) * (1 + 2 * log(weight + 1.0)) / weight : 0);
        }

        __forceinline__ __host__ __device__ double GetScore() {
            return Score;
        }


        __forceinline__ __host__ __device__ void Combine(const TSolarScoreCalcer& other) {
            Score += other.Score;
        }
    private:
        float Lambda = 0;
        float Score = 0;
    };


    class TL2ScoreCalcer {
    public:
        __host__ __device__ TL2ScoreCalcer(float lambda, float metaExponent = 1.0f)
        : Lambda(lambda)
        , MetaExponent(metaExponent) {

        }
        __host__ __device__ TL2ScoreCalcer(const TL2ScoreCalcer&)  = default;

        __forceinline__ __host__ __device__ void NextFeature(TCBinFeature) {
            Score = 0;
        }

        __forceinline__ __host__ __device__ void AddLeaf(double sum, double weight) {
            double leafScore = (weight > 1e-20f ? (-sum * sum) / (weight + Lambda) : 0);
            Score += (MetaExponent == 1.0f || leafScore == 0.0f) ? leafScore : std::copysign(pow(std::abs(leafScore) / weight, MetaExponent), leafScore) * weight;
        }

        __forceinline__ __host__ __device__ double GetScore() {
            return Score;
        }

        __forceinline__ __host__ __device__ void Combine(const TL2ScoreCalcer& other) {
            Score += other.Score;
        }
    private:
        float Score = 0;
        float Lambda = 0;
        float MetaExponent = 1;
    };

    class TLOOL2ScoreCalcer {
    public:
        __host__ __device__ TLOOL2ScoreCalcer(float) {

        }

        __host__ __device__ TLOOL2ScoreCalcer(const TLOOL2ScoreCalcer& other)  = default;

        __forceinline__ __host__ __device__ void NextFeature(TCBinFeature) {
            Score = 0;
        }

        __forceinline__ __host__ __device__ void AddLeaf(double sum, double weight) {
            float adjust = weight > 1 ? weight / (weight - 1) : 0;
            adjust = adjust * adjust;
            Score += (weight > 0 ? adjust * (-sum * sum) / weight : 0);
        }

        __forceinline__ __host__ __device__ double GetScore() {
            return Score;
        }

        __forceinline__ __host__ __device__ void Combine(const TLOOL2ScoreCalcer& other) {
            Score += other.Score;
        }

    private:
        float Score = 0;
    };

    class TSatL2ScoreCalcer {
    public:

        __host__ __device__ TSatL2ScoreCalcer(float) {

        }

        __host__ __device__ TSatL2ScoreCalcer(const TSatL2ScoreCalcer& other)  = default;

        __forceinline__ __host__ __device__ void NextFeature(TCBinFeature) {
            Score = 0;
        }

        __forceinline__ __host__ __device__ void AddLeaf(double sum, double weight) {
            float adjust = weight > 2 ? weight * (weight - 2) / (weight * weight - 3 * weight + 1) : 0;
            Score += (weight > 0 ? adjust * ((-sum * sum) / weight)  : 0);
        }

        __forceinline__ __host__ __device__ double GetScore() {
            return Score;
        }

        __forceinline__ __host__ __device__ void Combine(const TSatL2ScoreCalcer& other) {
            Score += other.Score;
        }
    private:
        float Score = 0;
    };


    class TCosineScoreCalcer {
    public:
        __host__ __device__ TCosineScoreCalcer(float lambda,
                                                    bool normalize,
                                                    float scoreStdDev,
                                                    ui64 globalSeed
        )
                : Lambda(lambda)
                  , Normalize(normalize)
                  , ScoreStdDev(scoreStdDev)
                  , GlobalSeed(globalSeed) {

        }
        __host__ __device__ TCosineScoreCalcer(const TCosineScoreCalcer& other) = default;

        __forceinline__ __host__ __device__ void NextFeature(TCBinFeature bf) {
            FeatureId = bf.FeatureId;
            Score = 0;
            DenumSqr = 1e-10f;
        }

        __forceinline__ __host__ __device__ void AddLeaf(double sum, double weight) {
            double lambda = Normalize ? Lambda * weight : Lambda;

            const double mu =  weight > 0.0f ? (sum / (weight + lambda)) : 0.0f;
            Score +=  sum * mu;
            DenumSqr += weight * mu * mu;
        }

        __forceinline__ __host__ __device__ float GetScore() {
            float score = DenumSqr > 1e-15f ? -Score / sqrt(DenumSqr) : FLT_MAX;
            if (ScoreStdDev) {
                ui64 seed = GlobalSeed + FeatureId;
                AdvanceSeed(&seed, 4);
                score += NextNormal(&seed) * ScoreStdDev;
            }

            return score;
        }

        __forceinline__ __host__ __device__ void Combine(const  TCosineScoreCalcer& other) {
            Score += other.Score;
            DenumSqr += other.DenumSqr;
        }
    private:
        float Lambda;
        bool Normalize;
        float ScoreStdDev;
        ui64 GlobalSeed;

        int FeatureId = 0;
        double Score = 0;
        double DenumSqr = 0;
    };



}
