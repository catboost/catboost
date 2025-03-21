#pragma once

#include "fwd.h"
#include "scale_and_bias.h"

#include <util/generic/algorithm.h>
#include <util/generic/array_ref.h>
#include <util/generic/maybe.h>
#include <util/generic/vector.h>

#include <catboost/libs/helpers/math_utils.h>

#include <cmath>

inline void CalcSoftmax(const TConstArrayRef<double> approx, TArrayRef<double> softmax) {
    double maxApprox = *MaxElement(approx.begin(), approx.end());
    for (size_t dim = 0; dim < approx.size(); ++dim) {
        softmax[dim] = approx[dim] - maxApprox;
    }
    NCB::FastExpWithInfInplace(softmax.data(), softmax.ysize());
    double sumExpApprox = 0;
    for (auto curSoftmax : softmax) {
        sumExpApprox += curSoftmax;
    }
    for (auto& curSoftmax : softmax) {
        curSoftmax /= sumExpApprox;
    }
}

inline TVector<double> CalcExponent(TVector<double> approx) {
    NCB::FastExpWithInfInplace(approx.data(), approx.ysize());
    return approx;
}

inline void CalcSquaredExponentInplace(TArrayRef<double> approx) {
    constexpr size_t blockSize = 4096;
    for (size_t i = 0; i < approx.size(); i += blockSize) {
        size_t currBlockSize = Min<size_t>(blockSize, approx.size() - i);
        double* blockStartPtr = approx.data() + i;
        for (size_t j = 0; j < currBlockSize; ++j) {
            blockStartPtr[j] *= 2.0f;
        }
        NCB::FastExpWithInfInplace(blockStartPtr, currBlockSize);
    }
}

inline TVector<double> CalcSquaredExponent(TVector<double> approx) {
    CalcSquaredExponentInplace(approx);
    return approx;
}

inline void CalcSoftmax(const TConstArrayRef<double> approx, TVector<double>* softmax) {
    CalcSoftmax(approx, *softmax);
}

inline void CalcLogSoftmax(const TConstArrayRef<double> approx, TArrayRef<double> logSoftmax) {
    double maxApprox = *MaxElement(approx.begin(), approx.end());
    for (size_t dim = 0; dim < approx.size(); ++dim) {
        logSoftmax[dim] = approx[dim] - maxApprox;
    }
    NCB::FastExpWithInfInplace(logSoftmax.data(), logSoftmax.ysize());
    double logSumExpApprox = 0;
    for (auto curLogSoftmax : logSoftmax) {
        logSumExpApprox += curLogSoftmax;
    }
    logSumExpApprox = std::log(logSumExpApprox);

    for (size_t dim = 0; dim < approx.size(); ++dim) {
        logSoftmax[dim] = approx[dim] - maxApprox - logSumExpApprox;
    }
}

inline void CalcLogSoftmax(const TConstArrayRef<double> approx, TVector<double>* softmax) {
    CalcLogSoftmax(approx, *softmax);
}

//approx and target could overlap
inline void InvertSign(const TConstArrayRef<double> approx, TArrayRef<double> target) {
     Y_ASSERT(approx.size() == target.size());
     for (size_t i = 0; i < approx.size(); ++i) {
         target[i] = -approx[i];
     }
}

inline TVector<double> InvertSign(const TConstArrayRef<double> approx) {
    TVector<double> target;
    target.yresize(approx.size());
    InvertSign(approx, target);
    return target;
}

//approx and target could overlap
inline void CalcSigmoid(const TConstArrayRef<double> approx, TArrayRef<double> target) {
    Y_ASSERT(approx.size() == target.size());
    InvertSign(approx, target);
    NCB::FastExpWithInfInplace(target.data(), target.size());
    for (auto& val : target) {
        val = 1. / (1. + val);
    }
}

inline TVector<double> CalcSigmoid(const TConstArrayRef<double> approx) {
    TVector<double> probabilities;
    probabilities.yresize(approx.size());
    // TODO(kirillovs): uncomment with canonization in next refactoring step
    //CalcSigmoid(approx, probabilities);
    for (size_t i = 0; i < approx.size(); ++i) {
        probabilities[i] = 1. / (1. + exp(-approx[i]));
    }
    return probabilities;
}

inline TVector<double> CalcEntropyFromProbabilities(const TConstArrayRef<double> probabilities) {
    TVector<double> entropy;
    entropy.yresize(probabilities.size());
    for (size_t i = 0; i < probabilities.size(); ++i) {
        entropy[i] = - probabilities[i] * std::log(probabilities[i]) - (1 - probabilities[i]) * std::log(1 - probabilities[i]);
    }
    return entropy;
}

//approx and target could overlap
inline void CalcLogSigmoid(const TConstArrayRef<double> approx, TArrayRef<double> target) {
    Y_ASSERT(approx.size() == target.size());
    InvertSign(approx, target);
    NCB::FastExpWithInfInplace(target.data(), target.size());
    for (auto& val : target) {
        val = -std::log(1. + val);
    }
}

inline TVector<double> CalcLogSigmoid(const TConstArrayRef<double> approx) {
    TVector<double> probabilities;
    probabilities.yresize(approx.size());
    // TODO(kirillovs): uncomment with canonization in next refactoring step
    //CalcLogSigmoid(approx, probabilities);
    for (size_t i = 0; i < approx.size(); ++i) {
        probabilities[i] = -std::log(1. + std::exp(-approx[i]));
    }
    return probabilities;
}

namespace NCB::NModelEvaluation {

    class TEvalResultProcessor {
    public:
        TEvalResultProcessor(
            size_t docCount,
            TArrayRef<double> results,
            EPredictionType predictionType,
            TScaleAndBias scaleAndBias,
            ui32 approxDimension,
            ui32 blockSize,
            TMaybe<double> binclassProbabilityBorder = Nothing()
        );

        inline TArrayRef<double> GetResultBlockView(ui32 blockId, ui32 dimension) {
            return Results.Slice(
                blockId * BlockSize * dimension,
                Min<ui32>(
                    BlockSize * dimension,
                    Results.size() - (blockId * BlockSize * dimension)
                )
            );
        }

        inline TArrayRef<double> GetViewForRawEvaluation(ui32 blockId) {
            if (!IntermediateBlockResults.empty()) {
                return IntermediateBlockResults;
            }
            return GetResultBlockView(blockId, ApproxDimension);
        }

        inline void ApplyScaleAndBias(ui32 blockId, ui32 startTree) {
            if (ScaleAndBias.IsIdentity()) {
                return;
            }
            Y_ASSERT(ApproxDimension == ScaleAndBias.GetBiasRef().size());
            ::ApplyScaleAndBias(ScaleAndBias, GetResultBlockView(blockId, ApproxDimension), startTree);
        }

        inline void PostprocessBlock(ui32 blockId, ui32 startTree) {
            ApplyScaleAndBias(blockId, startTree);
            if (PredictionType == EPredictionType::RawFormulaVal) {
                return;
            }
            if (ApproxDimension == 1) {
                auto blockView = GetResultBlockView(blockId, 1);
                switch (PredictionType) {
                    case EPredictionType::Probability:
                        CalcSigmoid(blockView, blockView);
                        break;
                    case EPredictionType::Exponent:
                        NCB::FastExpWithInfInplace(blockView.data(), blockView.ysize());
                        break;
                    case EPredictionType::Class:
                        for (auto &val : blockView) {
                            val = val > BinclassRawValueBorder;
                        }
                        break;
                    default:
                        CB_ENSURE(false, "unsupported prediction type");
                }
            } else {
                switch (PredictionType) {
                    case EPredictionType::RMSEWithUncertainty: {
                        auto blockView = GetResultBlockView(blockId, ApproxDimension);
                        for (size_t i = 1; i < blockView.size(); i += ApproxDimension) {
                            auto docView = blockView.Slice(i, 1);
                            CalcSquaredExponentInplace(MakeArrayRef<double>(docView.data(), 1));
                        }
                        break;
                    }
                    case EPredictionType::Probability: {
                        auto blockView = GetResultBlockView(blockId, ApproxDimension);
                        for (size_t i = 0; i < blockView.size(); i += ApproxDimension) {
                            auto docView = blockView.Slice(i, ApproxDimension);
                            CalcSoftmax(docView, docView);
                        }
                        break;
                    }
                    case EPredictionType::MultiProbability: {
                        auto blockView = GetResultBlockView(blockId, ApproxDimension);
                        CalcSigmoid(blockView, blockView);
                        break;
                    }
                    case EPredictionType::Class: {
                        auto resultView = GetResultBlockView(blockId, 1);
                        for (size_t objId = 0; objId < resultView.size(); ++objId) {
                            auto objRawIterator = IntermediateBlockResults.begin() + objId * ApproxDimension;
                            resultView[objId] =
                                MaxElement(objRawIterator, objRawIterator + ApproxDimension) - objRawIterator;
                        }
                        break;
                    }
                    default:
                        Y_ASSERT(false);
                }
            }
        }

    private:
        TArrayRef<double> Results;
        EPredictionType PredictionType;
        TScaleAndBias ScaleAndBias;
        ui32 ApproxDimension;
        ui32 BlockSize;

        TVector<double> IntermediateBlockResults;

        double BinclassRawValueBorder = 0.0;
    };
}
