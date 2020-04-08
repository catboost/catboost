#pragma once

#include "fwd.h"
#include "scale_and_bias.h"

#include <util/generic/algorithm.h>
#include <util/generic/array_ref.h>
#include <util/generic/maybe.h>
#include <util/generic/vector.h>

#include <library/fast_exp/fast_exp.h>

#include <cmath>

inline void CalcSoftmax(const TConstArrayRef<double> approx, TArrayRef<double> softmax) {
    double maxApprox = *MaxElement(approx.begin(), approx.end());
    for (size_t dim = 0; dim < approx.size(); ++dim) {
        softmax[dim] = approx[dim] - maxApprox;
    }
    FastExpInplace(softmax.data(), softmax.ysize());
    double sumExpApprox = 0;
    for (auto curSoftmax : softmax) {
        sumExpApprox += curSoftmax;
    }
    for (auto& curSoftmax : softmax) {
        curSoftmax /= sumExpApprox;
    }
}

inline TVector<double> CalcExponent(TVector<double> approx) {
    FastExpInplace(approx.data(), approx.ysize());
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
    FastExpInplace(logSoftmax.data(), logSoftmax.ysize());
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
    FastExpInplace(target.data(), target.size());
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

//approx and target could overlap
inline void CalcLogSigmoid(const TConstArrayRef<double> approx, TArrayRef<double> target) {
    Y_ASSERT(approx.size() == target.size());
    InvertSign(approx, target);
    FastExpInplace(target.data(), target.size());
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
            Y_ASSERT(ApproxDimension == 1);
            ::ApplyScaleAndBias(ScaleAndBias, GetResultBlockView(blockId, 1), startTree);
        }

        inline void PostprocessBlock(ui32 blockId, ui32 startTree) {
            ApplyScaleAndBias(blockId, startTree);
            if (PredictionType == EPredictionType::RawFormulaVal) {
                return;
            }
            if (ApproxDimension == 1) {
                auto blockView = GetResultBlockView(blockId, 1);
                if (PredictionType == EPredictionType::Probability) {
                    CalcSigmoid(blockView, blockView);
                }
                if (PredictionType == EPredictionType::Exponent) {
                    FastExpInplace(blockView.data(), blockView.ysize());;
                }
                if (PredictionType == EPredictionType::Class) {
                    for (auto &val : blockView) {
                        val = val > BinclassRawValueBorder;
                    }
                }
            } else {
                if (PredictionType == EPredictionType::Probability) {
                    auto blockView = GetResultBlockView(blockId, ApproxDimension);
                    for (size_t i = 0; i < blockView.size(); i += ApproxDimension) {
                        auto docView = blockView.Slice(i, ApproxDimension);
                        CalcSoftmax(docView, docView);
                    }
                } else {
                    Y_ASSERT(PredictionType == EPredictionType::Class);
                    auto resultView = GetResultBlockView(blockId, 1);
                    for (size_t objId = 0; objId < resultView.size(); ++objId) {
                        auto objRawIterator = IntermediateBlockResults.begin() + objId * ApproxDimension;
                        resultView[objId] = MaxElement(objRawIterator, objRawIterator + ApproxDimension) - objRawIterator;
                    }
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