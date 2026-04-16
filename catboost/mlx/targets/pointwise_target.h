#pragma once

// Pointwise target functions for CatBoost-MLX.
// RMSE (L2), Logloss (binary classification), MultiClass (softmax),
// MAE, Quantile, Huber, Poisson, Tweedie, and MAPE.

#include <catboost/mlx/targets/target_func.h>

namespace NCatboostMlx {

    // RMSE (Root Mean Squared Error) / L2 loss.
    // Loss = 0.5 * (pred - target)^2
    // Gradient = pred - target
    // Hessian = 1.0 (constant)
    class TRMSETarget : public IMLXTargetFunc {
    public:
        ui32 GetApproxDimension() const override { return 1; }

        void ComputeDerivatives(
            const mx::array& cursor,
            const mx::array& targets,
            const mx::array& weights,
            mx::array& gradients,
            mx::array& hessians
        ) const override {
            // Gradient = prediction - target
            gradients = mx::subtract(cursor, targets);

            // Apply sample weights
            gradients = mx::multiply(gradients, weights);

            // Hessian is constant = weight (for weighted case) or 1.0
            hessians = mx::copy(weights);

            // No EvalNow — gradients/hessians are consumed lazily by the histogram
            // kernel in structure_searcher.cpp. Eval happens at the iteration
            // boundary in mlx_boosting.cpp via EvalAtBoundary(cursor).
        }

        mx::array ComputeLoss(
            const mx::array& cursor,
            const mx::array& targets,
            const mx::array& weights
        ) const override {
            // RMSE = sqrt(weighted mean of squared errors)
            auto diff = mx::subtract(cursor, targets);
            auto sqDiff = mx::multiply(diff, diff);
            auto weightedSqDiff = mx::multiply(sqDiff, weights);
            auto loss = mx::sqrt(mx::mean(weightedSqDiff));
            // No EvalNow — caller uses .item<float>() which forces evaluation.
            return loss;
        }
    };

    // Logloss (Binary Cross-Entropy) for binary classification.
    // Target: 0 or 1
    // ApproxDimension: 1
    // Gradient = sigmoid(pred) - target
    // Hessian = sigmoid(pred) * (1 - sigmoid(pred))
    class TLoglossTarget : public IMLXTargetFunc {
    public:
        ui32 GetApproxDimension() const override { return 1; }

        void ComputeDerivatives(
            const mx::array& cursor,
            const mx::array& targets,
            const mx::array& weights,
            mx::array& gradients,
            mx::array& hessians
        ) const override {
            // sigmoid = 1 / (1 + exp(-cursor))
            auto sigmoid = mx::sigmoid(cursor);

            // Gradient = (sigmoid - target) * weight
            gradients = mx::multiply(mx::subtract(sigmoid, targets), weights);

            // Hessian = sigmoid * (1 - sigmoid) * weight
            hessians = mx::multiply(
                mx::multiply(sigmoid, mx::subtract(mx::array(1.0f), sigmoid)),
                weights
            );

            // Clamp hessian to avoid numerical issues
            hessians = mx::maximum(hessians, mx::array(1e-16f));

            // No EvalNow — consumed lazily by histogram kernel.
        }

        mx::array ComputeLoss(
            const mx::array& cursor,
            const mx::array& targets,
            const mx::array& weights
        ) const override {
            // Binary cross-entropy: -mean(w * (t*log(sig) + (1-t)*log(1-sig)))
            auto sigmoid = mx::sigmoid(cursor);
            auto eps = mx::array(1e-15f);
            auto logSig = mx::log(mx::add(sigmoid, eps));
            auto log1mSig = mx::log(mx::add(mx::subtract(mx::array(1.0f), sigmoid), eps));

            auto loss = mx::negative(mx::mean(mx::multiply(
                weights,
                mx::add(
                    mx::multiply(targets, logSig),
                    mx::multiply(mx::subtract(mx::array(1.0f), targets), log1mSig)
                )
            )));
            // No EvalNow — caller uses .item<float>() which forces evaluation.
            return loss;
        }
    };

    // MultiClass (softmax cross-entropy) for K-class classification.
    // Target: integer class index 0..numClasses-1
    // ApproxDimension: numClasses - 1 (K-th class is implicit with value 0)
    // Gradient_k = softmax_k - indicator(target == k)
    // Hessian_k = softmax_k * (1 - softmax_k)  (diagonal approximation)
    class TMultiClassTarget : public IMLXTargetFunc {
    public:
        explicit TMultiClassTarget(ui32 numClasses)
            : NumClasses_(numClasses)
            , ApproxDimension_(numClasses - 1)
        {}

        ui32 GetApproxDimension() const override { return ApproxDimension_; }

        void ComputeDerivatives(
            const mx::array& cursor,    // [K, numDocs] where K = numClasses-1
            const mx::array& targets,   // [numDocs] integer class indices
            const mx::array& weights,   // [numDocs]
            mx::array& gradients,       // [K, numDocs] output
            mx::array& hessians         // [K, numDocs] output
        ) const override {
            const int K = static_cast<int>(ApproxDimension_);
            const int numDocs = cursor.shape(-1);

            // Softmax with implicit K-th class (value 0)
            auto probs = ComputeSoftmax(cursor, numDocs);  // [K, numDocs]

            // Indicator: for each dim k, indicator_k[d] = 1.0 if targets[d] == k
            // Build one-hot: [K, numDocs]
            auto targetInt = mx::astype(targets, mx::uint32);
            auto oneHot = mx::zeros({K, numDocs}, mx::float32);
            for (int k = 0; k < K; ++k) {
                auto isClass = mx::astype(
                    mx::equal(targetInt, mx::array(static_cast<uint32_t>(k))),
                    mx::float32
                );  // [numDocs]
                oneHot = mx::where(
                    mx::equal(
                        mx::reshape(mx::arange(K), {K, 1}),
                        mx::array(k)
                    ),
                    mx::reshape(isClass, {1, numDocs}),
                    oneHot
                );
            }

            // Gradient = (prob - indicator) * weight
            auto weightsExpanded = mx::reshape(weights, {1, numDocs});  // [1, numDocs]
            gradients = mx::multiply(mx::subtract(probs, oneHot), weightsExpanded);

            // Hessian (diagonal approximation) = prob * (1 - prob) * weight
            hessians = mx::multiply(
                mx::multiply(probs, mx::subtract(mx::array(1.0f), probs)),
                weightsExpanded
            );
            hessians = mx::maximum(hessians, mx::array(1e-16f));

            // No EvalNow — consumed lazily by histogram kernel.
        }

        mx::array ComputeLoss(
            const mx::array& cursor,    // [K, numDocs]
            const mx::array& targets,   // [numDocs]
            const mx::array& weights    // [numDocs]
        ) const override {
            const int K = static_cast<int>(ApproxDimension_);
            const int numDocs = cursor.shape(-1);

            auto probs = ComputeSoftmax(cursor, numDocs);  // [K, numDocs]

            // For each doc, get probability of the target class
            // Use gather: prob_target[d] = probs[targets[d], d]
            auto targetInt = mx::astype(targets, mx::int32);

            // Build prob of target class: iterate dims and select
            auto probTarget = mx::zeros({numDocs}, mx::float32);
            for (int k = 0; k < K; ++k) {
                auto isClass = mx::astype(
                    mx::equal(targetInt, mx::array(k)),
                    mx::float32
                );  // [numDocs]
                auto probK = mx::slice(probs, {k, 0}, {k + 1, numDocs});
                probK = mx::reshape(probK, {numDocs});
                probTarget = mx::add(probTarget, mx::multiply(isClass, probK));
            }

            // Add probability for the implicit K-th class
            auto isLastClass = mx::astype(
                mx::equal(targetInt, mx::array(K)),
                mx::float32
            );
            // Implicit class prob = 1 - sum(probs, axis=0)
            auto implicitProb = mx::subtract(
                mx::array(1.0f),
                mx::sum(probs, /*axis=*/0)
            );  // [numDocs]
            probTarget = mx::add(probTarget, mx::multiply(isLastClass, implicitProb));

            auto eps = mx::array(1e-15f);
            auto loss = mx::negative(mx::mean(
                mx::multiply(weights, mx::log(mx::add(probTarget, eps)))
            ));
            // No EvalNow — caller uses .item<float>() which forces evaluation.
            return loss;
        }

    private:
        ui32 NumClasses_;
        ui32 ApproxDimension_;

        // Compute softmax with implicit K-th class (value 0).
        // Input cursor: [K, numDocs]. Returns [K, numDocs] probabilities.
        mx::array ComputeSoftmax(const mx::array& cursor, int numDocs) const {
            const int K = static_cast<int>(ApproxDimension_);

            // Numerical stability: max over dims and implicit 0
            auto maxCursor = mx::max(cursor, /*axis=*/0);  // [numDocs]
            maxCursor = mx::maximum(maxCursor, mx::array(0.0f));  // account for implicit class

            // exp(cursor - max) for each dim
            auto expCursor = mx::exp(mx::subtract(cursor, mx::reshape(maxCursor, {1, numDocs})));
            // exp(-max) for implicit K-th class
            auto expImplicit = mx::exp(mx::negative(maxCursor));  // [numDocs]

            // sum of exponentials
            auto sumExp = mx::add(mx::sum(expCursor, /*axis=*/0), expImplicit);  // [numDocs]

            // Probabilities: exp / sum
            auto probs = mx::divide(expCursor, mx::reshape(sumExp, {1, numDocs}));  // [K, numDocs]
            return probs;
        }
    };

    // MAE (Mean Absolute Error) / L1 loss.
    // Loss = mean(|pred - target|)
    // Gradient = sign(pred - target) (subgradient)
    // Hessian = 1.0 (constant — needed for Newton step denominator)
    class TMAETarget : public IMLXTargetFunc {
    public:
        ui32 GetApproxDimension() const override { return 1; }

        void ComputeDerivatives(
            const mx::array& cursor,
            const mx::array& targets,
            const mx::array& weights,
            mx::array& gradients,
            mx::array& hessians
        ) const override {
            gradients = mx::sign(mx::subtract(cursor, targets));
            gradients = mx::multiply(gradients, weights);
            hessians = mx::copy(weights);
            // No EvalNow — consumed lazily by histogram kernel.
        }

        mx::array ComputeLoss(
            const mx::array& cursor,
            const mx::array& targets,
            const mx::array& weights
        ) const override {
            auto loss = mx::mean(mx::multiply(
                mx::abs(mx::subtract(cursor, targets)), weights));
            // No EvalNow — caller uses .item<float>() which forces evaluation.
            return loss;
        }
    };

    // Quantile regression loss.
    // Asymmetric L1 loss parameterized by alpha ∈ (0, 1).
    // alpha = 0.5 reduces to MAE/median regression.
    // Gradient = (1 - alpha) if pred > target, else -alpha
    // Hessian = 1.0 (constant)
    class TQuantileTarget : public IMLXTargetFunc {
    public:
        explicit TQuantileTarget(float alpha = 0.5f) : Alpha_(alpha) {}

        ui32 GetApproxDimension() const override { return 1; }

        void ComputeDerivatives(
            const mx::array& cursor,
            const mx::array& targets,
            const mx::array& weights,
            mx::array& gradients,
            mx::array& hessians
        ) const override {
            auto diff = mx::subtract(cursor, targets);
            auto isPositive = mx::greater(diff, mx::array(0.0f));
            gradients = mx::where(isPositive,
                mx::array(1.0f - Alpha_), mx::array(-Alpha_));
            gradients = mx::multiply(gradients, weights);
            hessians = mx::copy(weights);
            // No EvalNow — consumed lazily by histogram kernel.
        }

        mx::array ComputeLoss(
            const mx::array& cursor,
            const mx::array& targets,
            const mx::array& weights
        ) const override {
            auto diff = mx::subtract(targets, cursor);
            auto isPositive = mx::greater(diff, mx::array(0.0f));
            auto loss = mx::where(isPositive,
                mx::multiply(mx::array(Alpha_), diff),
                mx::multiply(mx::array(Alpha_ - 1.0f), diff));
            auto result = mx::mean(mx::multiply(loss, weights));
            // No EvalNow — caller uses .item<float>() which forces evaluation.
            return result;
        }

    private:
        float Alpha_;
    };

    // Huber loss (smooth approximation of L1).
    // Quadratic for |diff| <= delta, linear for |diff| > delta.
    // Gradient = diff if |diff| <= delta, else delta * sign(diff)
    // Hessian = weight if |diff| <= delta, else small epsilon * weight
    class THuberTarget : public IMLXTargetFunc {
    public:
        explicit THuberTarget(float delta = 1.0f) : Delta_(delta) {}

        ui32 GetApproxDimension() const override { return 1; }

        void ComputeDerivatives(
            const mx::array& cursor,
            const mx::array& targets,
            const mx::array& weights,
            mx::array& gradients,
            mx::array& hessians
        ) const override {
            auto diff = mx::subtract(cursor, targets);
            auto absDiff = mx::abs(diff);
            auto isSmall = mx::less_equal(absDiff, mx::array(Delta_));
            gradients = mx::where(isSmall, diff,
                mx::multiply(mx::array(Delta_), mx::sign(diff)));
            gradients = mx::multiply(gradients, weights);
            hessians = mx::where(isSmall, mx::copy(weights),
                mx::multiply(mx::array(1e-6f), weights));
            // No EvalNow — consumed lazily by histogram kernel.
        }

        mx::array ComputeLoss(
            const mx::array& cursor,
            const mx::array& targets,
            const mx::array& weights
        ) const override {
            auto diff = mx::subtract(cursor, targets);
            auto absDiff = mx::abs(diff);
            auto isSmall = mx::less_equal(absDiff, mx::array(Delta_));
            auto loss = mx::where(isSmall,
                mx::multiply(mx::array(0.5f), mx::multiply(diff, diff)),
                mx::subtract(
                    mx::multiply(mx::array(Delta_), absDiff),
                    mx::array(0.5f * Delta_ * Delta_)));
            auto result = mx::mean(mx::multiply(loss, weights));
            // No EvalNow — caller uses .item<float>() which forces evaluation.
            return result;
        }

    private:
        float Delta_;
    };

    // Poisson regression (log-link).
    // pred = exp(cursor), so cursor is the log-space prediction.
    // Gradient = exp(cursor) - target
    // Hessian  = exp(cursor)  [always positive; clamped for stability]
    // Loss     = mean(w * (exp(cursor) - target * cursor))
    class TPoissonTarget : public IMLXTargetFunc {
    public:
        ui32 GetApproxDimension() const override { return 1; }

        void ComputeDerivatives(
            const mx::array& cursor,
            const mx::array& targets,
            const mx::array& weights,
            mx::array& gradients,
            mx::array& hessians
        ) const override {
            auto expPred = mx::exp(cursor);

            // Gradient = (exp(cursor) - target) * weight
            gradients = mx::multiply(mx::subtract(expPred, targets), weights);

            // Hessian = exp(cursor) * weight, clamped for numerical stability
            hessians = mx::maximum(
                mx::multiply(expPred, weights),
                mx::array(1e-6f)
            );

            // No EvalNow — consumed lazily by histogram kernel.
        }

        mx::array ComputeLoss(
            const mx::array& cursor,
            const mx::array& targets,
            const mx::array& weights
        ) const override {
            // NLL: mean(w * (exp(cursor) - target * cursor))
            auto expPred = mx::exp(cursor);
            auto loss = mx::mean(mx::multiply(
                mx::subtract(expPred, mx::multiply(targets, cursor)),
                weights
            ));
            // No EvalNow — caller uses .item<float>() which forces evaluation.
            return loss;
        }
    };

    // Tweedie regression (log-link with power variance p, typically 1 < p < 2).
    // pred = exp(cursor).
    // Gradient = exp((2-p)*cursor) - target * exp((1-p)*cursor)
    // Hessian  = (2-p)*exp((2-p)*cursor) - target*(1-p)*exp((1-p)*cursor)  [clamped]
    // Loss     = mean(w * (-target*exp((1-p)*cursor)/(1-p) + exp((2-p)*cursor)/(2-p)))
    class TTweedieTarget : public IMLXTargetFunc {
    public:
        explicit TTweedieTarget(float p = 1.5f) : P_(p) {}

        ui32 GetApproxDimension() const override { return 1; }

        void ComputeDerivatives(
            const mx::array& cursor,
            const mx::array& targets,
            const mx::array& weights,
            mx::array& gradients,
            mx::array& hessians
        ) const override {
            const float p = P_;
            // exp((2-p)*cursor) and exp((1-p)*cursor)
            auto exp2mp = mx::exp(mx::multiply(cursor, mx::array(2.0f - p)));
            auto exp1mp = mx::exp(mx::multiply(cursor, mx::array(1.0f - p)));

            // Gradient = exp((2-p)*cursor) - target * exp((1-p)*cursor)
            auto rawGrad = mx::subtract(exp2mp, mx::multiply(targets, exp1mp));
            gradients = mx::multiply(rawGrad, weights);

            // Hessian (full second derivative of Tweedie deviance):
            //   (2-p)*exp((2-p)*cursor) - target*(1-p)*exp((1-p)*cursor)
            auto rawHess = mx::subtract(
                mx::multiply(mx::array(2.0f - p), exp2mp),
                mx::multiply(mx::multiply(targets, mx::array(1.0f - p)), exp1mp)
            );
            hessians = mx::maximum(
                mx::multiply(rawHess, weights),
                mx::array(1e-6f)
            );

            // No EvalNow — consumed lazily by histogram kernel.
        }

        mx::array ComputeLoss(
            const mx::array& cursor,
            const mx::array& targets,
            const mx::array& weights
        ) const override {
            const float p = P_;
            // Tweedie deviance: -target*exp((1-p)*cursor)/(1-p) + exp((2-p)*cursor)/(2-p)
            auto term1 = mx::divide(
                mx::multiply(mx::negative(targets),
                             mx::exp(mx::multiply(cursor, mx::array(1.0f - p)))),
                mx::array(1.0f - p)
            );
            auto term2 = mx::divide(
                mx::exp(mx::multiply(cursor, mx::array(2.0f - p))),
                mx::array(2.0f - p)
            );
            auto loss = mx::mean(mx::multiply(mx::add(term1, term2), weights));
            // No EvalNow — caller uses .item<float>() which forces evaluation.
            return loss;
        }

    private:
        float P_;  // variance power, typically 1 < P < 2
    };

    // MAPE (Mean Absolute Percentage Error).
    // Gradient = sign(cursor - target) / max(|target|, epsilon)
    // Hessian  = 1 / max(|target|, epsilon)   [constant per sample]
    // Loss     = mean(w * |cursor - target| / max(|target|, epsilon))
    class TMAPETarget : public IMLXTargetFunc {
    public:
        ui32 GetApproxDimension() const override { return 1; }

        void ComputeDerivatives(
            const mx::array& cursor,
            const mx::array& targets,
            const mx::array& weights,
            mx::array& gradients,
            mx::array& hessians
        ) const override {
            auto absTarget = mx::maximum(mx::abs(targets), mx::array(1e-6f));

            // Gradient = sign(cursor - target) / |target| * weight
            gradients = mx::multiply(
                mx::divide(mx::sign(mx::subtract(cursor, targets)), absTarget),
                weights
            );

            // Hessian = 1 / |target| * weight
            hessians = mx::multiply(
                mx::divide(mx::ones_like(targets), absTarget),
                weights
            );

            // No EvalNow — consumed lazily by histogram kernel.
        }

        mx::array ComputeLoss(
            const mx::array& cursor,
            const mx::array& targets,
            const mx::array& weights
        ) const override {
            auto absTarget = mx::maximum(mx::abs(targets), mx::array(1e-6f));
            auto loss = mx::mean(mx::multiply(
                mx::divide(mx::abs(mx::subtract(cursor, targets)), absTarget),
                weights
            ));
            // No EvalNow — caller uses .item<float>() which forces evaluation.
            return loss;
        }
    };

}  // namespace NCatboostMlx
