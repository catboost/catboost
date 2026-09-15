#pragma once
// Appended before the vector tree search. CUDA source:
// methods/kernel/score_calcers.cuh. Public IDs preserve the shared scalar ABI:
// 4=SolarL2, 5=LOOL2, 6=SatL2. Each calcer ignores the L2 regularizer.
static const char* CBMMetalMulticlassScoresSource = R"METAL(
#include <metal_stdlib>
using namespace metal;
inline void VectorScoreAccumulate(thread float& high, thread float& low, float value) {
    const float sum = high + value;
    const float part = sum - high;
    const float error = (high - (sum - part)) + (value - part);
    const float tail = low + error, next = sum + tail, tail_part = next - sum;
    low = (sum - (next - tail_part)) + (tail - tail_part);
    high = next;
}
inline float2 VectorScoreAdd(float2 a, float2 b) {
    float high = a.x, low = a.y;
    VectorScoreAccumulate(high, low, b.x); VectorScoreAccumulate(high, low, b.y);
    return float2(high, low);
}
inline float2 VectorScoreMultiply(float2 a, float2 b) {
    const float high = a.x * b.x;
    float low = fma(a.x, b.x, -high);
    low += a.x * b.y + a.y * b.x;
    low += a.y * b.y;
    return VectorScoreAdd(float2(high, 0), float2(low, 0));
}
inline float2 VectorScoreDivide(float2 a, float2 b) {
    const float high = a.x / b.x;
    const float2 remainder = VectorScoreAdd(a, -VectorScoreMultiply(b, float2(high, 0)));
    const float low = (remainder.x + remainder.y) / b.x;
    return VectorScoreAdd(float2(high, 0), float2(low, 0));
}
inline float VectorScoreRound(float2 value) { return value.x + value.y; }

inline float VectorScoreSatAdjustment(float weight) {
    if (weight <= 2.0f) return 0.0f;
    float2 numerator, denominator;
    if (weight <= 4.0f) {
        // Both subtractions are exact here. Keep the quadratic residual near
        // (3+sqrt(5))/2: rounding weight² first moves or hides its pole.
        numerator = VectorScoreMultiply(float2(weight, 0), float2(weight - 2.0f, 0));
        denominator = VectorScoreAdd(VectorScoreMultiply(float2(weight, 0),
            float2(weight - 3.0f, 0)), float2(1, 0));
    } else {
        // Algebraically identical to CUDA double w*(w-2)/(w*w-3*w+1),
        // without overflowing float intermediates for otherwise valid weights.
        const float2 inverse = VectorScoreDivide(float2(1, 0), float2(weight, 0));
        numerator = VectorScoreAdd(float2(1, 0), VectorScoreMultiply(float2(-2, 0), inverse));
        denominator = VectorScoreAdd(VectorScoreAdd(float2(1, 0),
            VectorScoreMultiply(float2(-3, 0), inverse)), VectorScoreMultiply(inverse, inverse));
    }
    // CUDA explicitly stores adjust as FLOAT before multiplying its double
    // leaf term. Do not clamp the negative interval immediately above two.
    return VectorScoreRound(VectorScoreDivide(numerator, denominator));
}

inline float VectorScoreAddLeaf(float score, float2 gradient, float weight, uint kind) {
    if ((kind == 4 && weight <= 1.0e-20f) || (kind == 5 && weight <= 1.0f)
        || (kind == 6 && weight <= 2.0f)) return score;
    float2 adjustment;
    if (kind == 4) {
        const float2 x = VectorScoreAdd(float2(weight, 0), float2(1, 0));
        const float2 logarithm = float2(log(x.x), x.y / x.x);
        adjustment = VectorScoreAdd(float2(1, 0), VectorScoreMultiply(float2(2, 0), logarithm));
    } else if (kind == 5) {
        const float2 exactDenominator = VectorScoreAdd(float2(weight, 0), float2(-1, 0));
        float adjust = VectorScoreRound(VectorScoreDivide(float2(weight, 0), exactDenominator));
        adjust *= adjust; // CUDA's SECOND explicit float rounding.
        adjustment = float2(adjust, 0);
    } else {
        adjustment = float2(VectorScoreSatAdjustment(weight), 0);
    }
    if (gradient.x == 0.0f || adjustment.x == 0.0f) return score;
    if (!isfinite(score)) return score;
    // CUDA double retains tiny G² until a large LOO/Sat adjustment restores
    // a representable float score. Normalize every factor before multiplying:
    // merely dividing G by W first does not protect this underflow case.
    int gradientExponent, weightExponent, adjustmentExponent;
    const float gradientMantissa = frexp(gradient.x, gradientExponent);
    const float2 normalizedGradient = float2(gradientMantissa, ldexp(gradient.y, -gradientExponent));
    const float normalizedWeight = frexp(weight, weightExponent);
    const float adjustmentMantissa = frexp(adjustment.x, adjustmentExponent);
    const float2 normalizedAdjustment = float2(adjustmentMantissa, ldexp(adjustment.y, -adjustmentExponent));
    const float2 quotient = VectorScoreDivide(normalizedGradient, float2(normalizedWeight, 0));
    const float2 term = VectorScoreMultiply(VectorScoreMultiply(-normalizedGradient, quotient), normalizedAdjustment);
    const int termExponent = 2 * gradientExponent - weightExponent + adjustmentExponent;
    int scoreExponent = termExponent;
    if (score != 0.0f) frexp(score, scoreExponent);
    const int commonExponent = max(termExponent, scoreExponent);
    const float2 scaledTerm = float2(ldexp(term.x, termExponent - commonExponent),
                                     ldexp(term.y, termExponent - commonExponent));
    // The CUDA calcers have a float Score field, rounded after EVERY leaf.
    return ldexp(VectorScoreRound(VectorScoreAdd(float2(ldexp(score, -commonExponent), 0), scaledTerm)), commonExponent);
}

// Isolated actual-GPU diagnostic for source-calcer thresholds and rounding.
// gradient pairs/weights[candidates,terms], score[candidates], uint4 params:
// {candidates,terms,score_function,0}. Dispatch candidates threads.
kernel void MulticlassExtraScoreMath(
    const device float2* gradients [[buffer(0)]], const device float* weights [[buffer(1)]],
    device float* scores [[buffer(2)]], constant uint4& p [[buffer(3)]],
    uint candidate [[thread_position_in_grid]]) {
    if (candidate >= p.x) return;
    float score = 0.0f;
    for (uint term = 0; term < p.y; ++term) {
        const uint index = candidate * p.y + term;
        score = VectorScoreAddLeaf(score, gradients[index], weights[index], p.z);
    }
    scores[candidate] = score;
}
)METAL";
