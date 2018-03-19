#pragma once

inline bool CompareDocs(double approxLeft, float targetLeft, double approxRight, float targetRight) {
    return approxLeft != approxRight ? approxLeft > approxRight : targetLeft < targetRight;
}
