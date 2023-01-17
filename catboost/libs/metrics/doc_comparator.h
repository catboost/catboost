#pragma once

template<typename TFloat>
inline bool CompareDocs(double approxLeft, TFloat targetLeft, double approxRight, TFloat targetRight) {
    return approxLeft != approxRight ? approxLeft > approxRight : targetLeft < targetRight;
}
