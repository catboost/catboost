#include "vector_helpers.h"

#include <util/generic/cast.h>
#include <util/generic/utility.h>
#include <library/cpp/deprecated/atomic/atomic.h>
#include <util/system/guard.h>

#if defined(__x86_64__)

static void AtomicUpdateMin(double value, double volatile* target) {
    double current = *target;
    double replacement = Min(value, current);
    ui64 bits_current = BitCast<ui64, double>(current);
    ui64 bits_replacement = BitCast<ui64, double>(replacement);
    while (replacement < current && !AtomicCas(reinterpret_cast<ui64 volatile*>(target), bits_replacement, bits_current)) {
        current = *target;
        replacement = Min(value, current);
        bits_current = BitCast<ui64, double>(current);
        bits_replacement = BitCast<ui64, double>(replacement);
    }
}

static void AtomicUpdateMax(double value, double volatile* target) {
    double current = *target;
    double replacement = Max(value, current);
    ui64 bits_current = BitCast<ui64, double>(current);
    ui64 bits_replacement = BitCast<ui64, double>(replacement);
    while (replacement > current && !AtomicCas(reinterpret_cast<ui64 volatile*>(target), bits_replacement, bits_current)) {
        current = *target;
        replacement = Max(value, current);
        bits_current = BitCast<ui64, double>(current);
        bits_replacement = BitCast<ui64, double>(replacement);
    }
}

template <>
void GuardedUpdateMinMax(const TMinMax<double>& value, TMinMax<double> volatile* target, TMutex&) {
    AtomicUpdateMin(value.Min, &target->Min);
    AtomicUpdateMax(value.Max, &target->Max);
}

#else

template <>
void GuardedUpdateMinMax(const TMinMax<double>& value, TMinMax<double> volatile* target, TMutex& guard) {
    with_lock (guard) {
        target->Min = Min(double(target->Min), value.Min);
        target->Max = Max(double(target->Max), value.Max);
    }
}

#endif
