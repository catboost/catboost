#ifndef __SRC_LIB_FEASIBILITY_HPP__
#define __SRC_LIB_FEASIBILITY_HPP__

#include "runtime.hpp"
#include "crashsolution.hpp"
#include "feasibility_highs.hpp"
#include "feasibility_quass.hpp"

void computestartingpoint(Runtime& runtime, CrashSolution& result) {
    switch (runtime.settings.phase1strategy) {
        case Phase1Strategy::HIGHS:
            computestartingpoint_highs(runtime, result);
            break;
        case Phase1Strategy::QUASS:
            computestartingpoint_quass(runtime, result);
            break;
    }
}

#endif
