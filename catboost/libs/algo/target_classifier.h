#pragma once

#include "params.h"

#include <util/generic/vector.h>
#include <library/grid_creator/binarization.h>

class TTargetClassifier {
public:
    int GetTargetClass(double target) const {
        int resClass = 0;
        while (resClass < Borders.ysize() && target > Borders[resClass]) {
            ++resClass;
        }
        return resClass;
    }

    int GetClassesCount() const {
        return Borders.ysize() + 1;
    }

    explicit TTargetClassifier(const yvector<float>& borders)
        : Borders(borders)
    {
    }

private:
    yvector<float> Borders;
};

TTargetClassifier BuildTargetClassifier(const yvector<float>& target,
                                        int learnSampleCount,
                                        ELossFunction loss,
                                        const TMaybe<TCustomObjectiveDescriptor>& objectiveDescriptor,
                                        int targetBorderCount,
                                        EBorderSelectionType targetBorderType);
