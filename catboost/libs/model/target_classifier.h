#pragma once

#include <library/cpp/binsaver/bin_saver.h>

#include <util/generic/vector.h>
#include <util/ysaveload.h>


class TTargetClassifier {
public:
    TTargetClassifier() = default;

    TTargetClassifier(const TVector<float>& borders, ui32 targetId)
        : TargetId(targetId), Borders(borders)
    {
    }

    bool operator==(const TTargetClassifier& other) const {
        return Borders == other.Borders && TargetId == other.TargetId;
    }
    SAVELOAD(TargetId, Borders);
    Y_SAVELOAD_DEFINE(TargetId, Borders);

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

    ui32 GetTargetId() const {
        return TargetId;
    }

private:
    ui32 TargetId = 0;
    TVector<float> Borders;
};
