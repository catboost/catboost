#pragma once

#include <library/cpp/binsaver/bin_saver.h>

#include <util/generic/vector.h>
#include <util/ysaveload.h>


class TTargetClassifier {
public:
    TTargetClassifier() = default;

    explicit TTargetClassifier(const TVector<float>& borders)
        : Borders(borders)
    {
    }
    bool operator==(const TTargetClassifier& other) const {
        return Borders == other.Borders;
    }
    SAVELOAD(Borders);
    Y_SAVELOAD_DEFINE(Borders);

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

private:
    TVector<float> Borders;
};
