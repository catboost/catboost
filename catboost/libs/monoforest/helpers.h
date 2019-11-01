#pragma once

#include "grid.h"
#include "split.h"
#include "monom.h"
#include "polynom.h"

#include <util/generic/fwd.h>

namespace NMonoForest {
    TString ToHumanReadableString(const TBinarySplit& split, const IGrid& grid);
    TString ToHumanReadableString(const TMonomStructure& structure, const IGrid& grid);
    TString ToHumanReadableString(const TPolynom& polynom, const IGrid& grid);
}
