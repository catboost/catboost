#pragma once

#include <util/generic/vector.h>
#include <util/system/types.h>


// TODO(akhropov): will likely be removed after switch to new Pool format. MLTOOLS-140.

TVector<ui32> ToUnsigned(const TVector<int>& src);
