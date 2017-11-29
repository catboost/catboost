#pragma once

#include "codecs.h"
#include "common.h"

#include <util/generic/ptr.h>

namespace NBlockCodecs {
    TVector<TCodecPtr> LegacyZStd06Codec();
}
