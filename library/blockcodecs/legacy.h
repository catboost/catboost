#pragma once

#include "codecs.h"
#include "common.h"

#include <util/generic/ptr.h>

namespace NBlockCodecs {
    TCodecPtr LegacyZStdCodec();
    yvector<TCodecPtr> LegacyZStd06Codec();
}
