#pragma once

#include "codecs.h"

namespace NBlockCodecs{

    void RegisterCodec(TCodecPtr codec);
    void RegisterAlias(TStringBuf from, TStringBuf to);

}
