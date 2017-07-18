#ifndef lzma_util_h_sf6g5as78df6
#define lzma_util_h_sf6g5as78df6

#include "Types.h"
#include "LzmaDec.h"

SRes Decode2(CLzmaDec *state, ISeqOutStream *outStream, ISeqInStream *inStream, UInt64 unpackSize);

#endif
