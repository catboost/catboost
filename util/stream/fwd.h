#pragma once

#include <util/system/types.h>

class IInputStream;
class IOutputStream;

class IZeroCopyInput;
class IZeroCopyInputFastReadTo;
class IZeroCopyOutput;

using TStreamManipulator = void (*)(IOutputStream&);

class TLengthLimitedInput;
class TCountingInput;
class TCountingOutput;

class TMemoryInput;
class TMemoryOutput;
class TMemoryWriteBuffer;

class TMultiInput;

class TNullInput;
class TNullOutput;
class TNullIO;

class TPipeBase;
class TPipeInput;
class TPipeOutput;
class TPipedBase;
class TPipedInput;
class TPipedOutput;

class TStringInput;
class TStringOutput;
class TStringStream;

class TTeeOutput;

class TTempBufOutput;

struct TEol;

template <typename TEndOfToken>
class TStreamTokenizer;

enum ETraceLevel: ui8;

class IWalkInput;

struct TZLibError;
struct TZLibCompressorError;
struct TZLibDecompressorError;

namespace ZLib {
    enum StreamType: ui8;
}

class TZLibDecompress;
class TZLibCompress;
class TBufferedZLibDecompress;

using TZDecompress = TBufferedZLibDecompress;

class TAlignedInput;
class TAlignedOutput;

class TBufferInput;
class TBufferOutput;
class TBufferStream;

class TBufferedInput;
class TBufferedOutputBase;
class TBufferedOutput;
class TAdaptiveBufferedOutput;

template <class TSlave>
class TBuffered;

template <class TSlave>
class TAdaptivelyBuffered;

class TDebugOutput;

class TRandomAccessFileInput;
class TRandomAccessFileOutput;
class TBufferedFileOutputEx;

class TUnbufferedFileInput;
class TMappedFileInput;
class TUnbufferedFileOutput;

class TFileInput;
using TIFStream = TFileInput;

class TFixedBufferFileOutput;
using TOFStream = TFixedBufferFileOutput;

using TFileOutput = TAdaptivelyBuffered<TUnbufferedFileOutput>;
