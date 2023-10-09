#include <contrib/libs/protobuf-mutator/src/libfuzzer/libfuzzer_macro.h>
#include <google/protobuf/stubs/logging.h>

#include <library/cpp/blockcodecs/codecs.h>
#include <library/cpp/blockcodecs/fuzz/proto/case.pb.h>
#include <library/cpp/blockcodecs/stream.h>

#include <util/stream/input.h>
#include <util/stream/length.h>
#include <util/stream/mem.h>
#include <util/stream/null.h>
#include <util/stream/str.h>

using NBlockCodecs::NFuzz::TPackUnpackCase;
using NBlockCodecs::TCodedOutput;
using NBlockCodecs::TDecodedInput;

static void ValidateBufferSize(const ui32 size) {
    Y_ENSURE(size > 0 && size <= 16ULL * 1024);
}

static void DoOnlyDecode(const TPackUnpackCase& case_) {
    if (!case_.GetPacked()) {
        return;
    }

    TMemoryInput mi(case_.GetData().data(), case_.GetData().size());
    TDecodedInput di(&mi);
    TNullOutput no;
    TCountingOutput cno(&no);
    TransferData(&di, &cno);
}

static void DoDecodeEncode(const TPackUnpackCase& case_) {
    auto* const codec = NBlockCodecs::Codec(case_.GetCodecName());
    Y_ENSURE(codec);

    TMemoryInput mi(case_.GetData().data(), case_.GetData().size());
    TDecodedInput di(&mi, codec);
    TStringStream decoded;
    TransferData(&di, &decoded);
    TNullOutput no;
    TCountingOutput cno(&no);
    TCodedOutput co(&cno, codec, case_.GetBufferLength());
    TransferData(&decoded, &co);
    co.Flush();

    Y_ABORT_UNLESS((case_.GetData().size() > 0) == (cno.Counter() > 0));
    Y_ABORT_UNLESS((case_.GetData().size() > 0) == (decoded.Str().size() > 0));
}

static void DoEncodeDecode(const TPackUnpackCase& case_) {
    auto* const codec = NBlockCodecs::Codec(case_.GetCodecName());
    Y_ENSURE(codec);

    TMemoryInput mi(case_.GetData().data(), case_.GetData().size());
    TStringStream encoded;
    TCodedOutput co(&encoded, codec, case_.GetBufferLength());
    TransferData(&mi, &co);
    co.Flush();
    TStringStream decoded;
    TDecodedInput di(&encoded, codec);
    TransferData(&di, &decoded);

    Y_ABORT_UNLESS((case_.GetData().size() > 0) == (encoded.Str().size() > 0));
    Y_ABORT_UNLESS(case_.GetData() == decoded.Str());
}

DEFINE_BINARY_PROTO_FUZZER(const TPackUnpackCase& case_) {
    try {
        if (!case_.GetCodecName()) {
            DoOnlyDecode(case_);
            return;
        }

        ValidateBufferSize(case_.GetBufferLength());
        if (case_.GetPacked()) {
            DoDecodeEncode(case_);
        } else {
            DoEncodeDecode(case_);
        }
    } catch (const std::exception&) {
    }
}
