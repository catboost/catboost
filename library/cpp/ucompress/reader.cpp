#include "reader.h"
#include "common.h"

#include <library/cpp/blockcodecs/codecs.h>
#include <library/cpp/json/json_reader.h>

#include <util/system/byteorder.h>


using namespace NUCompress;

TDecodedInput::TDecodedInput(IInputStream* in)
    : S_(in)
{
    Y_ENSURE_EX(S_, TBadArgumentException() << "Null output stream");
}

TDecodedInput::~TDecodedInput() = default;

size_t TDecodedInput::DoUnboundedNext(const void** ptr) {
    if (!C_) {
        TBlockLen blockLen = 0;
        S_->LoadOrFail(&blockLen, sizeof(blockLen));
        blockLen = LittleToHost(blockLen);
        Y_ENSURE(blockLen <= MaxCompressedLen, "broken stream");

        TString buf = TString::Uninitialized(blockLen);
        S_->LoadOrFail(buf.Detach(), blockLen);

        NJson::TJsonValue hdr;
        Y_ENSURE(NJson::ReadJsonTree(buf, &hdr), "cannot parse header, suspect old format");

        auto& codecName = hdr["codec"].GetString();
        Y_ENSURE(codecName, "header does not have codec info");

        // Throws TNotFound
        C_ = NBlockCodecs::Codec(codecName);
        Y_ASSERT(C_);
    }

    TBlockLen blockLen = 0;
    size_t actualRead = S_->Load(&blockLen, sizeof(blockLen));
    if (!actualRead) {
        // End of stream
        return 0;
    }
    Y_ENSURE(actualRead == sizeof(blockLen), "broken stream: cannot read block length");
    blockLen = LittleToHost(blockLen);
    Y_ENSURE(blockLen <= MaxCompressedLen, "broken stream");

    TBuffer block;
    block.Resize(blockLen);
    S_->LoadOrFail(block.Data(), blockLen);

    C_->Decode(block, D_);
    *ptr = D_.Data();
    return D_.Size();
}
