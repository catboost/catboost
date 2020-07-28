#include "serialization.h"
#include "exception.h"

#include <library/cpp/json/json_value.h>

#include <util/stream/str.h>


namespace NCB {
    void AddPadding(TCountingOutput* const output, const ui32 alignment) {
        if (output->Counter() % alignment == 0) {
            return;
        }

        const auto bytesToWrite = alignment - (output->Counter() % alignment);
        for (ui64 i = 0; i < bytesToWrite; ++i) {
            output->Write('\0');
        }
    }

    void SkipPadding(TCountingInput* const input, const ui32 alignment) {
        if (input->Counter() % alignment == 0) {
            return;
        }

        const auto bytesToSkip = alignment - (input->Counter() % alignment);
        const auto bytesSkipped = input->Skip(bytesToSkip);
        CB_ENSURE(bytesToSkip == bytesSkipped);
    }

    void WriteMagic(const char* magic, ui32 magicSize, ui32 alignment, IOutputStream* stream) {
        TCountingOutput output(stream);
        output.Write(magic, magicSize);
        AddPadding(&output, alignment);
        Y_ASSERT(output.Counter() % alignment == 0);
    }

    void ReadMagic(const char* expectedMagic, ui32 magicSize, ui32 alignment, IInputStream* stream) {
        TCountingInput input(stream);
        TArrayHolder<char> loadedMagic = TArrayHolder<char>(new char[magicSize]);
        ui32 loadedBytes = input.Load(loadedMagic.Get(), magicSize);
        CB_ENSURE(
            loadedBytes == magicSize && Equal(loadedMagic.Get(), loadedMagic.Get() + magicSize, expectedMagic),
            "Failed to deserialize: couldn't read magic"
        );
        SkipPadding(&input, alignment);
    }

}


int operator&(NJson::TJsonValue& jsonValue, IBinSaver& binSaver) {
    TString serializedData;
    if (binSaver.IsReading()) {
        binSaver.Add(0, &serializedData);
        TStringInput in(serializedData);
        jsonValue.Load(&in);
    } else {
        TStringOutput out(serializedData);
        jsonValue.Save(&out);
        out.Finish();
        binSaver.Add(0, &serializedData);
    }

    return 0;
}
