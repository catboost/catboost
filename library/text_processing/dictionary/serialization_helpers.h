#pragma once

#include <util/stream/file.h>
#include <util/system/byteorder.h>
#include <util/system/unaligned_mem.h>
#include <util/generic/yexception.h>

namespace NTextProcessing::NDictionary {
    static const char MAGIC[] = "MMapDictionary";
    static const size_t MAGIC_SIZE = Y_ARRAY_SIZE(MAGIC);  // yes, with terminating zero

    template <typename T>
    void WriteLittleEndian(const T value, IOutputStream* const output) {
        const auto le = HostToLittle(value);
        output->Write(&le, sizeof(le));
    }

    template <typename T>
    void ReadLittleEndian(T* const value, IInputStream* const input) {
        T le;
        const auto bytesRead = input->Load(&le, sizeof(le));
        Y_ENSURE(bytesRead == sizeof(le));
        *value = LittleToHost(le);
    }

    inline void AddPadding(ui64 bytesToWrite, IOutputStream* const output) {
        for (ui64 i = 0; i < bytesToWrite; ++i) {
            output->Write('\0');
        }
    }

    inline void SkipPadding(ui64 bytesToSkip, IInputStream* const input) {
        const auto bytesSkipped = input->Skip(bytesToSkip);
        Y_ENSURE(bytesToSkip == bytesSkipped);
    }
}
