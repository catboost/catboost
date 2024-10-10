#include "varint.h"

#include "zigzag.h"

#include <util/generic/yexception.h>
#include <util/generic/ylimits.h>

namespace NYson {
    ////////////////////////////////////////////////////////////////////////////////

    int WriteVarUInt64(IOutputStream* output, ui64 value) {
        bool stop = false;
        int bytesWritten = 0;
        while (!stop) {
            ++bytesWritten;
            ui8 byte = static_cast<ui8>(value | 0x80);
            value >>= 7;
            if (value == 0) {
                stop = true;
                byte &= 0x7F;
            }
            output->Write(byte);
        }
        return bytesWritten;
    }

    int WriteVarInt32(IOutputStream* output, i32 value) {
        return WriteVarUInt64(output, static_cast<ui64>(ZigZagEncode32(value)));
    }

    int WriteVarInt64(IOutputStream* output, i64 value) {
        return WriteVarUInt64(output, static_cast<ui64>(ZigZagEncode64(value)));
    }

    int ReadVarUInt64(IInputStream* input, ui64* value) {
        size_t count = 0;
        ui64 result = 0;

        ui8 byte = 0;
        do {
            if (7 * count > 8 * sizeof(ui64)) {
                ythrow yexception() << "The data is too long to read ui64";
            }
            if (input->Read(&byte, 1) != 1) {
                ythrow yexception() << "The data is too short to read ui64";
            }
            result |= (static_cast<ui64>(byte & 0x7F)) << (7 * count);
            ++count;
        } while (byte & 0x80);

        *value = result;
        return count;
    }

    int ReadVarInt32(IInputStream* input, i32* value) {
        ui64 varInt;
        int bytesRead = ReadVarUInt64(input, &varInt);
        if (varInt > Max<ui32>()) {
            ythrow yexception() << "The data is too long to read i32";
        }
        *value = ZigZagDecode32(static_cast<ui32>(varInt));
        return bytesRead;
    }

    int ReadVarInt64(IInputStream* input, i64* value) {
        ui64 varInt;
        int bytesRead = ReadVarUInt64(input, &varInt);
        *value = ZigZagDecode64(varInt);
        return bytesRead;
    }

} // namespace NYson
