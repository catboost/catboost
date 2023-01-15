#pragma once

#include "packers.h"

#include <util/generic/yexception.h>

namespace NPackers {
    template <typename TProtoMessage>
    class TProtoMessagePacker {
    public:
        void UnpackLeaf(const char* bufferPtr, TProtoMessage& protoMessage) const {
            const size_t protoMessageByteSize = GetProtoMessageByteSize(bufferPtr);
            const size_t skipBytesCount = ProtoMessageByteSizePacker.SkipLeaf(bufferPtr);

            if (!protoMessage.ParseFromArray(static_cast<const void*>(bufferPtr + skipBytesCount), protoMessageByteSize)) {
                ythrow yexception() << "Cannot unpack leaf with proto message";
            }
        }

        void PackLeaf(char* bufferPtr, const TProtoMessage& protoMessage, const size_t totalByteSize) const {
            const size_t protoMessageByteSize = protoMessage.ByteSize();
            const size_t skipBytesCount = totalByteSize - protoMessageByteSize;

            ProtoMessageByteSizePacker.PackLeaf(bufferPtr, protoMessageByteSize, skipBytesCount);

            if (!protoMessage.SerializeToArray(static_cast<void*>(bufferPtr + skipBytesCount), protoMessageByteSize)) {
                ythrow yexception() << "Cannot pack leaf with proto message";
            }
        }

        size_t MeasureLeaf(const TProtoMessage& protoMessage) const {
            const size_t protoMessageByteSize = protoMessage.ByteSize();
            return ProtoMessageByteSizePacker.MeasureLeaf(protoMessageByteSize) + protoMessageByteSize;
        }

        size_t SkipLeaf(const char* bufferPtr) const {
            const size_t protoMessageByteSize = GetProtoMessageByteSize(bufferPtr);
            return ProtoMessageByteSizePacker.SkipLeaf(bufferPtr) + protoMessageByteSize;
        }

    private:
        TIntegralPacker<size_t> ProtoMessageByteSizePacker;

        size_t GetProtoMessageByteSize(const char* bufferPtr) const {
            size_t result;
            ProtoMessageByteSizePacker.UnpackLeaf(bufferPtr, result);
            return result;
        }
    };
}
