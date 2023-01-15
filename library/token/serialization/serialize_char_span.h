#pragma once

struct TCharSpan;

namespace NProto {
    class TCharSpan;
}

void SerializeCharSpan(const TCharSpan& span, NProto::TCharSpan& message);
void DeserializeCharSpan(TCharSpan& span, const NProto::TCharSpan& message);
