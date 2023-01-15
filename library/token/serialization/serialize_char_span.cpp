#include "serialize_char_span.h"

#include <library/token/token_structure.h>
#include <library/token/serialization/protos/char_span.pb.h>

void SerializeCharSpan(const TCharSpan& span, NProto::TCharSpan& message) {
    message.SetPos(span.Pos);
    message.SetLen(span.Len);
    message.SetSuffixLen(span.SuffixLen);
    message.SetType(span.Type);
    message.SetTokenDelim(span.TokenDelim);
    message.SetPrefixLen(span.PrefixLen);
}

void DeserializeCharSpan(TCharSpan& span, const NProto::TCharSpan& message) {
    span.Pos = message.GetPos();
    span.Len = message.GetLen();
    span.SuffixLen = message.GetSuffixLen();
    span.Type = (ETokenType)message.GetType();
    span.TokenDelim = (ETokenDelim)message.GetTokenDelim();
    span.PrefixLen = message.GetPrefixLen();
}
