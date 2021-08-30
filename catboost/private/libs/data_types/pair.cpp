#include "pair.h"

#include <util/stream/output.h>
#include <util/stream/str.h>


void OutputHumanReadable(const TPair& pair, IOutputStream* out) {
    (*out) << "(WinnerId=" << pair.WinnerId
           << ",LoserId=" << pair.LoserId
           << ",Weight=" << pair.Weight << ')';
}

template <>
void Out<TPair>(IOutputStream& out, const TPair& pair) {
    OutputHumanReadable(pair, &out);
}


TString HumanReadableDescription(const TPair& pair) {
    TStringStream out;
    OutputHumanReadable(pair, &out);
    return out.Str();
}
