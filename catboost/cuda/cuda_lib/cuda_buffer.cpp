#include "slice.h"
#include "cuda_buffer.h"
#include <util/stream/output.h>

template <>
void Out<TSlice>(IOutputStream& o, const TSlice& slice) {
    o.Write("[" + ToString(slice.Left) + "-" + ToString(slice.Right) + "]");
}
