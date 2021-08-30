#include "string.h"

TMaybe<TString> MakeMaybeUtf8String(TConstArrayRef<i8> data, i32 length) {
    return MakeMaybe<TString>((const char*)data.data(), length);
}
