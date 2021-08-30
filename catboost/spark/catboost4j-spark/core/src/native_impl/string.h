#pragma once

#include <util/generic/array_ref.h>
#include <util/generic/fwd.h>
#include <util/generic/maybe.h>
#include <util/generic/string.h>
#include <util/system/types.h>


// for Hadoop Text format data
TMaybe<TString> MakeMaybeUtf8String(TConstArrayRef<i8> data, i32 length);
