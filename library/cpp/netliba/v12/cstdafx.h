#pragma once

#ifdef _WIN32
#pragma warning(disable : 4530 4244 4996)
#include <malloc.h>
#include <util/system/winint.h>
#endif

#include <util/system/defaults.h>
#include <util/system/mutex.h>
#include <util/system/event.h>
#include <library/cpp/deprecated/atomic/atomic.h>
#include <util/system/yassert.h>
#include <util/system/compat.h>

#include <util/ysafeptr.h>

#include <util/stream/output.h>

#include <library/cpp/string_utils/url/url.h>

#include <library/cpp/charset/codepage.h>
#include <library/cpp/charset/recyr.hh>

#include <util/generic/vector.h>
#include <util/generic/hash.h>
#include <util/generic/list.h>
#include <util/generic/hash_set.h>
#include <util/generic/ptr.h>
#include <util/generic/ymath.h>
#include <util/generic/utility.h>
#include <util/generic/algorithm.h>

#include <array>
#include <cstdlib>
#include <cstdio>

namespace NNetliba_v12 {
    typedef unsigned char byte;
    typedef ssize_t yint;
}
