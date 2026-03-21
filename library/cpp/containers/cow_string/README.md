## Copy-on-Write based string implementation

Drop in replacement for TSring in the code which deeply relies on COW-semantic.

 * `#include <library/cpp/containers/cow_string/cow_string.h>` main header of the library with the COW-string class itself
 * `#include <library/cpp/containers/cow_string/reverse.h>` in-place strings reverse implementation
 * `#include <library/cpp/containers/cow_string/str_stl.h>` comparator/hashers/... template specialization allowing to use TCowString in tree-based or hash-based sets/maps.
 * `#include <library/cpp/containers/cow_string/subst.h>` TCowString implementation of the substitution function provided for TString in `<util/string/subst.h>`.
 * `#include <library/cpp/containers/cow_string/ysaveload.h>` TCowString support of the `<util/ysaveload.h>` serialization/deserialization.
