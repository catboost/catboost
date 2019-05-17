#pragma once

#include <util/generic/fwd.h>

#include <stlfwd>

/* Replace all occurences of substring `what` with string `with` starting from position `from`.
 *
 * @param text      String to modify.
 * @param what      Substring to replace.
 * @param with      Substring to use as replacement.
 * @param from      Position at with to start replacement.
 *
 * @return          Number of replacements occured.
 */
size_t SubstGlobal(TString& text, TStringBuf what, TStringBuf with, size_t from = 0);
size_t SubstGlobal(std::string& text, TStringBuf what, TStringBuf with, size_t from = 0);
size_t SubstGlobal(TUtf16String& text, TWtringBuf what, TWtringBuf with, size_t from = 0);
size_t SubstGlobal(std::u16string& text, TWtringBuf what, TWtringBuf with, size_t from = 0);
size_t SubstGlobal(TUtf32String& text, TUtf32StringBuf what, TUtf32StringBuf with, size_t from = 0);

/* Replace all occurences of character `what` with character `with` starting from position `from`.
 *
 * @param text      String to modify.
 * @param what      Character to replace.
 * @param with      Character to use as replacement.
 * @param from      Position at with to start replacement.
 *
 * @return          Number of replacements occured.
 */
size_t SubstGlobal(TString& text, char what, char with, size_t from = 0);
size_t SubstGlobal(std::string& text, char what, char with, size_t from = 0);
size_t SubstGlobal(TUtf16String& text, wchar16 what, wchar16 with, size_t from = 0);
size_t SubstGlobal(std::u16string& text, wchar16 what, wchar16 with, size_t from = 0);
size_t SubstGlobal(TUtf32String& text, wchar32 what, wchar32 with, size_t from = 0);

// TODO(yazevnul):
// - rename `SubstGlobal` to `ReplaceAll` for convenience
// - add `SubstGlobalCopy(TStringBuf)` for convenience
// - add `RemoveAll(text, what, from)` as a shortcut for `SubstGlobal(text, what, "", from)`
// - rename file to `replace.h`

/* Replace all occurences of substring or character `what` with string or character `with` starting from position `from`, and return result string.
 *
 * @param text      String to modify.
 * @param what      Substring/character to replace.
 * @param with      Substring/character to use as replacement.
 * @param from      Position at with to start replacement.
 *
 * @return          Result string
 */
template <class TStringType, class TPatternType>
Y_WARN_UNUSED_RESULT TStringType SubstGlobalCopy(TStringType result, TPatternType what, TPatternType with, size_t from = 0) {
    SubstGlobal(result, what, with, from);
    return result;
}
