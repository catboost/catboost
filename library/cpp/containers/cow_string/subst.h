#pragma once

#include <library/cpp/containers/cow_string/cow_string.h>

#include <util/string/subst.h>

/* Replace all occurences of substring `what` with string `with` starting from position `from`.
 *
 * @param text      String to modify.
 * @param what      Substring to replace.
 * @param with      Substring to use as replacement.
 * @param from      Position at with to start replacement.
 *
 * @return          Number of replacements occured.
 */
size_t SubstGlobal(TCowString& text, TStringBuf what, TStringBuf with, size_t from = 0);
size_t SubstGlobal(TUtf16CowString& text, TWtringBuf what, TWtringBuf with, size_t from = 0);
size_t SubstGlobal(TUtf32CowString& text, TUtf32StringBuf what, TUtf32StringBuf with, size_t from = 0);

/* Replace all occurences of character `what` with character `with` starting from position `from`.
 *
 * @param text      String to modify.
 * @param what      Character to replace.
 * @param with      Character to use as replacement.
 * @param from      Position at with to start replacement.
 *
 * @return          Number of replacements occured.
 */
size_t SubstGlobal(TCowString& text, char what, char with, size_t from = 0);
size_t SubstGlobal(TUtf16CowString& text, wchar16 what, wchar16 with, size_t from = 0);
size_t SubstGlobal(TUtf32CowString& text, wchar32 what, wchar32 with, size_t from = 0);
