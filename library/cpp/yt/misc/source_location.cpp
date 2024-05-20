#include "source_location.h"

#include <library/cpp/yt/string/format.h>

#include <string.h>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

#ifdef __cpp_lib_source_location

void FormatValue(TStringBuilderBase* builder, const std::source_location& location, TStringBuf /*format*/)
{
    if (location.file_name() != nullptr) {
        builder->AppendFormat(
            "%v:%v:%v",
            location.file_name(),
            location.line(),
            location.column());
    } else {
        builder->AppendString("<unknown>");
    }
}

TString ToString(const std::source_location& location)
{
    return ToStringViaBuilder(location);
}

#endif // __cpp_lib_source_location

////////////////////////////////////////////////////////////////////////////////

const char* TSourceLocation::GetFileName() const
{
    return FileName_;
}

int TSourceLocation::GetLine() const
{
    return Line_;
}

bool TSourceLocation::IsValid() const
{
    return FileName_ != nullptr;
}

bool TSourceLocation::operator<(const TSourceLocation& other) const
{
    const char* fileName = FileName_ ? FileName_ : "";
    const char* otherFileName = other.FileName_ ? other.FileName_ : "";
    int fileNameResult = strcmp(fileName, otherFileName);
    if (fileNameResult != 0) {
        return fileNameResult < 0;
    }

    if (Line_ < other.Line_) {
        return true;
    }
    if (Line_ > other.Line_) {
        return false;
    }

    return false;
}

bool TSourceLocation::operator==(const TSourceLocation& other) const
{
    const char* fileName = FileName_ ? FileName_ : "";
    const char* otherFileName = other.FileName_ ? other.FileName_ : "";
    return
        strcmp(fileName, otherFileName) == 0 &&
        Line_ == other.Line_;
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
