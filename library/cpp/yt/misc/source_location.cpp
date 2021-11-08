#include "source_location.h"

#include <string.h>

namespace NYT {

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
