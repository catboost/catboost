// Enum for kind codes used in Matplotlib Paths.

#ifndef MPL_KIND_CODE
#define MPL_KIND_CODE

namespace contourpy {

typedef enum
{
    MOVETO = 1,
    LINETO = 2,
    CLOSEPOLY = 79
} MplKindCode;

} // namespace contourpy

#endif // MPL_KIND_CODE
