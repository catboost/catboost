import sys

TEMPLATE="""
#include <yql/udfs/common/python/python_udf/python_udf.h>

#include <yql/library/udf/udf_registrator.h>

using namespace NKikimr::NUdf;

#ifdef BUILD_UDF

extern "C" UDF_API void Register(IRegistrator& registrator, ui32 flags)
{
    RegisterYqlPythonUdf(registrator, flags, AsStringBuf("@MODULE_NAME@"), AsStringBuf("@PACKAGE_NAME@"), EPythonFlavor::@FLAVOR@);
    }

    extern "C" UDF_API ui32 AbiVersion()
    {
        return CurrentAbiVersion();
    }

    extern "C" UDF_API void SetBackTraceCallback(TBackTraceCallback callback) {
        SetBackTraceCallbackImpl(callback);
    }
#endif
"""


def main():
    assert len(sys.argv) == 5
    flavor, module_name, package_name, path = sys.argv[1:]
    with open(path, 'w') as f:
        f.write(
            TEMPLATE
            .strip()
            .replace('@MODULE_NAME@', module_name)
            .replace('@PACKAGE_NAME@', package_name)
            .replace('@FLAVOR@', flavor)
        )
        f.write('\n')


if __name__ == "__main__":
    main()
