%header %{
#include <util/system/defaults.h>
#include <string>
%}

%include <stdint.i>
typedef int8_t i8;
typedef int16_t i16;
typedef int32_t i32;
typedef int64_t i64;
typedef uint8_t ui8;
typedef uint16_t ui16;
typedef uint32_t ui32;
typedef uint64_t ui64;

typedef int64_t time_t;

class yexception;
%typemap(throws) yexception %{
    SWIG_exception_fail(SWIG_RuntimeError, $1.what());
%}

%include "exception.i"
%exception {
    try {
        $action
    } catch (const yexception& e) {
        SWIG_exception(SWIG_RuntimeError, (std::string("C++ std::exception: ") + e.what()).c_str());
    }
    SWIG_CATCH_STDEXCEPT
    catch (...)
    {
        SWIG_exception(SWIG_UnknownError, "C++ anonymous exception");
    }
}


