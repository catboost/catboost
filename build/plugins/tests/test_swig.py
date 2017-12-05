from build.plugins import swig


def test_include_parser():
    swg_file = """
%module lemmer

%include <typemaps.i>
%include "defaults.swg"

%header %{
#include <util/generic/string.h>
#include "util/generic/vector.h"
%}

    %import <contrib/swiglib/stroka.swg>

%{
    #include "py_api.h"
%}

%begin %{
    #pragma clang diagnostic ignored "-Wself-assign"
%}

%insert(init) "1.swg"
%insert(runtime) "2.swg";
%insert("init") "3.swg"
%insert("runtime") "4.swg";
%insert (klsdgjsdflksdf) "5.swg";
%insert   ("lkgjkdfg") "6.swg";

%insert   ("lkgjkdfg") "7.swg"; // )

%insert("header") {
}

%insert(runtime) %{
}%

"""

    includes, induced = swig.SwigParser.parse_includes(swg_file.split('\n'))
    assert includes == ['typemaps.i', 'defaults.swg', 'contrib/swiglib/stroka.swg', '1.swg', '2.swg', '3.swg', '4.swg', '5.swg', '6.swg', '7.swg']
    assert induced == ['util/generic/string.h', 'util/generic/vector.h', 'py_api.h']
