
#pragma SWIG nowarn=801

%rename(ios_base_in) std::ios_base::in;

AUTODOC(cerr, "Standard C++ error stream");
AUTODOC(cout, "Standard C++ output stream");
AUTODOC(cin,  "Standard C++ input stream");
AUTODOC(clog, "Standard C++ logging stream");
AUTODOC(endl,  "Add an end line to stream");
AUTODOC(ends,  "Ends stream");
AUTODOC(flush, "Flush stream");

%include <std/std_ios.i>
