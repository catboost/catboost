namespace std
{
%callback("%s") endl;
%callback("%s") ends;
%callback("%s") flush;
}

%warnfilter(365) operator+=;
%warnfilter(802) std::basic_iostream;  // turn off multiple inheritance warning

%include <std/std_iostream.i>

