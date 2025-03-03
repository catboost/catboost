%define %array_class(TYPE,NAME)
  %array_class_wrap(TYPE,NAME,__getitem__,__setitem__)
%enddef

%include <typemaps/carrays.swg>

