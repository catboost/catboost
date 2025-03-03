%define %array_class(TYPE,NAME)
  %array_class_wrap(TYPE,NAME,__paren__,__paren_asgn__)
%enddef

%include <typemaps/carrays.swg>
