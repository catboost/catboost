%define %pythonabc(Type, Abc)
  %feature("python:abc", #Abc) Type;
%enddef
%pythoncode %{import collections.abc%}
%pythonabc(std::vector, collections.abc.MutableSequence);
%pythonabc(std::list, collections.abc.MutableSequence);
%pythonabc(std::map, collections.abc.MutableMapping);
%pythonabc(std::multimap, collections.abc.MutableMapping);
%pythonabc(std::set, collections.abc.MutableSet);
%pythonabc(std::multiset, collections.abc.MutableSet);
