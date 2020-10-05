%{
#include <util/generic/vector.h>
%}
%include <typemaps.i>
%include <std_vector.i>

namespace std {
    %template(IntVector) vector<int>;
};

%apply std::vector<int> { TVector<int> };
