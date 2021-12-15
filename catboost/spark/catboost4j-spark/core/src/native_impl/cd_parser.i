%{
#include <catboost/libs/column_description/cd_parser.h>
%}

%include <bindings/swiglib/stroka.swg>

%include "column.i"
%include "tvector.i"


DECLARE_TVECTOR(TVector_TColumn, TColumn)
