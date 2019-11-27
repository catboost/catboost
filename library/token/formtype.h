#pragma once

#include <util/generic/fwd.h>
#include <util/system/defaults.h>

enum TFormType {
    fGeneral,
    fExactWord,  //!
    fExactLemma, //!!
    fWeirdExactWord
};

const TString& ToString(TFormType);
bool FromString(const TString& name, TFormType& ret);
