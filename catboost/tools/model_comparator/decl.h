#pragma once

#include <util/generic/maybe.h>
#include <util/generic/strbuf.h>
#include <util/generic/string.h>


namespace NCB {

    template <class TModel>
    TMaybe<TModel> TryLoadModel(TStringBuf filePath);

    // returns true if models are equal
    template <class TModel>
    bool CompareModels(const TModel& model1, const TModel& model2, double diffLimit, TString* diffString);

}
