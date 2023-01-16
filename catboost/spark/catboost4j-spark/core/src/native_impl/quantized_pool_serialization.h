#pragma once

#include <catboost/libs/data/data_provider.h>

#include <util/generic/string.h>
#include <util/generic/yexception.h>

// needed for forwarding exceptions from C++ to JVM
void SaveQuantizedPoolWrapper(const NCB::TDataProviderPtr& dataProvider, TString fileName);
