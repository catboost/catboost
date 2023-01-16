#pragma once

#include <catboost/libs/helpers/exception.h>

#include <util/generic/string.h>
#include <util/generic/vector.h>
#include <util/system/types.h>


int ModeFitImpl(const TVector<TString>& args);


void ShutdownWorkers(const TString& hostsFile, i32 timeoutInSeconds);
