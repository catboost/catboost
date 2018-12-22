#pragma once

#include <catboost/libs/options/load_options.h>

#include <library/json/json_value.h>

#include <util/generic/string.h>


void ParseCommandLine(int argc, const char* argv[],
                      NJson::TJsonValue* plainJsonPtr,
                      TString* paramPath,
                      NCatboostOptions::TPoolLoadParams* params);
