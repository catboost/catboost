#pragma once

#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/options/load_options.h>
#include <catboost/libs/options/cat_feature_options.h>
#include <catboost/libs/options/output_file_options.h>

#include <library/getopt/small/last_getopt.h>
#include <library/json/json_value.h>

#include <util/generic/string.h>
#include <util/string/iterator.h>


void ParseCommandLine(int argc, const char* argv[],
                      NJson::TJsonValue* plainJsonPtr,
                      TString* paramPath,
                      NCatboostOptions::TPoolLoadParams* params);
