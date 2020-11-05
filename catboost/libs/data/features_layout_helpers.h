#pragma once

#include "features_layout.h"
#include <catboost/libs/model/model.h>

NCB::TFeaturesLayout MakeFeaturesLayout(const TFullModel& model);
