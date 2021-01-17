#pragma once

#include <catboost/private/libs/data_types/groupid.h>

#include <util/generic/string.h>


/* Need this version because CatBoost's CalcGroupIdFor uses TStringBuf which is not JVM-mapped type here
 * and change return type to Java-compatible i64
 */
inline i64 CalcGroupIdForString(const TString& token) {
    return CalcGroupIdFor(token);
}
