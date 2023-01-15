#include <library/cpp/resource/resource.h>

#include <stdlib.h>

extern "C" char* GetPyMain() {
    TString res = NResource::Find("PY_MAIN");
    return strdup(res.c_str());
}
