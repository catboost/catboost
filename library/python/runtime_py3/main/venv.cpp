#include <library/cpp/resource/resource.h>

extern "C" int IsYaIdeVenv() {
    TString dummy;
    return NResource::FindExact("YA_IDE_VENV", &dummy);
}
