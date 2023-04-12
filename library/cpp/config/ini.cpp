#include "ini.h"

#include <util/string/strip.h>
#include <util/stream/input.h>

using namespace NConfig;

namespace {
    inline TStringBuf StripComment(TStringBuf line) {
        return line.Before('#').Before(';');
    }
}

TConfig NConfig::ParseIni(IInputStream& in) {
    TConfig ret = ConstructValue(TDict());

    {
        TConfig* cur = &ret;
        TString line;

        while (in.ReadLine(line)) {
            TStringBuf tmp = StripComment(line);
            TStringBuf stmp = StripString(tmp);

            if (stmp.empty()) {
                continue;
            }

            if (stmp[0] == '[') {
                //start section
                if (*(stmp.end() - 1) != ']') {
                    ythrow TConfigParseError() << "malformed section " << stmp;
                }

                stmp = TStringBuf(stmp.data() + 1, stmp.end() - 1);
                cur = &ret;

                while (!!stmp) {
                    TStringBuf section;

                    stmp.Split('.', section, stmp);

                    cur = &cur->GetNonConstant<TDict>()[section];
                    if (!cur->IsA<TDict>()) {
                        *cur = ConstructValue(TDict());
                    }
                }
            } else {
                //value
                TStringBuf key, value;

                tmp.Split('=', key, value);

                auto& dict = cur->GetNonConstant<TDict>();
                auto strippedValue = TString(StripString(value));
                dict[StripString(key)] = ConstructValue(strippedValue);
            }
        }
    }

    return ret;
}
