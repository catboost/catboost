import sys

tmpl = """
#include "yabs_mx_calc_table.h"

#include <kernel/matrixnet/mn_sse.h>

#include <library/archive/yarchive.h>

#include <util/memory/blob.h>
#include <util/generic/hash.h>
#include <util/generic/ptr.h>
#include <util/generic/singleton.h>

using namespace NMatrixnet;

extern "C" {
    extern const unsigned char MxFormulas[];
    extern const ui32 MxFormulasSize;
}

namespace {
    struct TFml: public TBlob, public TMnSseInfo {
        inline TFml(const TBlob& b)
            : TBlob(b)
            , TMnSseInfo(Data(), Size())
        {
        }
    };

    struct TFormulas: public yhash<size_t, TAutoPtr<TFml>> {
        inline TFormulas() {
            TBlob b = TBlob::NoCopy(MxFormulas, MxFormulasSize);
            TArchiveReader ar(b);
            %s
        }

        inline const TMnSseInfo& at(size_t n) const throw () {
            return *find(n)->second;
        }
    };

    %s

    static func_descr_t yabs_funcs[] = {
        %s
    };
}

yabs_mx_calc_table_t yabs_mx_calc_table = {YABS_MX_CALC_VERSION, 10000, 0, yabs_funcs};
"""

if __name__ == '__main__':
    init = []
    body = []
    defs = {}

    for i in sys.argv[1:]:
        name = i.replace('.', '_')
        num = long(name.split('_')[1])

        init.append('(*this)[%s] = new TFml(ar.ObjectBlobByKey("%s"));' % (num, '/' + i))

        f1 = 'static void yabs_%s(size_t count, const float** args, double* res) {Singleton<TFormulas>()->at(%s).DoCalcRelevs(args, res, count);}' % (name, num)
        f2 = 'static size_t yabs_%s_factor_count() {return Singleton<TFormulas>()->at(%s).MaxFactorIndex() + 1;}' % (name, num)

        body.append(f1)
        body.append(f2)

        d1 = 'yabs_%s' % name
        d2 = 'yabs_%s_factor_count' % name

        defs[num] = '{%s, %s}' % (d1, d2)

    print tmpl % ('\n'.join(init), '\n\n'.join(body), ',\n'.join((defs.get(i, '{nullptr, nullptr}') for i in range(0, 10000))))
