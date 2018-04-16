#include "rpc.h"
#include "multi.h"
#include "location.h"
#include "factory.h"

#include <util/string/cast.h>
#include <util/generic/hash.h>

using namespace NNeh;

namespace {
    namespace NMulti {
        class TRequester: public IRequester {
        public:
            inline TRequester(const TListenAddrs& addrs, IOnRequest* cb) {
                for (const auto& addr : addrs) {
                    TParsedLocation loc(addr);
                    IRequesterRef& req = R_[ToString(loc.Scheme) + ToString(loc.GetPort())];

                    if (!req) {
                        req = ProtocolFactory()->Protocol(loc.Scheme)->CreateRequester(cb, loc);
                    }
                }
            }

        private:
            typedef THashMap<TString, IRequesterRef> TRequesters;
            TRequesters R_;
        };
    }
}

IRequesterRef NNeh::MultiRequester(const TListenAddrs& addrs, IOnRequest* cb) {
    return new NMulti::TRequester(addrs, cb);
}
