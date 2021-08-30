#include "factory.h"
#include "udp.h"
#include "netliba.h"
#include "https.h"
#include "http2.h"
#include "inproc.h"
#include "tcp.h"
#include "tcp2.h"

#include <util/generic/hash.h>
#include <util/generic/strbuf.h>
#include <util/generic/singleton.h>

using namespace NNeh;

namespace {
    class TProtocolFactory: public IProtocolFactory, public THashMap<TStringBuf, IProtocol*> {
    public:
        inline TProtocolFactory() {
            Register(NetLibaProtocol());
            Register(Http1Protocol());
            Register(Post1Protocol());
            Register(Full1Protocol());
            Register(UdpProtocol());
            Register(InProcProtocol());
            Register(TcpProtocol());
            Register(Tcp2Protocol());
            Register(Http2Protocol());
            Register(Post2Protocol());
            Register(Full2Protocol());
            Register(SSLGetProtocol());
            Register(SSLPostProtocol());
            Register(SSLFullProtocol());
            Register(UnixSocketGetProtocol());
            Register(UnixSocketPostProtocol());
            Register(UnixSocketFullProtocol());
        }

        IProtocol* Protocol(const TStringBuf& proto) override {
            const_iterator it = find(proto);

            if (it == end()) {
                ythrow yexception() << "unsupported scheme " << proto;
            }

            return it->second;
        }

        void Register(IProtocol* proto) override {
            (*this)[proto->Scheme()] = proto;
        }
    };

    IProtocolFactory* GLOBAL_FACTORY = nullptr;
}

void NNeh::SetGlobalFactory(IProtocolFactory* factory) {
    GLOBAL_FACTORY = factory;
}

IProtocolFactory* NNeh::ProtocolFactory() {
    if (GLOBAL_FACTORY) {
        return GLOBAL_FACTORY;
    }

    return Singleton<TProtocolFactory>();
}
