#include "stdafx.h"
#include "ib_low.h"

namespace NNetliba {
    static bool EnableROCEFlag = false;

    void EnableROCE(bool f) {
        EnableROCEFlag = f;
    }

#if defined(_linux_)
    static TMutex IBPortMutex;
    static TIntrusivePtr<TIBPort> IBPort;
    static bool IBWasInitialized;

    TIntrusivePtr<TIBPort> GetIBDevice() {
        TGuard<TMutex> gg(IBPortMutex);
        if (IBWasInitialized) {
            return IBPort;
        }
        IBWasInitialized = true;

        try {
            int rv = ibv_fork_init();
            if (rv != 0) {
                //printf("ibv_fork_init() failed");
                return nullptr;
            }
        } catch (...) {
            //we can not load ib interface, so no ib
            return nullptr;
        }

        TIntrusivePtr<TIBContext> ctx;
        TIntrusivePtr<TIBPort> resPort;
        int numDevices;
        ibv_device** deviceList = ibv_get_device_list(&numDevices);
        //for (int i = 0; i < numDevices; ++i) {
        //    ibv_device *dev = deviceList[i];

        //    printf("Dev %d\n", i);
        //    printf("name:%s\ndev_name:%s\ndev_path:%s\nibdev_path:%s\n",
        //        dev->name,
        //        dev->dev_name,
        //        dev->dev_path,
        //        dev->ibdev_path);
        //    printf("get_device_name(): %s\n", ibv_get_device_name(dev));
        //    ui64 devGuid = ibv_get_device_guid(dev);
        //    printf("ibv_get_device_guid: %" PRIx64 "\n", devGuid);
        //    printf("node type: %s\n", ibv_node_type_str(dev->node_type));
        //    printf("\n");
        //}
        if (numDevices == 1) {
            ctx = new TIBContext(deviceList[0]);
            TIBContext::TLock ibContext(ctx);
            ibv_device_attr devAttrs;
            CHECK_Z(ibv_query_device(ibContext.GetContext(), &devAttrs));

            for (int port = 1; port <= devAttrs.phys_port_cnt; ++port) {
                ibv_port_attr portAttrs;
                CHECK_Z(ibv_query_port(ibContext.GetContext(), port, &portAttrs));
                //ibv_gid myAddress; // ipv6 address of this port;
                //CHECK_Z(ibv_query_gid(ibContext.GetContext(), port, 0, &myAddress));
                //{
                //    ibv_gid p = myAddress;
                //    for (int k = 0; k < 4; ++k) {
                //        DoSwap(p.raw[k], p.raw[7 - k]);
                //        DoSwap(p.raw[8 + k], p.raw[15 - k]);
                //    }
                //    printf("Port %d, address %" PRIx64 ":%" PRIx64 "\n",
                //        port,
                //        p.global.subnet_prefix,
                //        p.global.interface_id);
                //}

                // skip ROCE if flag is not set
                if (portAttrs.lid == 0 && EnableROCEFlag == false) {
                    continue;
                }
                // bind to first active port
                if (portAttrs.state == IBV_PORT_ACTIVE) {
                    resPort = new TIBPort(ctx, port);
                    break;
                }
            }
        } else {
            //printf("%d IB devices found, fail\n", numDevices);
            ctx = nullptr;
        }
        ibv_free_device_list(deviceList);
        IBPort = resPort;
        return IBPort;
    }

    void MakeAH(ibv_ah_attr* res, TPtrArg<TIBPort> port, const TUdpAddress& remoteAddr, const TUdpAddress& localAddr, int serviceLevel) {
        ibv_gid localGid, remoteGid;
        localGid.global.subnet_prefix = localAddr.Network;
        localGid.global.interface_id = localAddr.Interface;
        remoteGid.global.subnet_prefix = remoteAddr.Network;
        remoteGid.global.interface_id = remoteAddr.Interface;

        Zero(*res);
        res->is_global = 1;
        res->port_num = port->GetPort();
        res->sl = serviceLevel;
        res->grh.dgid = remoteGid;
        //res->grh.flow_label = 0;
        res->grh.sgid_index = port->GetGIDIndex(localGid);
        res->grh.hop_limit = 7;
        //res->grh.traffic_class = 0;
    }

#endif
}
