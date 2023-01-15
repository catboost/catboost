#pragma once

namespace NNetliba_v12 {
    void RunUdpTest(bool client, const char* serverName, int packetSize, int packetsInFly, int srcPort = 0);
}
