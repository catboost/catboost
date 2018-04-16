#pragma once

namespace NNetliba {
    void RunUdpTest(bool client, const char* serverName, int packetSize, int packetsInFly, int srcPort = 0);
}
