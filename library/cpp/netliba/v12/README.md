# Netliba v12

Netliba v12 is a high-performance RPC/transport library built on raw UDP (with an InfiniBand/RDMA fast path) rather than on TCP. It implements its own reliable-delivery layer: fragmentation/reassembly, ACK tracking, retransmission, and congestion control — essentially "TCP done right" for datacenter workloads, plus features that TCP fundamentally cannot offer. It is exposed to higher-level frameworks via the `neh` transport library.

## Architecture (three layers)

1. **Socket layer** — `udp_socket.h`: a thin wrapper over UDP sockets with `sendmmsg`/`recvmmsg` batching, per-packet TOS set via `sendmsg` auxiliary data (no `setsockopt` syscall per packet), and coalescing of consecutive small packets into a single UDP datagram.
2. **Reliable host layer** — `udp_host.h`: the `IUdpHost` interface with `Connect()`, `Send()`, transfer cancellation, and connection GUIDs identifying each connection. Congestion control lives inside `TConnection` (`udp_host_connection.h`).
3. **Request/response layer** — `udp_http.h`: an HTTP-like request/response API with `TColors` for per-request QoS (separate TOS for request data, response data, and their ACKs, plus packet priority).

## What it has that other network libraries don't

- **Custom congestion control that beats TCP in the datacenter.** Internal benchmarks (~20 hosts, all-to-all transfers) showed netliba significantly outperforming TCP, whose window scaling and congestion avoidance are poorly suited to many-to-many cluster traffic. ACKs can even be sent with a *different TOS* so they aren't delayed behind the host's outgoing data traffic.
- **InfiniBand/RDMA support.** The `ib_*` files (`ib_low.h`, `ib_mem`, `ib_memstream`, `ib_collective`) wrap `libibverbs` directly (see the `contrib/libs/ibdrv` dependency). When peers share an IB fabric, netliba transparently switches to RDMA with zero-copy memory registration — something no TCP-based library (gRPC, plain HTTP) can do.
- **2 file descriptors for any number of peers/requests.** HTTP-family protocols burn FDs on keepalive connection pools, while netliba multiplexes everything over a single socket pair — at the cost of higher memory usage.
- **Per-packet QoS "coloring" without syscalls.** `TColors` lets you set DSCP/TOS separately for request data, response data, and each of their ACKs, encoded in the packet header via `sendmsg` aux data — you can "paint" every packet without a syscall. gRPC/HTTP provide no per-request traffic-class control.
- **Batching and small-packet coalescing.** `sendmmsg`/`recvmmsg` (receive queue of 128), plus merging consecutive small packets to one address into a single UDP datagram — measured as a multi-fold speedup for small packets in socket tests.
- **Memory-efficient exactly-once delivery tracking.** Instead of a multi-megabyte hashset of received transfer GUIDs per connection, v12 stores only the "holes" (unreceived transfers) in interval trees plus a 1024-entry circular buffer of recent transfer states — nearly free in the common case.
- **Robust restart/handshake semantics.** Connection GUID + ReceiverGUID in every packet guarantee that responses are matched to the right connection and that data is delivered exactly once to exactly one instance, even across receiver restarts — a known weak spot of connectionless UDP designs.
- **Packet priorities** (`EPacketPriority`) and per-IP bandwidth caps / slow start (`SetUdpMaxBandwidthPerIP`, `SetUdpSlowStart`) for cluster-friendliness.

## Trade-offs

- Not session-oriented: it doesn't immediately tell you when the peer restarted/disconnected (handled via timeouts and GUID checks instead).
- Higher memory consumption than socket-pool-based transports.
- UDP-based, so it depends on datacenter network configuration (netliba assumes packet loss signals congestion rather than corruption).

## Summary

Netliba v12 exists because TCP's congestion control and connection model are suboptimal for datacenter workloads such as MapReduce shuffles and all-to-all transfers. Its unique combination — user-space reliable UDP + custom congestion control + an RDMA fast path + per-packet QoS coloring + O(1) file-descriptor usage — is not available in gRPC, plain HTTP, or other generic network libraries.
