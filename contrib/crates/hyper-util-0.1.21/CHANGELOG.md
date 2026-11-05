# 0.1.21 (2026-09-24)

This release bumps the minimal supported Rust version (MSRV) from 1.64 to 1.85.

This release bumps the rust edition from 2021 to 2024.

## Additions

- Add crate-level documentation. ([#327](https://github.com/hyperium/hyper-util/pull/327))
- Add `client::legacy::Builder::http2_header_table_size()` method. ([#274](https://github.com/hyperium/hyper-util/pull/274))
- Add `client::legacy::Builder::http2_max_concurrent_streams()` method. ([#274](https://github.com/hyperium/hyper-util/pull/274))
- Add `client::legacy::Builder::http2_max_local_error_reset_streams()` method. ([#277](https://github.com/hyperium/hyper-util/pull/277))
- Add `client::legacy::connect::HttpConnector::set_mark()` method. ([#303](https://github.com/hyperium/hyper-util/pull/303))
- Add `rt::tracing::WithSpanExecutor<E>`, `hyper_util::rt::tracing::CurrentSpanExecutor<E>`, and `hyper_util::rt::tracing::MkSpanExecutor<E, F>` executors. ([#323](https://github.com/hyperium/hyper-util/pull/323))

## Fixes

- Fix `client::legacy::Client` so that it properly validates CONNECT responses. ([#315](https://github.com/hyperium/hyper-util/pull/315))
- Fix `client::legacy::Client` to cancel the idle interval once its pool empties. ([#292](https://github.com/hyperium/hyper-util/pull/292))
- Fix `client::legacy::Client` to properly handle IPv6 addresses when using a SOCKS proxy. ([#302](https://github.com/hyperium/hyper-util/pull/302))
- Fix `client::pool::cache` to preserve readiness with clones. ([#297](https://github.com/hyperium/hyper-util/pull/297))
- Fix `client::pool::cache` to wake its waiters in FIFO order. ([#298](https://github.com/hyperium/hyper-util/pull/298))
- Fix `client::pool::singleton::Singleton` to properly handle cancellation. ([#299](https://github.com/hyperium/hyper-util/pull/299))
- Fix `client::pool::singleton::Singleton` to share errors with all waiters. ([#296](https://github.com/hyperium/hyper-util/pull/296))
- Fix `client::proxy::matcher` handling for IP wildcards. ([#309](https://github.com/hyperium/hyper-util/pull/309))
- The `tokio/net` feature is narrowed to the `client-legacy` feature flag, from the `client` feature flag. ([#276](https://github.com/hyperium/hyper-util/pull/276))
- Various fixes to the `client::legacy::Client`'s SOCKS proxying. ([#302](https://github.com/hyperium/hyper-util/pull/302)) ([#307](https://github.com/hyperium/hyper-util/pull/307)) ([#308](https://github.com/hyperium/hyper-util/pull/308)) ([#310](https://github.com/hyperium/hyper-util/pull/310))

## Changes

This release contains a minor behavioral change for users of the `tracing`
feature flag to be aware of.

This feature flag was introduced in v0.1.11. When enabled,
`rt::TokioExecutor<E>` began propagating the currently active `tracing::Span`
to spawned tasks when `hyper::rt::Executor::execute()` is called. This caused
issues for some users, due to background tasks keeping a span open for the
duration of a long-lived connection.

This behavior has now been removed from `rt::TokioExecutor<E>`
([#322](https://github.com/hyperium/hyper-util/pull/322)) by default. A
collection of executor wrappers have been added to a new `rt::tracing`
submodule, to provide facilities for instrumenting a client or server's spawned
tasks. See the module-level documentation of `rt::tracing` for more
information.

To temporarily preserve the previous `rt::TokioExecutor<E>` span propagation
behavior, enable the `rt-tracing-exec-force` feature. Note that this feature
flag will be removed in a future release.

# 0.1.20 (2026-02-02)

- Fix `proxy::Matcher` to properly match domains regardless of casing
- Fix system proxy matcher dependency on macOS when used in sandboxed environements.
- Increased MSRV to 1.64.

# 0.1.19 (2025-12-03)

- Add `client::pool` module for composable pools. Enable with the `client-pool` feature.
- Add `pool::singleton` for sharing a single cloneable connection.
- Add `pool::cache` for caching a list of connections.
- Add `pool::negotiate` for combining two pools with upgrade and fallback negotiation.
- Add `pool::map` for customizable mapping of keys and connections.

# 0.1.18 (2025-11-13)

- Fix `rt::TokioTimer` to support Tokio's paused time.
- Fix `client::proxy::match::Matcher` to parse auth without passwords.

# 0.1.17 (2025-09-15)

- Fix `legacy::Client` to allow absolute-form URIs when `Connected::proxy(true)` is passed and the scheme is `https`.

# 0.1.16 (2025-07-22)

- Add `impl Clone` for `proxy::Tunnel` service.
- Fix `proxy::Matcher` to detect SOCKS4 schemes.
- Fix `legacy::Client` pool idle checker to trigger less aggresively, saving CPU.

# 0.1.15 (2025-07-07)

- Add header casing options to `auto::Builder`.
- Fix `proxy::Socksv5` to check for enough bytes before parsing ipv6 responses.
- Fix including `client-proxy` in the `full` feature set.

# 0.1.14 (2025-06-04)

- Fix `HttpConnector` to defer address family order to resolver sort order.
- Fix `proxy::Matcher` to find HTTPS system proxies on Windows.

# 0.1.13 (2025-05-27)

- Fix `HttpConnector` to always prefer IPv6 addresses first, if happy eyeballs is enabled.
- Fix `legacy::Client` to return better errors if available on the connection.

# 0.1.12 (2025-05-19)

- Add `client::legacy::proxy::Tunnel` connector that wraps another connector with HTTP tunneling.
- Add `client::legacy::proxy::{SocksV4, SocksV5}` connectors that wraps another connector with SOCKS.
- Add `client::proxy::matcher::Matcher` type that can use environment variables to match proxy rules.
- Add `server::graceful::Watcher` type that can be sent to watch a connection in another task.
- Add `GracefulShutdown::count()` method to get number of currently watched connections.
- Fix missing `must_use` attributes on `Connection` futures.
- Fix tracing span in GAI resolver that can cause panics.


# 0.1.11 (2025-03-31)

- Add `tracing` crate feature with support in `TokioExecutor`.
- Add `HttpConnector::interface()` support for macOS and Solarish systems.
- Add `rt::WithHyperIo` and `rt::WithTokioIo` combinators.
- Add `auto_date_header()` for auto server builder.
- Add `max_local_error_reset_streams()` for auto server builder.
- Add `ignore_invalid_headers()` for auto server builder.
- Add methods to determine if auto server is configured for HTTP/1 or HTTP/2.
- Implement `Connection` for `UnixStream` and `NamedPipeClient`.
- Fix HTTP/2 websocket requests sent through `legacy::Client`.

# 0.1.10 (2024-10-28)

- Add `http2_max_header_list_size(num)` option to legacy client builder.
- Add `set_tcp_user_timeout(dur)` option to legacy `HttpConnector`.

# 0.1.9 (2024-09-24)

- Add support for `client::legacy` DNS resolvers to set non-zero ports on returned addresses.
- Fix `client::legacy` wrongly retrying pooled connections that were created successfully but failed immediately after, resulting in a retry loop.


# 0.1.8 (2024-09-09)

- Add `server::conn::auto::upgrade::downcast()` for use with auto connection upgrades.

# 0.1.7 (2024-08-06)

- Add `Connected::poison()` to `legacy` client, a port from hyper v0.14.x.
- Add `Error::connect_info()` to `legacy` client, a port from hyper v0.14.x.

# 0.1.6 (2024-07-01)

- Add support for AIX operating system to `legacy` client.
- Fix `legacy` client to better use dying pooled connections.

# 0.1.5 (2024-05-28)

- Add `server::graceful::GracefulShutdown` helper to coordinate over many connections.
- Add `server::conn::auto::Connection::into_owned()` to unlink lifetime from `Builder`.
- Allow `service` module to be available with only `service` feature enabled.

# 0.1.4 (2024-05-24)

- Add `initial_max_send_streams()` to `legacy` client builder
- Add `max_pending_accept_reset_streams()` to `legacy` client builder
- Add `max_headers(usize)` to `auto` server builder
- Add `http1_onl()` and `http2_only()` to `auto` server builder
- Add connection capturing API to `legacy` client
- Add `impl Connection for TokioIo`
- Fix graceful shutdown hanging on reading the HTTP version

# 0.1.3 (2024-01-31)

### Added

- Add `Error::is_connect()` which returns true if error came from client `Connect`.
- Add timer support to `legacy` pool.
- Add support to enable http1/http2 parts of `auto::Builder` individually.

### Fixed

- Fix `auto` connection so it can handle requests shorter than the h2 preface.
- Fix `legacy::Client` to no longer error when keep-alive is diabled.

# 0.1.2 (2023-12-20)

### Added

- Add `graceful_shutdown()` method to `auto` connections.
- Add `rt::TokioTimer` type that implements `hyper::rt::Timer`.
- Add `service::TowerToHyperService` adapter, allowing using `tower::Service`s as a `hyper::service::Service`.
- Implement `Clone` for `auto::Builder`.
- Exports `legacy::{Builder, ResponseFuture}`.

### Fixed

- Enable HTTP/1 upgrades on the `legacy::Client`.
- Prevent divide by zero if DNS returns 0 addresses.

# 0.1.1 (2023-11-17)

### Added

- Make `server-auto` enable the `server` feature.

### Fixed

- Reduce `Send` bounds requirements for `auto` connections.
- Docs: enable all features when generating.

# 0.1.0 (2023-11-16)

Initial release.
