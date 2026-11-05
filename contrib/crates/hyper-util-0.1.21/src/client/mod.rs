//! HTTP client utilities

/// Legacy implementations of `connect` module and `Client`
#[cfg(feature = "client-legacy")]
pub mod legacy;

#[cfg(feature = "client-pool")]
pub mod pool;

#[cfg(feature = "client-proxy")]
pub mod proxy;

#[cfg(any(feature = "client-legacy", feature = "client-proxy"))]
fn strip_ipv6_brackets(host: &str) -> &str {
    host.strip_prefix('[')
        .and_then(|host| host.strip_suffix(']'))
        .unwrap_or(host)
}

#[cfg(all(test, any(feature = "client-legacy", feature = "client-proxy")))]
mod tests {
    use super::strip_ipv6_brackets;

    #[test]
    fn test_strip_ipv6_brackets() {
        assert_eq!(strip_ipv6_brackets("[::1]"), "::1");
        assert_eq!(strip_ipv6_brackets("::1"), "::1");
        assert_eq!(strip_ipv6_brackets("example.com"), "example.com");
        assert_eq!(strip_ipv6_brackets("[example.com"), "[example.com");
        assert_eq!(strip_ipv6_brackets("example.com]"), "example.com]");
    }
}
