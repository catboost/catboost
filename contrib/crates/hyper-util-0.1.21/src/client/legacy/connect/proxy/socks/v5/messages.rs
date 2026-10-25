use super::super::{ParsingError, SerializeError};

use bytes::{Buf, BufMut, BytesMut};
use std::net::SocketAddr;

///  +----+----------+----------+
/// |VER | NMETHODS | METHODS  |
/// +----+----------+----------+
/// | 1  |    1     | 1 to 255 |
/// +----+----------+----------+
#[derive(Debug, PartialEq)]
pub struct NegotiationReq<'a>(pub &'a AuthMethod);

/// +----+--------+
/// |VER | METHOD |
/// +----+--------+
/// | 1  |   1    |
/// +----+--------+
#[derive(Debug, PartialEq)]
pub struct NegotiationRes(pub AuthMethod);

/// +----+------+----------+------+----------+
/// |VER | ULEN |  UNAME   | PLEN |  PASSWD  |
/// +----+------+----------+------+----------+
/// | 1  |  1   | 1 to 255 |  1   | 1 to 255 |
/// +----+------+----------+------+----------+
#[derive(Debug, PartialEq)]
pub struct AuthenticationReq<'a>(pub &'a str, pub &'a str);

/// +----+--------+
/// |VER | STATUS |
/// +----+--------+
/// | 1  |   1    |
/// +----+--------+
#[derive(Debug, PartialEq)]
pub struct AuthenticationRes(pub bool);

/// +----+-----+-------+------+----------+----------+
/// |VER | CMD |  RSV  | ATYP | DST.ADDR | DST.PORT |
/// +----+-----+-------+------+----------+----------+
/// | 1  |  1  | X'00' |  1   | Variable |    2     |
/// +----+-----+-------+------+----------+----------+
#[derive(Debug, PartialEq)]
pub struct ProxyReq<'a>(pub &'a Address);

/// +----+-----+-------+------+----------+----------+
/// |VER | REP |  RSV  | ATYP | BND.ADDR | BND.PORT |
/// +----+-----+-------+------+----------+----------+
/// | 1  |  1  | X'00' |  1   | Variable |    2     |
/// +----+-----+-------+------+----------+----------+
#[derive(Debug, PartialEq)]
pub struct ProxyRes(pub Status);

#[repr(u8)]
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum AuthMethod {
    NoAuth = 0x00,
    UserPass = 0x02,
    NoneAcceptable = 0xFF,
}

#[derive(Debug, PartialEq)]
pub enum Address {
    Socket(SocketAddr),
    Domain(String, u16),
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum Status {
    Success,
    GeneralServerFailure,
    ConnectionNotAllowed,
    NetworkUnreachable,
    HostUnreachable,
    ConnectionRefused,
    TtlExpired,
    CommandNotSupported,
    AddressTypeNotSupported,
}

impl NegotiationReq<'_> {
    pub fn write_to_buf(&self, buf: &mut BytesMut) -> Result<usize, SerializeError> {
        if buf.capacity() - buf.len() < 3 {
            return Err(SerializeError::WouldOverflow);
        }

        buf.put_u8(0x05); // Version
        buf.put_u8(0x01); // Number of authentication methods
        buf.put_u8(*self.0 as u8); // Authentication method

        Ok(3)
    }
}

impl TryFrom<&mut &[u8]> for NegotiationRes {
    type Error = ParsingError;

    fn try_from(buf: &mut &[u8]) -> Result<Self, ParsingError> {
        if buf.remaining() < 2 {
            return Err(ParsingError::Incomplete);
        }

        if buf.get_u8() != 0x05 {
            return Err(ParsingError::Other);
        }

        let method = buf.get_u8().try_into()?;
        Ok(Self(method))
    }
}

impl AuthenticationReq<'_> {
    pub fn write_to_buf(&self, buf: &mut BytesMut) -> Result<usize, SerializeError> {
        if buf.capacity() - buf.len() < 3 + self.0.len() + self.1.len() {
            return Err(SerializeError::WouldOverflow);
        }

        buf.put_u8(0x01); // Version

        buf.put_u8(self.0.len() as u8); // Username length (guaranteed to be 255 or less)
        buf.put_slice(self.0.as_bytes()); // Username

        buf.put_u8(self.1.len() as u8); // Password length (guaranteed to be 255 or less)
        buf.put_slice(self.1.as_bytes()); // Password

        Ok(3 + self.0.len() + self.1.len())
    }
}

impl TryFrom<&mut &[u8]> for AuthenticationRes {
    type Error = ParsingError;

    fn try_from(buf: &mut &[u8]) -> Result<Self, ParsingError> {
        if buf.remaining() < 2 {
            return Err(ParsingError::Incomplete);
        }

        if buf.get_u8() != 0x01 {
            return Err(ParsingError::Other);
        }

        if buf.get_u8() == 0 {
            Ok(Self(true))
        } else {
            Ok(Self(false))
        }
    }
}

impl ProxyReq<'_> {
    pub fn write_to_buf(&self, buf: &mut BytesMut) -> Result<usize, SerializeError> {
        let addr_len = match self.0 {
            Address::Socket(SocketAddr::V4(_)) => 1 + 4 + 2,
            Address::Socket(SocketAddr::V6(_)) => 1 + 16 + 2,
            Address::Domain(domain, _) => 1 + 1 + domain.len() + 2,
        };

        if buf.capacity() - buf.len() < 3 + addr_len {
            return Err(SerializeError::WouldOverflow);
        }

        buf.put_u8(0x05); // Version
        buf.put_u8(0x01); // TCP tunneling command
        buf.put_u8(0x00); // Reserved
        let _ = self.0.write_to_buf(buf); // Address

        Ok(3 + addr_len)
    }
}

impl TryFrom<&mut &[u8]> for ProxyRes {
    type Error = ParsingError;

    fn try_from(buf: &mut &[u8]) -> Result<Self, ParsingError> {
        if buf.remaining() < 3 {
            return Err(ParsingError::Incomplete);
        }

        // VER
        if buf.get_u8() != 0x05 {
            return Err(ParsingError::Other);
        }

        // REP
        let status = buf.get_u8().try_into()?;

        // RSV
        if buf.get_u8() != 0x00 {
            return Err(ParsingError::Other);
        }

        // ATYP + ADDR
        Address::try_from(buf)?;

        Ok(Self(status))
    }
}

impl Address {
    pub fn write_to_buf(&self, buf: &mut BytesMut) -> Result<usize, SerializeError> {
        match self {
            Self::Socket(SocketAddr::V4(v4)) => {
                if buf.capacity() - buf.len() < 1 + 4 + 2 {
                    return Err(SerializeError::WouldOverflow);
                }

                buf.put_u8(0x01);
                buf.put_slice(&v4.ip().octets());
                buf.put_u16(v4.port()); // Network Order/BigEndian for port

                Ok(7)
            }

            Self::Socket(SocketAddr::V6(v6)) => {
                if buf.capacity() - buf.len() < 1 + 16 + 2 {
                    return Err(SerializeError::WouldOverflow);
                }

                buf.put_u8(0x04);
                buf.put_slice(&v6.ip().octets());
                buf.put_u16(v6.port()); // Network Order/BigEndian for port

                Ok(19)
            }

            Self::Domain(domain, port) => {
                if buf.capacity() - buf.len() < 1 + 1 + domain.len() + 2 {
                    return Err(SerializeError::WouldOverflow);
                }

                buf.put_u8(0x03);
                buf.put_u8(domain.len() as u8); // Guarenteed to be less than 255
                buf.put_slice(domain.as_bytes());
                buf.put_u16(*port);

                Ok(4 + domain.len())
            }
        }
    }
}

impl TryFrom<&mut &[u8]> for Address {
    type Error = ParsingError;

    fn try_from(buf: &mut &[u8]) -> Result<Self, Self::Error> {
        if buf.remaining() < 2 {
            return Err(ParsingError::Incomplete);
        }

        Ok(match buf.get_u8() {
            // IPv4
            0x01 => {
                let mut ip = [0; 4];

                if buf.remaining() < 6 {
                    return Err(ParsingError::Incomplete);
                }

                buf.copy_to_slice(&mut ip);
                let port = buf.get_u16();

                Self::Socket(SocketAddr::new(ip.into(), port))
            }
            // Domain
            0x03 => {
                let len = buf.get_u8() as usize;

                if len == 0 {
                    return Err(ParsingError::Other);
                } else if buf.remaining() < len + 2 {
                    return Err(ParsingError::Incomplete);
                }

                let domain = std::str::from_utf8(&buf.chunk()[..len])
                    .map_err(|_| ParsingError::Other)?
                    .to_string();
                buf.advance(len);

                let port = buf.get_u16();

                Self::Domain(domain, port)
            }
            // IPv6
            0x04 => {
                let mut ip = [0; 16];

                if buf.remaining() < 18 {
                    return Err(ParsingError::Incomplete);
                }
                buf.copy_to_slice(&mut ip);
                let port = buf.get_u16();

                Self::Socket(SocketAddr::new(ip.into(), port))
            }

            _ => return Err(ParsingError::Other),
        })
    }
}

impl TryFrom<u8> for Status {
    type Error = ParsingError;

    fn try_from(byte: u8) -> Result<Self, Self::Error> {
        Ok(match byte {
            0x00 => Self::Success,

            0x01 => Self::GeneralServerFailure,
            0x02 => Self::ConnectionNotAllowed,
            0x03 => Self::NetworkUnreachable,
            0x04 => Self::HostUnreachable,
            0x05 => Self::ConnectionRefused,
            0x06 => Self::TtlExpired,
            0x07 => Self::CommandNotSupported,
            0x08 => Self::AddressTypeNotSupported,
            _ => return Err(ParsingError::Other),
        })
    }
}

impl TryFrom<u8> for AuthMethod {
    type Error = ParsingError;

    fn try_from(byte: u8) -> Result<Self, Self::Error> {
        Ok(match byte {
            0x00 => Self::NoAuth,
            0x02 => Self::UserPass,
            0xFF => Self::NoneAcceptable,

            _ => return Err(ParsingError::Other),
        })
    }
}

impl std::fmt::Display for Status {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Success => "success",
            Self::GeneralServerFailure => "general server failure",
            Self::ConnectionNotAllowed => "connection not allowed",
            Self::NetworkUnreachable => "network unreachable",
            Self::HostUnreachable => "host unreachable",
            Self::ConnectionRefused => "connection refused",
            Self::TtlExpired => "ttl expired",
            Self::CommandNotSupported => "command not supported",
            Self::AddressTypeNotSupported => "address type not supported",
        })
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use std::net::{Ipv4Addr, Ipv6Addr};

    #[test]
    fn negotiation_req_serialization() {
        let expected = [
            0x05, // protocol version
            0x01, // number of authentication methods: 1
            0x00, // method: no authentication
        ];

        let mut buf = BytesMut::with_capacity(expected.len());
        let n = NegotiationReq(&AuthMethod::NoAuth)
            .write_to_buf(&mut buf)
            .unwrap();
        assert_eq!(n, buf.len());
        assert_eq!(&buf[..], &expected[..]);
    }

    #[test]
    fn negotiation_res_deserialization() {
        let raw = [
            0x05, // protocol version
            0x02, // selected method: username/password
        ];

        let mut view = &raw[..];

        let res = NegotiationRes::try_from(&mut view).unwrap();
        assert_eq!(res, NegotiationRes(AuthMethod::UserPass));
        assert!(view.is_empty());
    }

    #[test]
    fn authentication_req_serialization() {
        let expected = [
            0x01, // authentication version
            0x04, // username length: 4
            b'u', b's', b'e', b'r', // username
            0x04, // password length: 4
            b'p', b'a', b's', b's', // password
        ];

        let mut buf = BytesMut::with_capacity(expected.len());
        let n = AuthenticationReq("user", "pass")
            .write_to_buf(&mut buf)
            .unwrap();
        assert_eq!(n, buf.len());
        assert_eq!(&buf[..], &expected[..]);
    }

    #[test]
    fn authentication_res_deserialization() {
        let raw = [
            0x01, // authentication version
            0x00, // status: success
        ];
        assert_eq!(
            AuthenticationRes::try_from(&mut &raw[..]).unwrap(),
            AuthenticationRes(true)
        );

        let raw = [
            0x01, // authentication version
            0x01, // status: failure
        ];
        assert_eq!(
            AuthenticationRes::try_from(&mut &raw[..]).unwrap(),
            AuthenticationRes(false)
        );
    }

    #[test]
    fn proxy_req_serialization() {
        let expected = [
            0x05, // protocol version
            0x01, // command: connect
            0x00, // reserved
            0x01, // address type: IPv4
            127, 0, 0, 1, // destination address: 127.0.0.1
            0x1F, 0x90, // destination port: 8080
        ];

        let addr = Address::Socket(SocketAddr::new(Ipv4Addr::LOCALHOST.into(), 8080));

        let mut buf = BytesMut::with_capacity(expected.len());
        let n = ProxyReq(&addr).write_to_buf(&mut buf).unwrap();
        assert_eq!(n, buf.len());
        assert_eq!(&buf[..], &expected[..]);
    }

    #[test]
    fn proxy_res_deserialization() {
        let raw = [
            0x05, // protocol version
            0x00, // reply: success
            0x00, // reserved
            0x01, // address type: IPv4
            127, 0, 0, 1, // bound address: 127.0.0.1
            0x1F, 0x90, // bound port: 8080
        ];
        let mut view = &raw[..];

        let res = ProxyRes::try_from(&mut view).unwrap();
        assert_eq!(res, ProxyRes(Status::Success));
        assert!(view.is_empty());
    }

    #[test]
    fn serialization_would_overflow() {
        let mut buf = BytesMut::with_capacity(2);

        let err = NegotiationReq(&AuthMethod::NoAuth)
            .write_to_buf(&mut buf)
            .unwrap_err();
        assert!(matches!(err, SerializeError::WouldOverflow));
    }

    fn assert_address_roundtrips(addr: Address) {
        let mut buf = BytesMut::with_capacity(64);

        let n = addr.write_to_buf(&mut buf).unwrap();
        assert_eq!(n, buf.len());

        let mut view = &buf[..];
        assert_eq!(Address::try_from(&mut view).unwrap(), addr);
        assert!(view.is_empty(), "address bytes should be fully consumed");
    }

    #[test]
    fn address_roundtrip_ipv4() {
        assert_address_roundtrips(Address::Socket(SocketAddr::new(
            Ipv4Addr::LOCALHOST.into(),
            8080,
        )));
    }

    #[test]
    fn address_roundtrip_ipv6() {
        assert_address_roundtrips(Address::Socket(SocketAddr::new(
            Ipv6Addr::LOCALHOST.into(),
            8080,
        )));
    }

    #[test]
    fn address_roundtrip_domain() {
        assert_address_roundtrips(Address::Domain("example.com".into(), 8080));
    }
}
