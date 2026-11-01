use super::super::{ParsingError, SerializeError};

use bytes::{Buf, BufMut};
use std::net::SocketAddrV4;

/// +-----+-----+----+----+----+----+----+----+-------------+------+------------+------+
/// |  VN |  CD | DSTPORT |        DSTIP      |    USERID   | NULL |   DOMAIN   | NULL |
/// +-----+-----+----+----+----+----+----+----+-------------+------+------------+------+
/// |  1  |  1  |    2    |         4         |   Variable  |  1   |  Variable  |   1  |
/// +-----+-----+----+----+----+----+----+----+-------------+------+------------+------+
///                                                                ^^^^^^^^^^^^^^^^^^^^^
///                                                   optional: only do if IP is 0.0.0.X
#[derive(Debug, PartialEq)]
pub struct Request<'a>(pub &'a Address);

/// +-----+-----+----+----+----+----+----+----+
/// |  VN |  CD | DSTPORT |       DSTIP       |
/// +-----+-----+----+----+----+----+----+----+
/// |  1  |  1  |    2    |         4         |
/// +-----+-----+----+----+----+----+----+----+
///             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
///              ignore: only for SOCKSv4 BIND
#[derive(Debug, PartialEq)]
pub struct Response(pub Status);

#[derive(Debug, PartialEq)]
pub enum Address {
    Socket(SocketAddrV4),
    Domain(String, u16),
}

#[derive(Debug, PartialEq)]
pub enum Status {
    Success = 90,
    Failed = 91,
    IdentFailure = 92,
    IdentMismatch = 93,
}

impl Request<'_> {
    pub fn write_to_buf<B: BufMut>(&self, mut buf: B) -> Result<usize, SerializeError> {
        match self.0 {
            Address::Socket(socket) => {
                if buf.remaining_mut() < 9 {
                    return Err(SerializeError::WouldOverflow);
                }

                buf.put_u8(0x04); // Version
                buf.put_u8(0x01); // CONNECT

                buf.put_u16(socket.port()); // Port
                buf.put_slice(&socket.ip().octets()); // IP

                buf.put_u8(0x00); // NULL terminating an empty USERID

                Ok(9)
            }

            Address::Domain(domain, port) => {
                if buf.remaining_mut() < 9 + domain.len() + 1 {
                    return Err(SerializeError::WouldOverflow);
                }

                buf.put_u8(0x04); // Version
                buf.put_u8(0x01); // CONNECT

                buf.put_u16(*port); // Port
                buf.put_slice(&[0x00, 0x00, 0x00, 0xFF]); // Invalid IP

                buf.put_u8(0x00); // NULL terminating an empty USERID

                buf.put_slice(domain.as_bytes()); // Domain
                buf.put_u8(0x00); // NULL

                Ok(9 + domain.len() + 1)
            }
        }
    }
}

impl TryFrom<&mut &[u8]> for Response {
    type Error = ParsingError;

    fn try_from(buf: &mut &[u8]) -> Result<Self, Self::Error> {
        if buf.remaining() < 8 {
            return Err(ParsingError::Incomplete);
        }

        if buf.get_u8() != 0x00 {
            return Err(ParsingError::Other);
        }

        let status = buf.get_u8().try_into()?;
        let _addr = {
            let port = buf.get_u16();
            let mut ip = [0; 4];
            buf.copy_to_slice(&mut ip);

            SocketAddrV4::new(ip.into(), port)
        };

        Ok(Self(status))
    }
}

impl TryFrom<u8> for Status {
    type Error = ParsingError;

    fn try_from(byte: u8) -> Result<Self, Self::Error> {
        Ok(match byte {
            90 => Self::Success,
            91 => Self::Failed,
            92 => Self::IdentFailure,
            93 => Self::IdentMismatch,
            _ => return Err(ParsingError::Other),
        })
    }
}

impl std::fmt::Display for Status {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Success => "success",
            Self::Failed => "server failed to execute command",
            Self::IdentFailure => "server ident service failed",
            Self::IdentMismatch => "server ident service did not recognise client identifier",
        })
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use bytes::BytesMut;
    use std::net::Ipv4Addr;

    #[test]
    fn request_serialization_with_socket() {
        let expected = [
            0x04, // protocol version
            0x01, // command: connect
            0x1F, 0x90, // destination port: 8080
            127, 0, 0, 1,    // destination address: 127.0.0.1
            0x00, // null terminating an empty userid
        ];

        let addr = Address::Socket(SocketAddrV4::new(Ipv4Addr::LOCALHOST, 8080));
        let mut buf = BytesMut::with_capacity(expected.len());
        let n = Request(&addr).write_to_buf(&mut buf).unwrap();
        assert_eq!(n, buf.len());
        assert_eq!(&buf[..], &expected[..]);
    }

    #[test]
    fn request_serialization_with_domain() {
        let expected = [
            0x04, // protocol version
            0x01, // command: connect
            0x1F, 0x90, // destination port: 8080
            0x00, 0x00, 0x00, 0xFF, // invalid IP: signals that a domain follows (SOCKS4a)
            0x00, // null terminating an empty userid
            b'e', b'x', b'a', b'm', b'p', b'l', b'e', b'.', b'c', b'o', b'm', // domain
            0x00, // null terminator
        ];

        let addr = Address::Domain("example.com".into(), 8080);
        let mut buf = BytesMut::with_capacity(expected.len());
        let n = Request(&addr).write_to_buf(&mut buf).unwrap();
        assert_eq!(n, buf.len());
        assert_eq!(&buf[..], &expected[..]);
    }

    #[test]
    fn response_deserialization() {
        let raw = [
            0x00, // reply version
            90,   // status: request granted
            0x1F, 0x90, // port: 8080 (ignored, only used for BIND)
            127, 0, 0, 1, // address: 127.0.0.1 (ignored, only used for BIND)
        ];
        let mut view = &raw[..];

        let res = Response::try_from(&mut view).unwrap();
        assert_eq!(res, Response(Status::Success));
        assert!(view.is_empty());
    }

    #[test]
    fn response_incomplete() {
        let raw = [
            0x00, // reply version
            90,   // status: request granted
            0x00, // truncated mid-port
        ];

        let err = Response::try_from(&mut &raw[..]).unwrap_err();
        assert!(matches!(err, ParsingError::Incomplete));
    }

    #[test]
    fn request_serialization_would_overflow() {
        let addr = Address::Socket(SocketAddrV4::new(Ipv4Addr::LOCALHOST, 8080));
        let mut short = [0u8; 5];
        let err = Request(&addr).write_to_buf(&mut short[..]).unwrap_err();
        assert!(matches!(err, SerializeError::WouldOverflow));
    }
}
