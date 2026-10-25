use crate::IpNet;
use crate::Ipv4Net;
use crate::Ipv6Net;

#[cfg(not(feature = "std"))]
use alloc::borrow::Cow;
#[cfg(feature = "std")]
use std::borrow::Cow;

use schemars1::{json_schema, JsonSchema, Schema, SchemaGenerator};

impl JsonSchema for Ipv4Net {
    fn schema_name() -> Cow<'static, str> {
        "Ipv4Net".into()
    }

    fn json_schema(_: &mut SchemaGenerator) -> Schema {
        json_schema!({
            "title": "IPv4 network",
            "description": "An IPv4 address with prefix length",
            "examples": [
                "0.0.0.0/0",
                "192.168.0.0/24"
            ],
            "type": "string",
            "maxLength": 18,
            "pattern": r#"^(?:(?:25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9][0-9]|[0-9])\.){3}(?:25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9][0-9]|[0-9])\/(?:3[0-2]|[1-2][0-9]|[0-9])$"#
        })
    }
}
impl JsonSchema for Ipv6Net {
    fn schema_name() -> Cow<'static, str> {
        "Ipv6Net".into()
    }

    fn json_schema(_: &mut SchemaGenerator) -> Schema {
        json_schema!({
            "title": "IPv6 network",
            "description": "An IPv6 address with prefix length",
            "examples": [
                "::/0",
                "fd00::/32"
            ],
            "type": "string",
            "maxLength": 43,
            "pattern": r#"^[0-9A-Fa-f:\.]+\/(?:[0-9]|[1-9][0-9]|1[0-1][0-9]|12[0-8])$"#
        })
    }
}
impl JsonSchema for IpNet {
    fn schema_name() -> Cow<'static, str> {
        "IpNet".into()
    }

    fn json_schema(gen: &mut SchemaGenerator) -> Schema {
        json_schema!({
            "title": "IP network",
            "description": "An IPv4 or IPv6 address with prefix length",
            "examples": [
                "192.168.0.0/24",
                "fd00::/32"
            ],
            "oneOf": [
                gen.subschema_for::<Ipv4Net>(),
                gen.subschema_for::<Ipv6Net>()
            ]
        })
    }
}
