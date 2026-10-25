use serde::{
    Deserialize, Deserializer,
    de::{Error as DeError, Visitor},
};
use std::fmt::{Formatter, Result as FmtResult};

const JSON: &str = r#"{
    "channels": [
      {
        "default_auto_archive_duration": 10080
      }
    ],
    "unavailable": false
}"#;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(untagged)]
pub enum GuildCreate {
    Available(Guild),
    Unavailable(UnavailableGuild),
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq)]
pub struct UnavailableGuild {
    pub unavailable: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Hash)]
pub struct Guild {
    pub channels: Vec<Channel>,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq)]
pub struct Channel {
    pub default_auto_archive_duration: Option<AutoArchiveDuration>,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
// Fix 1: Uncomment this and comment the handwritten deser impl.
// #[derive(Deserialize)]
// #[serde(from = "u16")]
pub enum AutoArchiveDuration {
    Hour,
    Day,
    ThreeDays,
    Week,
    Unknown { value: u16 },
}

impl From<u16> for AutoArchiveDuration {
    fn from(value: u16) -> Self {
        match value {
            60 => Self::Hour,
            1440 => Self::Day,
            4320 => Self::ThreeDays,
            10080 => Self::Week,
            value => Self::Unknown { value },
        }
    }
}

impl<'de> Deserialize<'de> for AutoArchiveDuration {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_u16(U16EnumVisitor).map(u16::into)
    }
}

pub struct U16EnumVisitor;

impl Visitor<'_> for U16EnumVisitor {
    type Value = u16;

    fn expecting(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.write_str("u16")
    }

    fn visit_u16<E: DeError>(self, value: u16) -> Result<Self::Value, E> {
        Ok(value)
    }

    fn visit_u64<E>(self, v: u64) -> Result<Self::Value, E>
    where
        E: DeError,
    {
        v.try_into().map_err(DeError::custom)
    }

    // Fix 2: Uncomment this
    // fn visit_i64<E>(self, v: i64) -> Result<Self::Value, E>
    // where
    //     E: DeError,
    // {
    //     v.try_into().map_err(DeError::custom)
    // }
}

#[test]
fn test_deser_u16() -> Result<(), Box<dyn std::error::Error>> {
    unsafe {
        let mut json = JSON.to_string();
        let a = dbg!(crate::from_str::<Guild>(&mut json)?);
        let b = dbg!(crate::from_str::<UnavailableGuild>(&mut json)?);
        let c = dbg!(crate::from_str::<GuildCreate>(&mut json)?);
        assert_eq!(a, serde_json::from_str::<Guild>(&json)?);
        assert_eq!(b, serde_json::from_str::<UnavailableGuild>(&json)?);
        assert_eq!(c, serde_json::from_str::<GuildCreate>(&json)?);
    };
    Ok(())
}
