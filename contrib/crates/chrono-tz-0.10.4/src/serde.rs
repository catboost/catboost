extern crate serde;

use self::serde::{de, Deserialize, Deserializer, Serialize, Serializer};
use core::fmt;

use crate::timezones::Tz;

impl Serialize for Tz {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.name())
    }
}

impl<'de> Deserialize<'de> for Tz {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct Visitor;

        impl<'de> de::Visitor<'de> for Visitor {
            type Value = Tz;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                write!(formatter, "an IANA timezone string")
            }

            fn visit_str<E: de::Error>(self, value: &str) -> Result<Tz, E> {
                value
                    .parse::<Tz>()
                    .map_err(|_| E::custom(SerdeError(value)))
            }
        }

        deserializer.deserialize_str(Visitor)
    }
}

struct SerdeError<'a>(&'a str);

impl fmt::Display for SerdeError<'_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "failed to parse timezone: '{}'", self.0)
    }
}

#[cfg(test)]
mod tests {
    use crate::timezones::Tz::{self, Etc__UTC, Europe__London, UTC};
    use serde_test::{assert_de_tokens_error, assert_tokens, Token};

    #[test]
    fn serde_ok_both_ways() {
        assert_tokens(&Europe__London, &[Token::String("Europe/London")]);
        assert_tokens(&Etc__UTC, &[Token::String("Etc/UTC")]);
        assert_tokens(&UTC, &[Token::String("UTC")]);
    }

    #[test]
    fn serde_de_error() {
        assert_de_tokens_error::<Tz>(
            &[Token::Str("Europe/L")],
            "failed to parse timezone: 'Europe/L'",
        );
    }
}
