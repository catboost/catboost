use serde::ser::{SerializeMap, SerializeSeq};
use serde::{Serialize, Serializer};

use super::*;

impl Serialize for Schema {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Schema::Null => serializer.serialize_str("null"),
            Schema::Boolean => serializer.serialize_str("boolean"),
            Schema::Int(logical) => match logical {
                None => serializer.serialize_str("int"),
                Some(logical) => {
                    let mut map = serializer.serialize_map(Some(2))?;
                    map.serialize_entry("type", "int")?;
                    let name = match logical {
                        IntLogical::Date => "date",
                        IntLogical::Time => "time-millis",
                    };
                    map.serialize_entry("logicalType", name)?;
                    map.end()
                }
            },
            Schema::Long(logical) => match logical {
                None => serializer.serialize_str("long"),
                Some(logical) => {
                    let mut map = serializer.serialize_map(Some(2))?;
                    map.serialize_entry("type", "long")?;
                    let name = match logical {
                        LongLogical::Time => "time-micros",
                        LongLogical::TimestampMillis => "timestamp-millis",
                        LongLogical::TimestampMicros => "timestamp-micros",
                        LongLogical::LocalTimestampMillis => "local-timestamp-millis",
                        LongLogical::LocalTimestampMicros => "local-timestamp-micros",
                    };
                    map.serialize_entry("logicalType", name)?;
                    map.end()
                }
            },
            Schema::Float => serializer.serialize_str("float"),
            Schema::Double => serializer.serialize_str("double"),
            Schema::Bytes(logical) => match logical {
                None => serializer.serialize_str("bytes"),
                Some(logical) => match logical {
                    BytesLogical::Decimal(precision, scale) => {
                        let mut map = serializer.serialize_map(Some(4))?;
                        map.serialize_entry("type", "bytes")?;
                        map.serialize_entry("logicalType", "decimal")?;
                        map.serialize_entry("precision", precision)?;
                        if *scale > 0 {
                            map.serialize_entry("scale", scale)?;
                        }
                        map.end()
                    }
                },
            },
            Schema::String(logical) => match logical {
                None => serializer.serialize_str("string"),
                Some(logical) => match logical {
                    StringLogical::Uuid => {
                        let mut map = serializer.serialize_map(Some(1))?;
                        map.serialize_entry("type", "string")?;
                        map.serialize_entry("logicalType", "uuid")?;
                        map.end()
                    }
                },
            },
            Schema::Record(record) => {
                let Record {
                    name,
                    namespace,
                    doc,
                    aliases,
                    fields,
                } = record;
                let mut map = serializer.serialize_map(Some(3))?;
                map.serialize_entry("type", "record")?;
                map.serialize_entry("name", name)?;
                if let Some(namespace) = namespace {
                    map.serialize_entry("namespace", namespace)?;
                }
                if !aliases.is_empty() {
                    map.serialize_entry("aliases", aliases)?;
                }
                if let Some(doc) = doc {
                    map.serialize_entry("doc", doc)?;
                }
                map.serialize_entry("fields", fields)?;
                map.end()
            }
            Schema::Enum(enum_) => {
                let Enum {
                    name,
                    namespace,
                    aliases,
                    doc,
                    symbols,
                    default,
                } = enum_;
                let mut map = serializer.serialize_map(Some(3))?;
                map.serialize_entry("type", "enum")?;
                map.serialize_entry("name", name)?;
                if let Some(namespace) = namespace {
                    map.serialize_entry("namespace", namespace)?;
                }
                if !aliases.is_empty() {
                    map.serialize_entry("aliases", aliases)?;
                }
                if let Some(doc) = doc {
                    map.serialize_entry("doc", doc)?;
                }
                if let Some(default) = default {
                    map.serialize_entry("default", default)?;
                }
                map.serialize_entry("symbols", symbols)?;
                map.end()
            }
            Schema::Array(schema) => {
                let mut map = serializer.serialize_map(Some(2))?;
                map.serialize_entry("type", "array")?;
                map.serialize_entry("items", schema.as_ref())?;
                map.end()
            }
            Schema::Map(schema) => {
                let mut map = serializer.serialize_map(Some(2))?;
                map.serialize_entry("type", "map")?;
                map.serialize_entry("values", schema.as_ref())?;
                map.end()
            }
            Schema::Union(schemas) => {
                let mut seq = serializer.serialize_seq(Some(schemas.len()))?;
                for schema in schemas {
                    seq.serialize_element(schema)?;
                }
                seq.end()
            }
            Schema::Fixed(fixed) => {
                let Fixed {
                    name,
                    namespace,
                    doc,
                    aliases,
                    size,
                    logical,
                } = fixed;

                let mut map = serializer.serialize_map(None)?;
                map.serialize_entry("type", "fixed")?;
                map.serialize_entry("name", name)?;
                if let Some(namespace) = namespace {
                    map.serialize_entry("namespace", namespace)?;
                }
                if !aliases.is_empty() {
                    map.serialize_entry("aliases", aliases)?;
                }
                if let Some(doc) = doc {
                    map.serialize_entry("doc", doc)?;
                }
                map.serialize_entry("size", size)?;

                if let Some(logical) = logical {
                    match logical {
                        FixedLogical::Decimal(precision, scale) => {
                            map.serialize_entry("logicalType", "decimal")?;
                            map.serialize_entry("precision", precision)?;
                            if *scale > 0 {
                                map.serialize_entry("scale", scale)?;
                            }
                        }
                        FixedLogical::Duration => map.serialize_entry("logicalType", "duration")?,
                    }
                }

                map.end()
            }
        }
    }
}

impl Serialize for Field {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let Field {
            name,
            doc,
            schema,
            default,
            order,
            aliases,
        } = self;

        let mut map = serializer.serialize_map(None)?;
        map.serialize_entry("name", name)?;
        if !aliases.is_empty() {
            map.serialize_entry("aliases", aliases)?;
        }
        if let Some(doc) = doc {
            map.serialize_entry("doc", doc)?;
        }
        if let Some(default) = default {
            map.serialize_entry("default", default)?;
        }
        map.serialize_entry("type", schema)?;
        if let Some(order) = order {
            let order = match order {
                Order::Ascending => "ascending",
                Order::Descending => "descending",
                Order::Ignore => "ignore",
            };
            map.serialize_entry("order", order)?;
        }

        map.end()
    }
}
