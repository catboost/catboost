//! Contains structs defining Avro's logical types
mod de;
mod se;

/// An Avro Schema. It describes all _physical_ and _logical_ types.
/// See [the spec](https://avro.apache.org/docs/current/spec.html) for details.
#[derive(Debug, Clone, PartialEq, Hash)]
pub enum Schema {
    /// A null type
    Null,
    /// Boolean (physically represented as a single byte)
    Boolean,
    /// 32 bit signed integer (physically represented as a zigzag encoded variable number of bytes)
    Int(Option<IntLogical>),
    /// 64 bit signed integer (physically represented as a zigzag encoded variable number of bytes)
    Long(Option<LongLogical>),
    /// 32 bit float (physically represented as 4 bytes in little endian)
    Float,
    /// 64 bit float (physically represented as 8 bytes in little endian)
    Double,
    /// variable length bytes (physically represented by a zigzag encoded positive integer followed by its number of bytes)
    Bytes(Option<BytesLogical>),
    /// variable length utf8 (physically represented by a zigzag encoded positive integer followed by its number of bytes)
    String(Option<StringLogical>),
    /// Record
    Record(Record),
    /// Enum with a known number of variants
    Enum(Enum),
    /// Array of a uniform type with N entries
    Array(Box<Schema>),
    /// A map String -> type
    Map(Box<Schema>),
    /// A union of a heterogeneous number of types
    Union(Vec<Schema>),
    /// todo
    Fixed(Fixed),
}

/// Order of a [`Field`].
#[derive(Debug, Clone, Copy, PartialEq, Hash)]
pub enum Order {
    /// Ascending order
    Ascending,
    /// Descending order
    Descending,
    /// Order is to be ignored
    Ignore,
}

/// An Avro field.
/// See [the spec](https://avro.apache.org/docs/current/spec.html) for details.
#[derive(Debug, Clone, PartialEq, Hash)]
pub struct Field {
    /// Its name
    pub name: String,
    /// Its optional documentation
    pub doc: Option<String>,
    /// Its Schema
    pub schema: Schema,
    /// Its default value
    pub default: Option<Schema>,
    /// Its optional order
    pub order: Option<Order>,
    /// Its aliases
    pub aliases: Vec<String>,
}

impl Field {
    /// Returns a new [`Field`] without a doc, default, order or aliases
    pub fn new<I: Into<String>>(name: I, schema: Schema) -> Self {
        Self {
            name: name.into(),
            doc: None,
            schema,
            default: None,
            order: None,
            aliases: vec![],
        }
    }
}

/// Struct to hold data from a [`Schema::Record`].
#[derive(Debug, Clone, PartialEq, Hash)]
pub struct Record {
    /// Its name
    pub name: String,
    /// Its optional namespace
    pub namespace: Option<String>,
    /// Its optional documentation
    pub doc: Option<String>,
    /// Its aliases
    pub aliases: Vec<String>,
    /// Its children fields
    pub fields: Vec<Field>,
}

impl Record {
    /// Returns a new [`Record`] without a namespace, doc or aliases
    pub fn new<I: Into<String>>(name: I, fields: Vec<Field>) -> Self {
        Self {
            name: name.into(),
            namespace: None,
            doc: None,
            fields,
            aliases: vec![],
        }
    }
}

/// Struct to hold data from a [`Schema::Fixed`].
#[derive(Debug, Clone, PartialEq, Hash)]
pub struct Fixed {
    /// Its name
    pub name: String,
    /// Its optional namespace
    pub namespace: Option<String>,
    /// Its optional documentation
    pub doc: Option<String>,
    /// Its aliases
    pub aliases: Vec<String>,
    /// Its size
    pub size: usize,
    /// Its optional logical type
    pub logical: Option<FixedLogical>,
}

impl Fixed {
    /// Returns a new [`Fixed`] without a namespace, doc or aliases
    pub fn new<I: Into<String>>(name: I, size: usize) -> Self {
        Self {
            name: name.into(),
            namespace: None,
            doc: None,
            size,
            aliases: vec![],
            logical: None,
        }
    }
}

/// Struct to hold data from a [`Schema::Enum`].
#[derive(Debug, Clone, PartialEq, Hash)]
pub struct Enum {
    /// Its name
    pub name: String,
    /// Its optional namespace
    pub namespace: Option<String>,
    /// Its aliases
    pub aliases: Vec<String>,
    /// Its optional documentation
    pub doc: Option<String>,
    /// Its set of symbols
    pub symbols: Vec<String>,
    /// Its default symbol
    pub default: Option<String>,
}

impl Enum {
    /// Returns a minimal [`Enum`].
    pub fn new<I: Into<String>>(name: I, symbols: Vec<String>) -> Self {
        Self {
            name: name.into(),
            namespace: None,
            doc: None,
            symbols,
            aliases: vec![],
            default: None,
        }
    }
}

impl From<Enum> for Schema {
    fn from(enum_: Enum) -> Self {
        Schema::Enum(enum_)
    }
}

impl From<Record> for Schema {
    fn from(record: Record) -> Self {
        Schema::Record(record)
    }
}

impl From<Fixed> for Schema {
    fn from(fixed: Fixed) -> Self {
        Schema::Fixed(fixed)
    }
}

/// Enum of all logical types of [`Schema::Int`]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IntLogical {
    /// A date
    Date,
    /// A time
    Time,
}

/// Enum of all logical types of [`Schema::Long`]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LongLogical {
    /// A time
    Time,
    /// A timestamp
    TimestampMillis,
    /// A timestamp
    TimestampMicros,
    /// A timestamp without timezone
    LocalTimestampMillis,
    /// A timestamp without timezone
    LocalTimestampMicros,
}

/// Enum of all logical types of [`Schema::String`]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StringLogical {
    /// A UUID
    Uuid,
}

/// Enum of all logical types of [`Schema::Fixed`]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FixedLogical {
    /// A decimal
    Decimal(usize, usize),
    /// A duration
    Duration,
}

/// Enum of all logical types of [`Schema::Bytes`]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BytesLogical {
    /// A decimal
    Decimal(usize, usize),
}
