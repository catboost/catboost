//! Generated code of protobuf (Flight.proto) in [format](https://github.com/apache/arrow/tree/master/format)
// This file was automatically generated through the build.rs script, and should not be edited.

///
/// The request that a client provides to a server on handshake.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct HandshakeRequest {
    ///
    /// A defined protocol version
    #[prost(uint64, tag = "1")]
    pub protocol_version: u64,
    ///
    /// Arbitrary auth/handshake info.
    #[prost(bytes = "vec", tag = "2")]
    pub payload: ::prost::alloc::vec::Vec<u8>,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct HandshakeResponse {
    ///
    /// A defined protocol version
    #[prost(uint64, tag = "1")]
    pub protocol_version: u64,
    ///
    /// Arbitrary auth/handshake info.
    #[prost(bytes = "vec", tag = "2")]
    pub payload: ::prost::alloc::vec::Vec<u8>,
}
///
/// A message for doing simple auth.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct BasicAuth {
    #[prost(string, tag = "2")]
    pub username: ::prost::alloc::string::String,
    #[prost(string, tag = "3")]
    pub password: ::prost::alloc::string::String,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct Empty {}
///
/// Describes an available action, including both the name used for execution
/// along with a short description of the purpose of the action.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ActionType {
    #[prost(string, tag = "1")]
    pub r#type: ::prost::alloc::string::String,
    #[prost(string, tag = "2")]
    pub description: ::prost::alloc::string::String,
}
///
/// A service specific expression that can be used to return a limited set
/// of available Arrow Flight streams.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct Criteria {
    #[prost(bytes = "vec", tag = "1")]
    pub expression: ::prost::alloc::vec::Vec<u8>,
}
///
/// An opaque action specific for the service.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct Action {
    #[prost(string, tag = "1")]
    pub r#type: ::prost::alloc::string::String,
    #[prost(bytes = "vec", tag = "2")]
    pub body: ::prost::alloc::vec::Vec<u8>,
}
///
/// An opaque result returned after executing an action.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct Result {
    #[prost(bytes = "vec", tag = "1")]
    pub body: ::prost::alloc::vec::Vec<u8>,
}
///
/// Wrap the result of a getSchema call
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SchemaResult {
    /// The schema of the dataset in its IPC form:
    ///    4 bytes - an optional IPC_CONTINUATION_TOKEN prefix
    ///    4 bytes - the byte length of the payload
    ///    a flatbuffer Message whose header is the Schema
    #[prost(bytes = "vec", tag = "1")]
    pub schema: ::prost::alloc::vec::Vec<u8>,
}
///
/// The name or tag for a Flight. May be used as a way to retrieve or generate
/// a flight or be used to expose a set of previously defined flights.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct FlightDescriptor {
    #[prost(enumeration = "flight_descriptor::DescriptorType", tag = "1")]
    pub r#type: i32,
    ///
    /// Opaque value used to express a command. Should only be defined when
    /// type = CMD.
    #[prost(bytes = "vec", tag = "2")]
    pub cmd: ::prost::alloc::vec::Vec<u8>,
    ///
    /// List of strings identifying a particular dataset. Should only be defined
    /// when type = PATH.
    #[prost(string, repeated, tag = "3")]
    pub path: ::prost::alloc::vec::Vec<::prost::alloc::string::String>,
}
/// Nested message and enum types in `FlightDescriptor`.
pub mod flight_descriptor {
    ///
    /// Describes what type of descriptor is defined.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
    #[repr(i32)]
    pub enum DescriptorType {
        /// Protobuf pattern, not used.
        Unknown = 0,
        ///
        /// A named path that identifies a dataset. A path is composed of a string
        /// or list of strings describing a particular dataset. This is conceptually
        ///   similar to a path inside a filesystem.
        Path = 1,
        ///
        /// An opaque command to generate a dataset.
        Cmd = 2,
    }
    impl DescriptorType {
        /// String value of the enum field names used in the ProtoBuf definition.
        ///
        /// The values are not transformed in any way and thus are considered stable
        /// (if the ProtoBuf definition does not change) and safe for programmatic use.
        pub fn as_str_name(&self) -> &'static str {
            match self {
                DescriptorType::Unknown => "UNKNOWN",
                DescriptorType::Path => "PATH",
                DescriptorType::Cmd => "CMD",
            }
        }
    }
}
///
/// The access coordinates for retrieval of a dataset. With a FlightInfo, a
/// consumer is able to determine how to retrieve a dataset.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct FlightInfo {
    /// The schema of the dataset in its IPC form:
    ///    4 bytes - an optional IPC_CONTINUATION_TOKEN prefix
    ///    4 bytes - the byte length of the payload
    ///    a flatbuffer Message whose header is the Schema
    #[prost(bytes = "vec", tag = "1")]
    pub schema: ::prost::alloc::vec::Vec<u8>,
    ///
    /// The descriptor associated with this info.
    #[prost(message, optional, tag = "2")]
    pub flight_descriptor: ::core::option::Option<FlightDescriptor>,
    ///
    /// A list of endpoints associated with the flight. To consume the
    /// whole flight, all endpoints (and hence all Tickets) must be
    /// consumed. Endpoints can be consumed in any order.
    ///
    /// In other words, an application can use multiple endpoints to
    /// represent partitioned data.
    ///
    /// There is no ordering defined on endpoints. Hence, if the returned
    /// data has an ordering, it should be returned in a single endpoint.
    #[prost(message, repeated, tag = "3")]
    pub endpoint: ::prost::alloc::vec::Vec<FlightEndpoint>,
    /// Set these to -1 if unknown.
    #[prost(int64, tag = "4")]
    pub total_records: i64,
    #[prost(int64, tag = "5")]
    pub total_bytes: i64,
}
///
/// A particular stream or split associated with a flight.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct FlightEndpoint {
    ///
    /// Token used to retrieve this stream.
    #[prost(message, optional, tag = "1")]
    pub ticket: ::core::option::Option<Ticket>,
    ///
    /// A list of URIs where this ticket can be redeemed via DoGet().
    ///
    /// If the list is empty, the expectation is that the ticket can only
    /// be redeemed on the current service where the ticket was
    /// generated.
    ///
    /// If the list is not empty, the expectation is that the ticket can
    /// be redeemed at any of the locations, and that the data returned
    /// will be equivalent. In this case, the ticket may only be redeemed
    /// at one of the given locations, and not (necessarily) on the
    /// current service.
    ///
    /// In other words, an application can use multiple locations to
    /// represent redundant and/or load balanced services.
    #[prost(message, repeated, tag = "2")]
    pub location: ::prost::alloc::vec::Vec<Location>,
}
///
/// A location where a Flight service will accept retrieval of a particular
/// stream given a ticket.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct Location {
    #[prost(string, tag = "1")]
    pub uri: ::prost::alloc::string::String,
}
///
/// An opaque identifier that the service can use to retrieve a particular
/// portion of a stream.
///
/// Tickets are meant to be single use. It is an error/application-defined
/// behavior to reuse a ticket.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct Ticket {
    #[prost(bytes = "vec", tag = "1")]
    pub ticket: ::prost::alloc::vec::Vec<u8>,
}
///
/// A batch of Arrow data as part of a stream of batches.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct FlightData {
    ///
    /// The descriptor of the data. This is only relevant when a client is
    /// starting a new DoPut stream.
    #[prost(message, optional, tag = "1")]
    pub flight_descriptor: ::core::option::Option<FlightDescriptor>,
    ///
    /// Header for message data as described in Message.fbs::Message.
    #[prost(bytes = "vec", tag = "2")]
    pub data_header: ::prost::alloc::vec::Vec<u8>,
    ///
    /// Application-defined metadata.
    #[prost(bytes = "vec", tag = "3")]
    pub app_metadata: ::prost::alloc::vec::Vec<u8>,
    ///
    /// The actual batch of Arrow data. Preferably handled with minimal-copies
    /// coming last in the definition to help with sidecar patterns (it is
    /// expected that some implementations will fetch this field off the wire
    /// with specialized code to avoid extra memory copies).
    #[prost(bytes = "vec", tag = "1000")]
    pub data_body: ::prost::alloc::vec::Vec<u8>,
}
/// *
/// The response message associated with the submission of a DoPut.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct PutResult {
    #[prost(bytes = "vec", tag = "1")]
    pub app_metadata: ::prost::alloc::vec::Vec<u8>,
}
