#pragma once

namespace NYT::NYson::NDetail {

////////////////////////////////////////////////////////////////////////////////

//! Indicates the beginning of a list.
constexpr char BeginListSymbol = '[';
//! Indicates the end of a list.
constexpr char EndListSymbol = ']';

//! Indicates the beginning of a map.
constexpr char BeginMapSymbol = '{';
//! Indicates the end of a map.
constexpr char EndMapSymbol = '}';

//! Indicates the beginning of an attribute map.
constexpr char BeginAttributesSymbol = '<';
//! Indicates the end of an attribute map.
constexpr char EndAttributesSymbol = '>';

//! Separates items in lists, maps, attributes.
constexpr char ItemSeparatorSymbol = ';';
//! Separates keys from values in maps.
constexpr char KeyValueSeparatorSymbol = '=';

//! Indicates an entity.
constexpr char EntitySymbol = '#';
//! Marks the beginning of a binary string literal.
constexpr char StringMarker = '\x01';
//! Marks the beginning of a binary i64 literal.
constexpr char Int64Marker = '\x02';
//! Marks the beginning of a binary double literal.
constexpr char DoubleMarker = '\x03';
//! Marks |false| boolean value.
constexpr char FalseMarker = '\x04';
//! Marks |true| boolean value.
constexpr char TrueMarker = '\x05';
//! Marks the beginning of a binary ui64 literal.
constexpr char Uint64Marker = '\x06';

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT::NYson::NDetail
