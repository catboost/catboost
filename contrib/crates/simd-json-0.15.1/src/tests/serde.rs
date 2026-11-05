#![allow(
    clippy::unnecessary_operation,
    clippy::non_ascii_literal,
    clippy::ignored_unit_patterns
)]
use crate::{
    Deserializer,
    owned::{Object, Value, to_value},
    prelude::*,
    serde::from_slice,
    to_borrowed_value, to_owned_value,
};
#[cfg(not(target_arch = "wasm32"))]
use crate::{OwnedValue, deserialize};
use halfbrown::HashMap;
#[cfg(not(target_arch = "wasm32"))]
use proptest::prelude::*;
use serde::Deserialize;
use serde_ext::Serialize;

static JSON: &str = r#"{
    "code": 200,
    "success": true,
    "payload": {
        "features": [
            "serde",
            "json"
        ]
    }
}"#;

#[derive(Deserialize, PartialEq, Debug)]
struct TestPayload {
    features: Vec<String>,
}

#[derive(Deserialize, PartialEq, Debug)]
struct TestEnvelope {
    code: usize,
    success: bool,
    payload: TestPayload,
}

#[test]
fn test_deser_to_value() {
    let mut json = JSON.as_bytes().to_vec();
    let d = Deserializer::from_slice(&mut json).expect("Invalid JSON");

    let original_index = d.idx;
    let original_nodes = d.tape.len();

    let v = d.as_value();

    assert!(v.contains_key("payload"), "Failed to find payload key");
    let v = v.get("payload").expect("Can't get payload");
    assert!(v.is_object(), "payload not recognized as object: {v:?}");
    assert!(v.contains_key("features"), "Failed to find features key");
    let v = v.get("features").expect("can't get features");
    assert!(v.is_array(), "features not recognized as array: {v:?}");

    // proving that value peeking doesn't affect the deserializer

    assert_eq!(
        original_index, d.idx,
        "Deserializer has been internally modified"
    );
    assert_eq!(
        original_nodes,
        d.tape.len(),
        "Deserializer has been internally modified"
    );
}

#[test]
fn test_deser_restart() {
    let mut json = JSON.as_bytes().to_vec();
    let mut d = Deserializer::from_slice(&mut json).expect("Invalid JSON");

    let original_index = d.idx;
    let original_nodes = d.tape.len();

    let test1 = TestEnvelope::deserialize(&mut d).expect("Deserialization failed");

    assert!(
        original_index != d.idx,
        "Deserializer has NOT been internally modified"
    );
    assert_eq!(
        original_nodes,
        d.tape.len(),
        "Deserializer nodes are NOT intact"
    );

    d.restart();

    let test2 = TestEnvelope::deserialize(&mut d).expect("Deserialization failed");

    assert_eq!(test2, test1, "Deserializer is not idempotent");
}

#[test]
fn empty() {
    let mut d = String::new();
    let d = unsafe { d.as_bytes_mut() };
    let v_simd = from_slice::<Value>(d);
    let v_serde = serde_json::from_slice::<Value>(d);
    assert!(v_simd.is_err());
    assert!(v_serde.is_err());
}

#[test]
fn bool_true() {
    let mut d = String::from("true");
    let mut d1 = d.clone();
    let d1 = unsafe { d1.as_bytes_mut() };
    let d = unsafe { d.as_bytes_mut() };

    let v_serde: serde_json::Value = serde_json::from_slice(d).expect("");
    let v_simd: serde_json::Value = from_slice(d).expect("");
    assert_eq!(v_simd, v_serde);
    assert_eq!(to_value(d1), Ok(Value::from(true)));
}

#[test]
fn bool_false() {
    let mut d = String::from("false");
    let mut d1 = d.clone();
    let d1 = unsafe { d1.as_bytes_mut() };
    let d = unsafe { d.as_bytes_mut() };
    let v_serde: serde_json::Value = serde_json::from_slice(d).expect("");
    let v_simd: serde_json::Value = from_slice(d).expect("");
    assert_eq!(v_simd, v_serde);
    assert_eq!(to_value(d1), Ok(Value::from(false)));
    //assert!(false)
}

#[test]
fn union() {
    let mut d = String::from("null");
    let mut d1 = d.clone();
    let d1 = unsafe { d1.as_bytes_mut() };
    let d = unsafe { d.as_bytes_mut() };
    let v_serde: serde_json::Value = serde_json::from_slice(d).expect("");
    let v_simd: serde_json::Value = from_slice(d).expect("");
    assert_eq!(v_simd, v_serde);
    assert_eq!(to_value(d1), Ok(Value::Static(StaticNode::Null)));
}

#[test]
fn int() {
    let mut d = String::from("42");
    let mut d1 = d.clone();
    let d1 = unsafe { d1.as_bytes_mut() };
    let d = unsafe { d.as_bytes_mut() };
    let v_serde: serde_json::Value = serde_json::from_slice(d).expect("");
    let v_simd: serde_json::Value = from_slice(d).expect("");
    assert_eq!(v_simd, v_serde);
    assert_eq!(to_value(d1), Ok(Value::from(42)));
}

#[test]
fn zero() {
    let mut d = String::from("0");
    let mut d1 = d.clone();
    let d1 = unsafe { d1.as_bytes_mut() };
    let d = unsafe { d.as_bytes_mut() };
    let v_serde: serde_json::Value = serde_json::from_slice(d).expect("");
    let v_simd: serde_json::Value = from_slice(d).expect("");
    assert_eq!(v_simd, v_serde);
    assert_eq!(to_value(d1), Ok(Value::from(0)));
}

#[test]
fn one() {
    let mut d = String::from("1");
    let mut d1 = d.clone();
    let d1 = unsafe { d1.as_bytes_mut() };
    let d = unsafe { d.as_bytes_mut() };
    let v_serde: serde_json::Value = serde_json::from_slice(d).expect("");
    let v_simd: serde_json::Value = from_slice(d).expect("");
    assert_eq!(v_simd, v_serde);
    assert_eq!(to_value(d1), Ok(Value::from(1)));
}

#[test]
fn minus_one() {
    let mut d = String::from("-1");
    let mut d1 = d.clone();
    let d1 = unsafe { d1.as_bytes_mut() };
    let d = unsafe { d.as_bytes_mut() };
    let v_serde: serde_json::Value = serde_json::from_slice(d).expect("");
    let v_simd: serde_json::Value = from_slice(d).expect("");
    assert_eq!(v_simd, v_serde);
    assert_eq!(to_value(d1), Ok(Value::from(-1)));
}

#[test]
fn float() {
    let mut d = String::from("23.0");
    let mut d1 = d.clone();
    let d1 = unsafe { d1.as_bytes_mut() };
    let d = unsafe { d.as_bytes_mut() };
    let v_serde: serde_json::Value = serde_json::from_slice(d).expect("");
    let v_simd: serde_json::Value = from_slice(d).expect("");
    assert_eq!(v_simd, v_serde);
    assert_eq!(to_value(d1), Ok(Value::from(23.0)));
}

#[test]
fn string() {
    let mut d = String::from(r#""snot""#);
    let mut d1 = d.clone();
    let d1 = unsafe { d1.as_bytes_mut() };
    let d = unsafe { d.as_bytes_mut() };
    let v_serde: serde_json::Value = serde_json::from_slice(d).expect("");
    let v_simd: serde_json::Value = from_slice(d).expect("");
    assert_eq!(to_value(d1), Ok(Value::from("snot")));
    assert_eq!(v_simd, v_serde);
}

#[test]
fn lonely_quote() {
    let mut d = String::from(r#"""#);
    let d = unsafe { d.as_bytes_mut() };
    let v_serde = serde_json::from_slice::<serde_json::Value>(d).is_err();
    let v_simd = from_slice::<serde_json::Value>(d).is_err();
    assert!(v_simd);
    assert!(v_serde);
}

#[test]
fn lonely_quote1() {
    let mut d = String::from(r#"["]"#);
    let d = unsafe { d.as_bytes_mut() };
    let v_serde = serde_json::from_slice::<serde_json::Value>(d).is_err();
    let v_simd = from_slice::<serde_json::Value>(d).is_err();
    assert!(v_simd);
    assert!(v_serde);
}
#[test]
fn lonely_quote2() {
    let mut d = String::from(r#"[1, "]"#);
    let d = unsafe { d.as_bytes_mut() };
    let v_serde = serde_json::from_slice::<serde_json::Value>(d).is_err();
    let v_simd = from_slice::<serde_json::Value>(d).is_err();
    assert!(v_simd);
    assert!(v_serde);
}

#[test]
fn lonely_quote3() {
    let mut d = String::from(r#"{": 1}"#);
    let d = unsafe { d.as_bytes_mut() };
    let v_serde = serde_json::from_slice::<serde_json::Value>(d).is_err();
    let v_simd = from_slice::<serde_json::Value>(d).is_err();
    assert!(v_simd);
    assert!(v_serde);
}

#[test]
fn empty_string() {
    let mut d = String::from(r#""""#);
    let mut d1 = d.clone();
    let d1 = unsafe { d1.as_bytes_mut() };
    let d = unsafe { d.as_bytes_mut() };
    let v_serde: serde_json::Value = serde_json::from_slice(d).expect("");
    let v_simd: serde_json::Value = from_slice(d).expect("");
    assert_eq!(to_value(d1), Ok(Value::from("")));
    assert_eq!(v_simd, v_serde);
}

#[test]
fn empty_array() {
    let mut d = String::from("[]");
    let mut d1 = d.clone();
    let d1 = unsafe { d1.as_bytes_mut() };
    let d = unsafe { d.as_bytes_mut() };
    let v_serde: serde_json::Value = serde_json::from_slice(d).expect("parse_serde");
    let v_simd: serde_json::Value = from_slice(d).expect("parse_simd");
    assert_eq!(to_value(d1), Ok(Value::array()));
    assert_eq!(v_simd, v_serde);
}

#[test]
fn malformed_array() {
    let mut d = String::from("[[");
    let mut d1 = d.clone();
    let mut d2 = d.clone();
    let d = unsafe { d.as_bytes_mut() };
    let d1 = unsafe { d1.as_bytes_mut() };
    let d2 = unsafe { d2.as_bytes_mut() };
    let v_serde: Result<serde_json::Value, _> = serde_json::from_slice(d);
    let v_simd_owned_value = to_owned_value(d);
    let v_simd_borrowed_value = to_borrowed_value(d1);
    let v_simd: Result<serde_json::Value, _> = from_slice(d2);
    assert!(v_simd_owned_value.is_err());
    assert!(v_simd_borrowed_value.is_err());
    assert!(v_simd.is_err());
    assert!(v_serde.is_err());
}

#[test]
fn double_array() {
    let mut d = String::from("[[]]");
    let mut d1 = d.clone();
    let d1 = unsafe { d1.as_bytes_mut() };
    let d = unsafe { d.as_bytes_mut() };
    let v_serde: serde_json::Value = serde_json::from_slice(d).expect("parse_serde");
    let v_simd: serde_json::Value = from_slice(d).expect("parse_simd");
    assert_eq!(to_value(d1), Ok(Value::from(vec![Value::array()])));
    assert_eq!(v_simd, v_serde);
}

#[test]
fn null_null_array() {
    let mut d = String::from("[[],null,null]");
    let mut d1 = d.clone();
    let d1 = unsafe { d1.as_bytes_mut() };
    let d = unsafe { d.as_bytes_mut() };
    let v_serde: serde_json::Value = serde_json::from_slice(d).expect("parse_serde");
    let v_simd: serde_json::Value = from_slice(d).expect("parse_simd");
    assert_eq!(
        to_value(d1),
        Ok(Value::from(vec![
            Value::array(),
            Value::Static(StaticNode::Null),
            Value::Static(StaticNode::Null),
        ]))
    );
    assert_eq!(v_simd, v_serde);
}

#[test]
fn one_element_array() {
    let mut d = String::from(r#"["snot"]"#);
    let mut d1 = d.clone();
    let d1 = unsafe { d1.as_bytes_mut() };
    let d = unsafe { d.as_bytes_mut() };
    assert_eq!(to_value(d1), Ok(Value::from(vec![Value::from("snot")])));
    let v_serde: serde_json::Value = serde_json::from_slice(d).expect("");
    let v_simd: serde_json::Value = from_slice(d).expect("");
    assert_eq!(v_simd, v_serde);
}

#[test]
fn two_element_array() {
    let mut d = String::from(r#"["snot", "badger"]"#);
    let mut d1 = d.clone();
    let d1 = unsafe { d1.as_bytes_mut() };
    let d = unsafe { d.as_bytes_mut() };
    assert_eq!(
        to_value(d1),
        Ok(Value::from(vec![
            Value::from("snot"),
            Value::from("badger")
        ]))
    );
    let v_serde: serde_json::Value = serde_json::from_slice(d).expect("");
    let v_simd: serde_json::Value = from_slice(d).expect("");
    assert_eq!(v_simd, v_serde);
}

#[test]
fn list() {
    let mut d = String::from(r#"[42, 23.0, "snot badger"]"#);
    let mut d1 = d.clone();
    let d1 = unsafe { d1.as_bytes_mut() };
    let d = unsafe { d.as_bytes_mut() };
    let v_serde: serde_json::Value = serde_json::from_slice(d).expect("");
    let v_simd: serde_json::Value = from_slice(d).expect("");
    assert_eq!(v_simd, v_serde);
    assert_eq!(
        to_value(d1),
        Ok(Value::from(vec![
            Value::from(42),
            Value::from(23.0),
            Value::from("snot badger")
        ]))
    );
}

#[test]
fn nested_list1() {
    let mut d = String::from(r#"[42, [23.0, "snot"], "bad", "ger"]"#);
    let mut d1 = d.clone();
    let d1 = unsafe { d1.as_bytes_mut() };
    let d = unsafe { d.as_bytes_mut() };
    assert_eq!(
        to_value(d1),
        Ok(Value::from(vec![
            Value::from(42),
            Value::from(vec![Value::from(23.0), Value::from("snot")]),
            Value::from("bad"),
            Value::from("ger")
        ]))
    );

    let v_serde: serde_json::Value = serde_json::from_slice(d).expect("");
    let v_simd: serde_json::Value = from_slice(d).expect("");
    assert_eq!(v_simd, v_serde);
}

#[test]
fn nested_list2() {
    let mut d = String::from(r#"[42, [23.0, "snot"], {"bad": "ger"}]"#);
    let d = unsafe { d.as_bytes_mut() };
    let v_serde: serde_json::Value = serde_json::from_slice(d).expect("");
    let v_simd: serde_json::Value = from_slice(d).expect("");
    assert_eq!(v_simd, v_serde);
}

#[test]
fn utf8() {
    let mut d = String::from(r#""\u000e""#);
    let d = unsafe { d.as_bytes_mut() };
    let v_simd: serde_json::Value = from_slice(d).expect("");
    assert_eq!(v_simd, "\u{e}");
    // NOTE: serde is broken for this
    //assert_eq!(v_serde, "\u{e}");
    //assert_eq!(v_simd, v_serde);
}
#[test]
fn utf8_invalid_surrogates() {
    // This is invalid UTF-8, the first character is a high surrogate
    let mut d = String::from(r#""\uDE71""#);
    let d = unsafe { d.as_bytes_mut() };
    let v_simd: Result<serde_json::Value, _> = from_slice(d);
    assert!(v_simd.is_err());
}

#[test]
fn unicode() {
    let mut d = String::from(r#""¡\"""#);
    let d = unsafe { d.as_bytes_mut() };
    let v_serde: serde_json::Value = serde_json::from_slice(d).expect("");
    let v_simd: serde_json::Value = from_slice(d).expect("");
    assert_eq!(v_simd, v_serde);
}

#[test]
fn odd_array() {
    let mut d = String::from("[{},null]");
    let mut d1 = d.clone();
    let d1 = unsafe { d1.as_bytes_mut() };
    let d = unsafe { d.as_bytes_mut() };
    let v_serde: serde_json::Value = serde_json::from_slice(d).expect("");
    let v_simd: serde_json::Value = from_slice(d).expect("");
    assert_eq!(v_simd, v_serde);
    assert_eq!(
        to_value(d1),
        Ok(Value::from(vec![
            Value::from(Object::default()),
            Value::Static(StaticNode::Null)
        ]))
    );
}

#[test]
fn min_i64() {
    let mut d =
        String::from(r#"{"name": "max_unsafe_auto_id_timestamp", "value": -9223372036854775808}"#);

    let mut d1 = d.clone();
    let d1 = unsafe { d1.as_bytes_mut() };
    let d = unsafe { d.as_bytes_mut() };
    let v_serde: serde_json::Value = serde_json::from_slice(d).expect("");
    let v_simd: serde_json::Value = from_slice(d).expect("");
    assert_eq!(v_simd, v_serde);
    let mut o = Value::object();
    o.insert("name", "max_unsafe_auto_id_timestamp")
        .expect("failed to set key");
    o.insert("value", -9_223_372_036_854_775_808_i64)
        .expect("failed to set key");
    assert_eq!(to_value(d1), Ok(o));
}

#[test]
fn map2() {
    let mut d = String::from(r#"[{"\u0000":null}]"#);
    let d = unsafe { d.as_bytes_mut() };
    let v_serde: serde_json::Value = serde_json::from_slice(d).expect("");
    let v_simd: serde_json::Value = from_slice(d).expect("");
    assert_eq!(v_simd, v_serde);
}

#[test]
fn null() {
    let mut d = String::from("null");
    let mut d1 = d.clone();
    let d1 = unsafe { d1.as_bytes_mut() };
    let d = unsafe { d.as_bytes_mut() };
    assert_eq!(to_value(d1), Ok(Value::Static(StaticNode::Null)));
    let v_serde: serde_json::Value = serde_json::from_slice(d).expect("");
    let v_simd: serde_json::Value = from_slice(d).expect("");
    assert_eq!(v_simd, v_serde);
}
#[test]
fn null_null() {
    let mut d = String::from("[null, null]");
    let mut d1 = d.clone();
    let d1 = unsafe { d1.as_bytes_mut() };
    let d = unsafe { d.as_bytes_mut() };
    assert_eq!(
        to_value(d1),
        Ok(Value::from(vec![
            Value::Static(StaticNode::Null),
            Value::Static(StaticNode::Null),
        ]))
    );
    let v_serde: serde_json::Value = serde_json::from_slice(d).expect("");
    let v_simd: serde_json::Value = from_slice(d).expect("");
    assert_eq!(v_simd, v_serde);
}

#[test]
fn nested_null() {
    let mut d = String::from("[[null, null]]");
    let mut d1 = d.clone();
    let d1 = unsafe { d1.as_bytes_mut() };
    let d = unsafe { d.as_bytes_mut() };
    assert_eq!(
        to_value(d1),
        Ok(Value::from(vec![Value::from(vec![
            Value::Static(StaticNode::Null),
            Value::Static(StaticNode::Null),
        ])]))
    );

    let v_serde: serde_json::Value = serde_json::from_slice(d).expect("");
    let v_simd: serde_json::Value = from_slice(d).expect("");
    assert_eq!(v_simd, v_serde);
}

#[test]
fn nestednested_null() {
    let mut d = String::from("[[[null, null]]]");
    let mut d1 = d.clone();
    let d1 = unsafe { d1.as_bytes_mut() };
    let d = unsafe { d.as_bytes_mut() };
    let v_serde: serde_json::Value = serde_json::from_slice(d).expect("");
    let v_simd: serde_json::Value = from_slice(d).expect("");
    assert_eq!(v_simd, v_serde);
    assert_eq!(
        to_value(d1),
        Ok(Value::from(vec![Value::from(vec![Value::from(vec![
            Value::Static(StaticNode::Null),
            Value::Static(StaticNode::Null),
        ])])]))
    );
}

#[test]
fn odd_array2() {
    let mut d = String::from("[[\"\\u0000\\\"\"]]");
    let d = unsafe { d.as_bytes_mut() };
    let v_serde: serde_json::Value = serde_json::from_slice(d).expect("");
    let v_simd: serde_json::Value = from_slice(d).expect("");
    assert_eq!(v_simd, v_serde);
}

#[test]
fn odd_array3() {
    let mut d = String::from("[{\"\\u0000\\u0000\":null}]");
    let d = unsafe { d.as_bytes_mut() };
    let v_serde: serde_json::Value = serde_json::from_slice(d).expect("");
    let v_simd: serde_json::Value = from_slice(d).expect("");
    assert_eq!(v_simd, v_serde);
}

#[test]
fn odd_array4() {
    let mut d = String::from("[{\"\\u0000𐀀a\":null}]");
    let d = unsafe { d.as_bytes_mut() };
    let v_serde: serde_json::Value = serde_json::from_slice(d).expect("");
    let v_simd: serde_json::Value = from_slice(d).expect("");
    assert_eq!(v_simd, v_serde);
}

#[test]
fn float1() {
    let mut d = String::from("2.3250706903316115e307");
    let d = unsafe { d.as_bytes_mut() };
    let v_serde: serde_json::Value = serde_json::from_slice(d).expect("serde_json");
    let v_simd: serde_json::Value = from_slice(d).expect("simd_json");
    assert_eq!(v_simd, v_serde);
}

// We ignore this since serde is less precise on this test
#[ignore]
#[test]
fn float2() {
    let mut d = String::from("-4.5512678569607477e306");
    let d = unsafe { d.as_bytes_mut() };
    let v_serde: serde_json::Value = serde_json::from_slice(d).expect("serde_json");
    let v_simd: serde_json::Value = from_slice(d).expect("simd_json");
    assert_eq!(v_simd, v_serde);
}

#[cfg(not(feature = "approx-number-parsing"))]
#[test]
fn float3() {
    let mut d = String::from("0.6");
    let d = unsafe { d.as_bytes_mut() };
    let v_serde: serde_json::Number = serde_json::from_slice(d).expect("serde_json");
    let v_simd: serde_json::Number = from_slice(d).expect("simd_json");
    assert_eq!(v_simd, v_serde);
}

#[test]
fn map0() {
    let mut d = String::from(r#"{"snot": "badger"}"#);
    let mut d1 = d.clone();
    let d1 = unsafe { d1.as_bytes_mut() };
    let d = unsafe { d.as_bytes_mut() };
    let v_serde: serde_json::Value = serde_json::from_slice(d).expect("");
    let v_simd: serde_json::Value = from_slice(d).expect("");
    assert_eq!(v_simd, v_serde);
    let mut h = Object::default();
    h.insert("snot".into(), Value::from("badger"));
    assert_eq!(to_value(d1), Ok(Value::from(h)));
}

#[test]
fn map1() {
    let mut d = String::from(r#"{"snot": "badger", "badger": "snot"}"#);
    let mut d1 = d.clone();
    let d1 = unsafe { d1.as_bytes_mut() };
    let d = unsafe { d.as_bytes_mut() };
    let v_serde: serde_json::Value = serde_json::from_slice(d).expect("");
    let v_simd: serde_json::Value = from_slice(d).expect("");
    assert_eq!(v_simd, v_serde);
    let mut h = Object::default();
    h.insert("snot".into(), Value::from("badger"));
    h.insert("badger".into(), Value::from("snot"));
    assert_eq!(to_value(d1), Ok(Value::from(h)));
}

#[test]
fn tpl1() {
    let mut d = String::from("[-65.613616999999977, 43.420273000000009]");
    let d = unsafe { d.as_bytes_mut() };
    let v_serde: (f32, f32) = serde_json::from_slice(d).expect("serde_json");
    let v_simd: (f32, f32) = from_slice(d).expect("simd_json");
    assert_eq!(v_simd, v_serde);
}

#[test]
fn tpl2() {
    let mut d = String::from("[[-65.613616999999977, 43.420273000000009]]");
    let d = unsafe { d.as_bytes_mut() };
    let v_serde: Vec<(f32, f32)> = serde_json::from_slice(d).expect("serde_json");
    let v_simd: Vec<(f32, f32)> = from_slice(d).expect("simd_json");
    assert_eq!(v_simd, v_serde);
}

#[test]
fn tpl3() {
    let mut d = String::from(
        "[[-65.613616999999977,43.420273000000009], [-65.613616999999977,43.420273000000009]]",
    );
    let d = unsafe { d.as_bytes_mut() };
    let v_serde: Vec<(f32, f32)> = serde_json::from_slice(d).expect("serde_json");
    let v_simd: Vec<(f32, f32)> = from_slice(d).expect("simd_json");
    assert_eq!(v_simd, v_serde);
}
#[test]
fn tpl4() {
    let mut d = String::from("[[[-65.613616999999977,43.420273000000009]]]");
    let d = unsafe { d.as_bytes_mut() };
    let v_serde: Vec<Vec<(f32, f32)>> = serde_json::from_slice(d).expect("serde_json");
    let v_simd: Vec<Vec<(f32, f32)>> = from_slice(d).expect("simd_json");
    assert_eq!(v_simd, v_serde);
}
#[test]
fn tpl5() {
    let mut d = String::from(
        "[[[-65.613616999999977,43.420273000000009], [-65.613616999999977,43.420273000000009]]]",
    );
    let d = unsafe { d.as_bytes_mut() };
    let v_serde: Vec<Vec<(f32, f32)>> = serde_json::from_slice(d).expect("serde_json");
    let v_simd: Vec<Vec<(f32, f32)>> = from_slice(d).expect("simd_json");
    assert_eq!(v_simd, v_serde);
}

#[test]
fn tpl6() {
    let mut d = String::from(
        "[[[[-65.613616999999977,43.420273000000009], [-65.613616999999977,43.420273000000009]]]]",
    );
    let d = unsafe { d.as_bytes_mut() };
    let v_serde: Vec<Vec<Vec<(f32, f32)>>> = serde_json::from_slice(d).expect("serde_json");
    let v_simd: Vec<Vec<Vec<(f32, f32)>>> = from_slice(d).expect("simd_json");
    assert_eq!(v_simd, v_serde);
}

#[test]
fn tpl7() {
    let mut d = String::from(
        "[[[[-65.613616999999977,43.420273000000009], [-65.613616999999977,43.420273000000009]]]]",
    );
    let d = unsafe { d.as_bytes_mut() };
    let v_serde: Vec<Vec<Vec<[f32; 2]>>> = serde_json::from_slice(d).expect("serde_json");
    let v_simd: Vec<Vec<Vec<[f32; 2]>>> = from_slice(d).expect("simd_json");
    assert_eq!(v_simd, v_serde);
}

#[derive(Deserialize, PartialEq, Debug)]
struct Obj {
    a: u64,
    b: u64,
}

#[derive(Deserialize, PartialEq, Debug)]
struct Obj1 {
    a: Obj,
}

#[test]
fn obj1() {
    let mut d = String::from(r#"{"a": 1, "b":1}"#);
    let d = unsafe { d.as_bytes_mut() };
    let v_serde: Obj = serde_json::from_slice(d).expect("serde_json");
    let v_simd: Obj = from_slice(d).expect("simd_json");
    assert_eq!(v_simd, v_serde);
}

#[test]
fn obj2() {
    let mut d =
        String::from(r#"{"a": {"a": 1, "b":1}, "b": {"a": 1, "b":1}, "c": {"a": 1, "b": 1}}"#);
    let d = unsafe { d.as_bytes_mut() };
    let v_serde: HashMap<String, Obj> = serde_json::from_slice(d).expect("serde_json");
    let v_simd: HashMap<String, Obj> = from_slice(d).expect("simd_json");
    assert_eq!(v_simd, v_serde);
}

#[test]
fn obj3() {
    let mut d = String::from(
        r#"{"c": {"a": {"a": 1, "b":1}, "b": {"a": 1, "b":1}, "c": {"a": 1, "b": 1}}}"#,
    );
    let d = unsafe { d.as_bytes_mut() };
    let v_serde: HashMap<String, HashMap<String, Obj>> =
        serde_json::from_slice(d).expect("serde_json");
    let v_simd: HashMap<String, HashMap<String, Obj>> = from_slice(d).expect("simd_json");
    assert_eq!(v_simd, v_serde);
}

#[test]
fn obj4() {
    let mut d = String::from(r#"{"c": {"a": {"a": 1, "b":1}}}"#);
    let d = unsafe { d.as_bytes_mut() };
    let v_serde: HashMap<String, Obj1> = serde_json::from_slice(d).expect("serde_json");
    let v_simd: HashMap<String, Obj1> = from_slice(d).expect("simd_json");
    assert_eq!(v_simd, v_serde);
}

#[test]
fn vecvec() {
    let mut d = String::from(
        "[[[-65.613616999999977,43.420273000000009], [-65.613616999999977,43.420273000000009]], [[-65.613616999999977,43.420273000000009], [-65.613616999999977,43.420273000000009]]]",
    );
    let d = unsafe { d.as_bytes_mut() };
    let v_serde: Vec<Vec<(f32, f32)>> = serde_json::from_slice(d).expect("serde_json");
    let v_simd: Vec<Vec<(f32, f32)>> = from_slice(d).expect("simd_json");
    assert_eq!(v_simd, v_serde);
}

#[test]
fn invalid_float_array() {
    let mut data = b"[11111111111111111111111111111E1,-111111111111111111111E111111111".to_vec();

    assert!(to_owned_value(&mut data).is_err());
}

#[test]
fn crazy_string() {
    // there is unicode in here!
    let d = "\"𐀀𐀀  𐀀𐀀0 𐀀A\\u00000A0 A \\u000b\"";
    let mut d = String::from(d);
    let d = unsafe { d.as_bytes_mut() };
    let v_serde: serde_json::Value = serde_json::from_slice(d).expect("serde_json");
    let v_simd: serde_json::Value = from_slice(d).expect("simd_json");
    assert_eq!(v_simd, v_serde);
}

#[test]
fn event() {
    #[derive(Deserialize, Debug, PartialEq, Eq)]
    #[serde(deny_unknown_fields, rename_all = "camelCase")]
    pub struct CitmCatalog {
        pub area_names: HashMap<String, String>,
        pub audience_sub_category_names: HashMap<String, String>,
        pub block_names: HashMap<String, String>,
        pub events: HashMap<String, Event>,
    }
    pub type Id = u32;
    #[derive(Deserialize, Debug, PartialEq, Eq)]
    #[serde(deny_unknown_fields, rename_all = "camelCase")]
    pub struct Event {
        pub description: (),
        pub id: Id,
        pub logo: Option<String>,
        pub name: String,
        pub sub_topic_ids: Vec<Id>,
        pub subject_code: (),
        pub subtitle: (),
        pub topic_ids: Vec<Id>,
    }

    let mut d = String::from(
        r#"
{
    "areaNames": {
        "205705993": "Arrière-scène central",
        "205705994": "1er balcon central",
        "205705995": "2ème balcon bergerie cour",
        "205705996": "2ème balcon bergerie jardin",
        "205705998": "1er balcon bergerie jardin",
        "205705999": "1er balcon bergerie cour",
        "205706000": "Arrière-scène jardin",
        "205706001": "Arrière-scène cour",
        "205706002": "2ème balcon jardin",
        "205706003": "2ème balcon cour",
        "205706004": "2ème Balcon central",
        "205706005": "1er balcon jardin",
        "205706006": "1er balcon cour",
        "205706007": "Orchestre central",
        "205706008": "Orchestre jardin",
        "205706009": "Orchestre cour",
        "342752287": "Zone physique secrète"
    },
    "audienceSubCategoryNames": {
        "337100890": "Abonné"
    },
    "blockNames": {},
  "events": {
    "138586341": {
      "description": null,
      "id": 138586341,
      "logo": null,
      "name": "30th Anniversary Tour",
      "subTopicIds": [
        337184269,
        337184283
      ],
      "subjectCode": null,
      "subtitle": null,
      "topicIds": [
        324846099,
        107888604
      ]
    },
    "138586345": {
      "description": null,
      "id": 138586345,
      "logo": "/images/UE0AAAAACEKo6QAAAAZDSVRN",
      "name": "Berliner Philharmoniker",
      "subTopicIds": [
        337184268,
        337184283,
        337184275
      ],
      "subjectCode": null,
      "subtitle": null,
      "topicIds": [
        324846099,
        107888604,
        324846100
      ]
    }
  }
}
"#,
    );
    let d = unsafe { d.as_bytes_mut() };
    let v_serde: CitmCatalog = serde_json::from_slice(d).expect("serde_json");
    let v_simd: CitmCatalog = from_slice(d).expect("simd_json");
    assert_eq!(v_simd, v_serde);
}

//6.576692109929364e305
#[cfg(not(target_arch = "wasm32"))]
fn arb_json() -> BoxedStrategy<String> {
    let leaf = prop_oneof![
        Just(Value::Static(StaticNode::Null)),
        any::<bool>()
            .prop_map(StaticNode::Bool)
            .prop_map(Value::Static),
        // (-1.0e306f64..1.0e306f64).prop_map(Value::from), // The float parsing of simd and serde are too different
        any::<i64>().prop_map(Value::from),
        ".*".prop_map(Value::from),
    ];
    leaf.prop_recursive(
        8,   // 8 levels deep
        256, // Shoot for maximum size of 256 nodes
        10,  // We put up to 10 items per collection
        |inner| {
            prop_oneof![
                // Take the inner strategy and make the two recursive cases.
                prop::collection::vec(inner.clone(), 0..10).prop_map(Value::from),
                prop::collection::hash_map(".*", inner, 0..10).prop_map(Value::from),
            ]
        },
    )
    .prop_map(|v| serde_json::to_string(&v).expect(""))
    .boxed()
}

#[test]
fn int_map_key() -> Result<(), crate::Error> {
    use std::collections::BTreeMap;

    let mut map = BTreeMap::new();
    map.insert(0, "foo");
    map.insert(1, "bar");
    map.insert(2, "baz");

    assert_eq!(
        r#"{"0":"foo","1":"bar","2":"baz"}"#,
        crate::to_string(&map)?
    );
    Ok(())
}

#[cfg(feature = "serde_impl")]
#[test]
fn enum_test() -> Result<(), crate::Error> {
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
    struct MyStruct {
        field: u8,
    }

    #[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
    enum MyEnum {
        First(MyStruct),
        Second(u8),
    }

    let thing = MyEnum::First(MyStruct { field: 1 });
    let mut ser = crate::serde::to_string(&thing)?;
    println!("Ser {ser:?}");
    let des: MyEnum = unsafe { crate::serde::from_str(&mut ser)? };
    println!("Des {des:?}");
    assert_eq!(thing, des);
    Ok(())
}

#[test]
fn invalid_float() {
    let mut s: Vec<u8> = b"[100,9e999]".to_vec();
    assert!(to_owned_value(&mut s).is_err());
}

#[test]
fn zero_element_touple() {
    #[derive(Serialize, Deserialize, PartialEq, Debug)]
    pub enum FullEnum {
        DoubeEmptyTuple(u8, u8),
        SingeEmptyTuple(u8),
        EmptyTuple(),
        Nothing,
    }

    assert_eq!(
        crate::serde::to_string(&FullEnum::DoubeEmptyTuple(42, 23)).expect("serde"),
        r#"{"DoubeEmptyTuple":[42,23]}"#
    );
    assert_eq!(
        crate::serde::to_string(&FullEnum::SingeEmptyTuple(42)).expect("serde"),
        r#"{"SingeEmptyTuple":42}"#
    );
    assert_eq!(
        crate::serde::to_string(&FullEnum::EmptyTuple()).expect("serde"),
        r#"{"EmptyTuple":[]}"#
    );
    assert_eq!(
        crate::serde::to_string(&FullEnum::Nothing).expect("serde"),
        r#""Nothing""#
    );
}

#[cfg(not(target_arch = "wasm32"))]
proptest! {
    #![proptest_config(ProptestConfig {
        // Setting both fork and timeout is redundant since timeout implies
        // fork, but both are shown for clarity.
        // Disabled for code coverage, enable to track bugs
        // fork: true,
        .. ProptestConfig::default()
    })]

    #[test]
    fn prop_json(d in arb_json()) {
        if let Ok(v_serde) = serde_json::from_slice::<serde_json::Value>(d.as_bytes()) {
            let mut d1 = d.clone();
            let d1 = unsafe{ d1.as_bytes_mut()};
            let v_simd_serde: serde_json::Value = from_slice(d1).expect("");
            // We add our own encoder in here.
            let mut d2 = v_simd_serde.to_string();
            let d2 = unsafe{ d2.as_bytes_mut()};
            let mut d3 = d.clone();
            let d3 = unsafe{ d3.as_bytes_mut()};
            let mut d4 = d.clone();
            let d4 = unsafe{ d4.as_bytes_mut()};
            assert_eq!(v_simd_serde, v_serde);
            let v_simd_owned = to_owned_value(d2).expect("to_owned_value failed");
            let v_simd_borrowed = to_borrowed_value(d3).expect("to_borrowed_value failed");
            assert_eq!(v_simd_borrowed, v_simd_owned);
            let v_deserialize: OwnedValue = deserialize(d4).expect("deserialize failed");
            assert_eq!(v_deserialize, v_simd_owned);
        }

    }

}

#[cfg(not(target_arch = "wasm32"))]
fn arb_junk() -> BoxedStrategy<Vec<u8>> {
    prop::collection::vec(any::<u8>(), 0..(1024 * 8)).boxed()
}
#[cfg(not(target_arch = "wasm32"))]
proptest! {
    #![proptest_config(ProptestConfig {
        // Setting both fork and timeout is redundant since timeout implies
        // fork, but both are shown for clarity.
        // Disabled for code coverage, enable to track bugs
        // fork: true,
        .. ProptestConfig::default()
    })]
    #[test]
     #[allow(clippy::should_panic_without_expect)]
    #[should_panic]
    fn prop_junk(d in arb_junk()) {
        let mut d1 = d.clone();
        let mut d2 = d.clone();
        let mut d3 = d;

        from_slice::<serde_json::Value>(&mut d1).expect("from_slice");
        to_borrowed_value(&mut d2).expect("to_borrowed_value");
        to_owned_value(&mut d3).expect("to_owned_value");

    }
}

#[cfg(not(target_arch = "wasm32"))]
proptest! {
    #![proptest_config(ProptestConfig {
        // Setting both fork and timeout is redundant since timeout implies
        // fork, but both are shown for clarity.
        // Disabled for code coverage, enable to track bugs
        // fork: true,
        .. ProptestConfig::default()
    })]

    #[test]
    #[allow(clippy::should_panic_without_expect)]
    #[should_panic]
    fn prop_string(d in "\\PC*") {
        let mut d1 = d.clone();
        let d1 = unsafe{ d1.as_bytes_mut()};
        let mut d2 = d.clone();
        let d2 = unsafe{ d2.as_bytes_mut()};
        let mut d3 = d;
        let d3 = unsafe{ d3.as_bytes_mut()};
        from_slice::<serde_json::Value>(d1).expect("from_slice");
        to_borrowed_value(d2).expect("to_borrowed_value");
        to_owned_value(d3).expect("to_owned_value");

    }
}
