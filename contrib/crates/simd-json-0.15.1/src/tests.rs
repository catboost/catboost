#![allow(clippy::ignored_unit_patterns)]

#[cfg(feature = "serde_impl")]
mod serde;

mod impls;

#[cfg(not(target_arch = "wasm32"))]
use crate::to_borrowed_value;
use crate::{Deserializer, owned::Value, tape::Node, to_owned_value};
#[cfg(not(target_arch = "wasm32"))]
use proptest::prelude::*;
use value_trait::prelude::*;

#[cfg(not(feature = "approx-number-parsing"))]
#[test]
#[allow(clippy::float_cmp)]
fn alligned_number_parse() {
    let str = "9521.824380305317";
    let mut slice: Vec<u8> = str.as_bytes().to_owned();
    let value: crate::BorrowedValue<'_> =
        crate::to_borrowed_value(&mut slice).expect("failed to parse");
    assert_eq!(value, 9_521.824_380_305_317);
}

#[test]
fn test_send_sync() {
    struct TestStruct<T: Sync + Send>(T);
    #[allow(let_underscore_drop)] // test
    let _: TestStruct<_> = TestStruct(super::AlignedBuf::with_capacity(0));
}

#[test]
fn count1() {
    let mut d = String::from("[]");
    let d = unsafe { d.as_bytes_mut() };
    let simd = Deserializer::from_slice(d).expect("");
    assert_eq!(simd.tape[0], Node::Array { len: 0, count: 0 });
}

#[test]
fn count2() {
    let mut d = String::from("[1]");
    let d = unsafe { d.as_bytes_mut() };
    let simd = Deserializer::from_slice(d).expect("");
    assert_eq!(simd.tape[0], Node::Array { len: 1, count: 1 });
}

#[test]
fn count3() {
    let mut d = String::from("[1,2]");
    let d = unsafe { d.as_bytes_mut() };
    let simd = Deserializer::from_slice(d).expect("");
    assert_eq!(simd.tape[0], Node::Array { len: 2, count: 2 });
}

#[test]
fn count4() {
    let mut d = String::from(" [ 1 , [ 3 ] , 2 ]");
    let d = unsafe { d.as_bytes_mut() };
    let simd = Deserializer::from_slice(d).expect("");
    assert_eq!(simd.tape[0], Node::Array { len: 3, count: 4 });
    assert_eq!(simd.tape[2], Node::Array { len: 1, count: 1 });
}

#[test]
fn count5() {
    let mut d = String::from("[[],null,null]");
    let d = unsafe { d.as_bytes_mut() };
    let simd = Deserializer::from_slice(d).expect("");
    assert_eq!(simd.tape[0], Node::Array { len: 3, count: 3 });
    assert_eq!(simd.tape[1], Node::Array { len: 0, count: 0 });
}

#[test]
fn test_tape_object_simple() {
    let mut d = String::from(r#" { "hello": 1 , "b": 1 }"#);
    let d = unsafe { d.as_bytes_mut() };
    let simd = Deserializer::from_slice(d).expect("");
    assert_eq!(
        simd.tape,
        [
            Node::Object { len: 2, count: 4 },
            Node::String("hello"), // <-- This is already escaped
            Node::Static(StaticNode::I64(1)),
            Node::String("b"),
            Node::Static(StaticNode::I64(1)),
        ]
    );
}

#[test]
fn test_tape_object_escaped() {
    let mut d = String::from(r#" { "hell\"o": 1 , "b": [ 1, 2, 3 ] }"#);
    let d = unsafe { d.as_bytes_mut() };
    let simd = Deserializer::from_slice(d).expect("");
    assert_eq!(
        simd.tape,
        [
            Node::Object { len: 2, count: 7 },
            Node::String(r#"hell"o"#), // <-- This is already escaped
            Node::Static(StaticNode::I64(1)),
            Node::String("b"),
            Node::Array { len: 3, count: 3 },
            Node::Static(StaticNode::I64(1)),
            Node::Static(StaticNode::I64(2)),
            Node::Static(StaticNode::I64(3))
        ]
    );
}

#[test]
fn string_array() {
    const STR: &str = r#""{\"arg\":\"test\"}""#;
    let mut d = String::from(STR);
    let d = unsafe { d.as_bytes_mut() };
    let simd = Deserializer::from_slice(d).expect("");
    assert_eq!(simd.tape[0], Node::String("{\"arg\":\"test\"}"));
}

#[cfg(feature = "128bit")]
#[test]
fn odd_nuber() {
    use super::value::owned::to_value;
    use value_trait::prelude::*;

    let mut d =
        String::from(r#"{"name": "max_unsafe_auto_id_timestamp", "value": -9223372036854776000}"#);

    let mut d = unsafe { d.as_bytes_mut() };
    let mut o = Value::object();
    o.insert("name", "max_unsafe_auto_id_timestamp")
        .expect("failed to set key");
    o.insert("value", -9_223_372_036_854_776_000_i128)
        .expect("failed to set key");
    assert_eq!(to_value(&mut d), Ok(o));
}

#[cfg(feature = "128bit")]
#[test]
fn odd_nuber2() {
    use super::value::owned::to_value;
    use value_trait::prelude::*;

    let mut d =
        String::from(r#"{"name": "max_unsafe_auto_id_timestamp", "value": 9223372036854776000}"#);

    let mut d = unsafe { d.as_bytes_mut() };
    let mut o = Value::object();
    o.insert("name", "max_unsafe_auto_id_timestamp")
        .expect("failed to set key");
    o.insert("value", 9_223_372_036_854_776_000_u128)
        .expect("failed to set key");
    assert_eq!(to_value(&mut d), Ok(o));
}
// How much do we care about this, it's within the same range and
// based on floating point math imprecisions during parsing.
// Is this a real issue worth improving?
#[test]
fn silly_float1() {
    let v = Value::from(3.090_144_804_232_201_7e305);
    let s = v.encode();
    let mut bytes = s.as_bytes().to_vec();
    let parsed = to_owned_value(&mut bytes).expect("failed to parse generated float");
    assert_eq!(v, parsed);
}

#[test]
#[ignore]
fn silly_float2() {
    let v = Value::from(-6.990_585_694_841_803e305);
    let s = v.encode();
    let mut bytes = s.as_bytes().to_vec();
    let parsed = to_owned_value(&mut bytes).expect("failed to parse generated float");
    assert_eq!(v, parsed);
}
#[cfg(not(feature = "128bit"))]
#[cfg(not(target_arch = "wasm32"))]
fn arb_json_value() -> BoxedStrategy<Value> {
    let leaf = prop_oneof![
        Just(Value::Static(StaticNode::Null)),
        any::<bool>().prop_map(Value::from),
        //(-1.0e306f64..1.0e306f64).prop_map(Value::from), // damn you float!
        any::<i64>().prop_map(Value::from),
        any::<u64>().prop_map(Value::from),
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
    .boxed()
}

#[cfg(feature = "128bit")]
#[cfg(not(target_arch = "wasm32"))]
fn arb_json_value() -> BoxedStrategy<Value> {
    let leaf = prop_oneof![
        Just(Value::Static(StaticNode::Null)),
        any::<bool>().prop_map(Value::from),
        //(-1.0e306f64..1.0e306f64).prop_map(Value::from), // damn you float!
        any::<i64>().prop_map(Value::from),
        any::<u64>().prop_map(Value::from),
        any::<i128>().prop_map(Value::from),
        any::<u128>().prop_map(Value::from),
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
    .boxed()
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
    fn prop_json_encode_decode(val in arb_json_value()) {
        let mut encoded: Vec<u8> = Vec::new();
        val.write(&mut encoded).expect("write");
        println!("{}", String::from_utf8_lossy(&encoded));
        let mut e = encoded.clone();
        let res = to_owned_value(&mut e).expect("can't convert");
        assert_eq!(val, res);
        let mut e = encoded.clone();
        let res = to_borrowed_value(&mut e).expect("can't convert");
        assert_eq!(val, res);
        #[cfg(not(feature = "128bit"))]
        { // we can't do 128 bit w/ serde
            use crate::{deserialize, BorrowedValue, OwnedValue};
            let mut e = encoded.clone();
            let res: OwnedValue = deserialize(&mut e).expect("can't convert");
            assert_eq!(val, res);
            let mut e = encoded;
            let res: BorrowedValue = deserialize(&mut e).expect("can't convert");
            assert_eq!(val, res);
        }
    }

}
