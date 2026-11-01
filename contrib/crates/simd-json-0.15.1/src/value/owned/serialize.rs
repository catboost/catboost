// This is mostly taken from json-rust's codegen
// as it seems to perform well and it makes sense to see
// if we can adopt the approach
//
// https://github.com/maciejhirsz/json-rust/blob/master/src/codegen.rs

use super::{Object, Value};
use crate::prelude::*;
use std::io;
use std::io::Write;
use value_trait::generator::{
    DumpGenerator, PrettyGenerator, PrettyWriterGenerator, WriterGenerator,
};

//use util::print_dec;

impl Writable for Value {
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn encode(&self) -> String {
        let mut g = DumpGenerator::new();
        let _r = g.write_json(self);
        g.consume()
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn encode_pp(&self) -> String {
        let mut g = PrettyGenerator::new(2);
        let _r = g.write_json(self);
        g.consume()
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn write<'writer, W>(&self, w: &mut W) -> io::Result<()>
    where
        W: 'writer + Write,
    {
        let mut g = WriterGenerator::new(w);
        g.write_json(self)
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn write_pp<'writer, W>(&self, w: &mut W) -> io::Result<()>
    where
        W: 'writer + Write,
    {
        let mut g = PrettyWriterGenerator::new(w, 2);
        g.write_json(self)
    }
}

trait Generator: BaseGenerator {
    type T: Write;

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn write_object(&mut self, object: &Object) -> io::Result<()> {
        if object.is_empty() {
            self.write(b"{}")
        } else {
            let mut iter = object.iter();
            stry!(self.write(b"{"));

            // We know this exists since it's not empty
            let Some((key, value)) = iter.next() else {
                // We check against size
                unreachable!("object is not empty but has no next");
            };
            self.indent();
            stry!(self.new_line());
            stry!(self.write_simple_string(key));
            stry!(self.write_min(b": ", b':'));
            stry!(self.write_json(value));

            for (key, value) in iter {
                stry!(self.write(b","));
                stry!(self.new_line());
                stry!(self.write_simple_string(key));
                stry!(self.write_min(b": ", b':'));
                stry!(self.write_json(value));
            }
            self.dedent();
            stry!(self.new_line());
            self.write(b"}")
        }
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn write_json(&mut self, json: &Value) -> io::Result<()> {
        match *json {
            Value::Static(StaticNode::Null) => self.write(b"null"),
            Value::Static(StaticNode::I64(number)) => self.write_int(number),
            #[cfg(feature = "128bit")]
            Value::Static(StaticNode::I128(number)) => self.write_int(number),
            Value::Static(StaticNode::U64(number)) => self.write_int(number),
            #[cfg(feature = "128bit")]
            Value::Static(StaticNode::U128(number)) => self.write_int(number),
            #[allow(clippy::useless_conversion)] // .into() required by ordered-float
            Value::Static(StaticNode::F64(number)) => self.write_float(number.into()),
            Value::Static(StaticNode::Bool(true)) => self.write(b"true"),
            Value::Static(StaticNode::Bool(false)) => self.write(b"false"),
            Value::String(ref string) => self.write_string(string),
            Value::Array(ref array) => {
                if array.is_empty() {
                    self.write(b"[]")
                } else {
                    let mut iter = <[Value]>::iter(array);
                    // We know we have one item

                    let Some(item) = iter.next() else {
                        // We check against size
                        unreachable!("array is not empty but has no next");
                    };

                    stry!(self.write(b"["));

                    self.indent();
                    stry!(self.new_line());
                    stry!(self.write_json(item));

                    for item in iter {
                        stry!(self.write(b","));
                        stry!(self.new_line());
                        stry!(self.write_json(item));
                    }

                    self.dedent();
                    stry!(self.new_line());
                    self.write(b"]")
                }
            }
            Value::Object(ref object) => self.write_object(object),
        }
    }
}

trait FastGenerator: BaseGenerator {
    type T: Write;

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn write_object(&mut self, object: &Object) -> io::Result<()> {
        if object.is_empty() {
            self.write(b"{}")
        } else {
            let mut iter = object.iter();
            stry!(self.write(b"{\""));

            // We know this exists since it's not empty
            let Some((key, value)) = iter.next() else {
                // We check against size
                unreachable!("object is not empty but has no next");
            };
            stry!(self.write_simple_str_content(key));
            stry!(self.write(b"\":"));
            stry!(self.write_json(value));

            for (key, value) in iter {
                stry!(self.write(b",\""));
                stry!(self.write_simple_str_content(key));
                stry!(self.write(b"\":"));
                stry!(self.write_json(value));
            }
            self.write(b"}")
        }
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn write_json(&mut self, json: &Value) -> io::Result<()> {
        match *json {
            Value::Static(StaticNode::Null) => self.write(b"null"),
            Value::Static(StaticNode::I64(number)) => self.write_int(number),
            #[cfg(feature = "128bit")]
            Value::Static(StaticNode::I128(number)) => self.write_int(number),
            Value::Static(StaticNode::U64(number)) => self.write_int(number),
            #[cfg(feature = "128bit")]
            Value::Static(StaticNode::U128(number)) => self.write_int(number),
            #[allow(clippy::useless_conversion)] // .into() required by ordered-float
            Value::Static(StaticNode::F64(number)) => self.write_float(number.into()),
            Value::Static(StaticNode::Bool(true)) => self.write(b"true"),
            Value::Static(StaticNode::Bool(false)) => self.write(b"false"),
            Value::String(ref string) => self.write_string(string),
            Value::Array(ref array) => {
                if array.is_empty() {
                    self.write(b"[]")
                } else {
                    let mut iter = <[Value]>::iter(array);
                    // We know we have one item
                    let Some(item) = iter.next() else {
                        // We check against size
                        unreachable!("array is not empty but has no next");
                    };

                    stry!(self.write(b"["));
                    stry!(self.write_json(item));

                    for item in iter {
                        stry!(self.write(b","));
                        stry!(self.write_json(item));
                    }
                    self.write(b"]")
                }
            }
            Value::Object(ref object) => self.write_object(object),
        }
    }
}

impl FastGenerator for DumpGenerator {
    type T = Vec<u8>;
}

impl Generator for PrettyGenerator {
    type T = Vec<u8>;
}

impl<W> FastGenerator for WriterGenerator<'_, W>
where
    W: Write,
{
    type T = W;
}

impl<W> Generator for PrettyWriterGenerator<'_, W>
where
    W: Write,
{
    type T = W;
}

#[cfg(test)]
mod test {
    use super::Value;
    use crate::prelude::*;

    #[test]
    fn null() {
        assert_eq!(Value::Static(StaticNode::Null).encode(), "null");
    }
    #[test]
    fn bool_true() {
        assert_eq!(Value::Static(StaticNode::Bool(true)).encode(), "true");
    }
    #[test]
    fn bool_false() {
        assert_eq!(Value::Static(StaticNode::Bool(false)).encode(), "false");
    }

    #[test]
    fn obj() {
        let mut o = Value::object();
        o.insert("k", ()).expect("insert");
        assert_eq!(o.encode(), r#"{"k":null}"#);
    }
    fn assert_str(from: &str, to: &str) {
        assert_eq!(Value::String(from.into()).encode(), to);
    }

    #[test]
    fn string() {
        assert_str("this is a test", r#""this is a test""#);
        assert_str(r#"this is a test ""#, r#""this is a test \"""#);
        assert_str(r#"this is a test """#, r#""this is a test \"\"""#);
        assert_str(
            "this is a test a long test that should span the 32 byte boundary",
            r#""this is a test a long test that should span the 32 byte boundary""#,
        );
        assert_str(
            r#"this is a test a "long" test that should span the 32 byte boundary"#,
            r#""this is a test a \"long\" test that should span the 32 byte boundary""#,
        );

        assert_str(
            r#"this is a test a \"long\" test that should span the 32 byte boundary"#,
            r#""this is a test a \\\"long\\\" test that should span the 32 byte boundary""#,
        );
    }
}
