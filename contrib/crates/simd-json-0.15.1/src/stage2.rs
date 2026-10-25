#![allow(dead_code)]
use crate::charutils::is_not_structural_or_whitespace;
use crate::macros::unlikely;
use crate::safer_unchecked::GetSaferUnchecked;
use crate::value::tape::Node;
use crate::{Deserializer, Error, ErrorType, InternalError, Result};
use value_trait::StaticNode;

#[cfg_attr(not(feature = "no-inline"), inline)]
pub fn is_valid_true_atom(loc: &[u8]) -> bool {
    debug_assert!(loc.len() >= 8, "loc too short for a u64 read");

    // TODO is this expensive?
    let mut error: u64;
    unsafe {
        //let tv: u64 = *(b"true    ".as_ptr() as *const u64);
        // this is the same:
        const TV: u64 = 0x00_00_00_00_65_75_72_74;
        const MASK4: u64 = 0x00_00_00_00_ff_ff_ff_ff;

        let locval: u64 = loc.as_ptr().cast::<u64>().read_unaligned();

        error = (locval & MASK4) ^ TV;
        error |= u64::from(is_not_structural_or_whitespace(*loc.get_kinda_unchecked(4)));
    }
    error == 0
}

macro_rules! get {
    ($a:expr_2021, $i:expr_2021) => {{ unsafe { $a.get_kinda_unchecked($i) } }};
}

#[cfg_attr(not(feature = "no-inline"), inline)]
pub fn is_valid_false_atom(loc: &[u8]) -> bool {
    const FV: u64 = 0x00_00_00_65_73_6c_61_66;
    const MASK5: u64 = 0x00_00_00_ff_ff_ff_ff_ff;

    debug_assert!(loc.len() >= 8, "loc too short for a u64 read");

    // TODO: this is ugly and probably copies data every time
    let mut error: u64;
    //let fv: u64 = *(b"false   ".as_ptr() as *const u64);
    // this is the same:

    let locval: u64 = unsafe { loc.as_ptr().cast::<u64>().read_unaligned() };

    // FIXME the original code looks like this:
    // error = ((locval & mask5) ^ fv) as u32;
    // but that fails on falsy as the u32 conversion
    // will mask the error on the y so we re-write it
    // it would be interesting what the consequences are
    error = (locval & MASK5) ^ FV;
    error |= u64::from(is_not_structural_or_whitespace(*get!(loc, 5)));

    error == 0
}

#[cfg_attr(not(feature = "no-inline"), inline)]
pub fn is_valid_null_atom(loc: &[u8]) -> bool {
    //let nv: u64 = *(b"null   ".as_ptr() as *const u64);
    // this is the same:
    const NV: u64 = 0x00_00_00_00_6c_6c_75_6e;
    const MASK4: u64 = 0x00_00_00_00_ff_ff_ff_ff;

    debug_assert!(loc.len() >= 8, "loc too short for a u64 read");

    // TODO is this expensive?
    let mut error: u64;

    let locval: u64 = unsafe { loc.as_ptr().cast::<u64>().read_unaligned() };

    error = (locval & MASK4) ^ NV;
    error |= u64::from(is_not_structural_or_whitespace(*get!(loc, 4)));

    error == 0
}

#[derive(Debug)]
enum State {
    ObjectKey,
    ScopeEnd,
    MainArraySwitch,
}
#[derive(Debug)]
pub(crate) enum StackState {
    Start,
    Object { last_start: usize, cnt: usize },
    Array { last_start: usize, cnt: usize },
}

impl<'de> Deserializer<'de> {
    #[cfg_attr(not(feature = "no-inline"), inline)]
    #[allow(
        clippy::cognitive_complexity,
        clippy::too_many_lines,
        unused_unsafe,
        clippy::needless_continue
    )]
    pub(crate) fn build_tape(
        input: &'de mut [u8],
        input2: &[u8],
        buffer: &mut [u8],
        structural_indexes: &[u32],
        stack: &mut Vec<StackState>,
        res: &mut Vec<Node<'de>>,
    ) -> Result<()> {
        res.clear();
        res.reserve(structural_indexes.len());
        // While a valid json can have at max len/2 (`[[[]]]`)elements that are relevant
        // a invalid json might exceed this `[[[[[[` and we need to protect against that.
        stack.clear();
        stack.reserve(structural_indexes.len());

        let res_ptr = res.as_mut_ptr();
        let stack_ptr = stack.as_mut_ptr();

        let mut depth: usize = 0;
        let mut last_start;
        let mut cnt: usize;
        let mut r_i = 0;

        // let mut i: usize = 0; // index of the structural character (0,1,2,3...)
        // location of the structural character in the input (buf)
        let mut idx: usize = 0;
        // used to track the (structural) character we are looking at, updated
        // by UPDATE_CHAR macro
        let mut c: u8 = 0;
        // skip the zero index
        let mut i: usize = 0;
        let mut state;

        macro_rules! s2try {
            ($e:expr_2021) => {
                match $e {
                    ::std::result::Result::Ok(val) => val,
                    ::std::result::Result::Err(err) => {
                        // We need to ensure that rust doesn't
                        // try to free strings that we never
                        // allocated
                        unsafe {
                            res.set_len(r_i);
                        };
                        return ::std::result::Result::Err(err);
                    }
                }
            };
        }

        macro_rules! insert_res {
            ($t:expr_2021) => {
                unsafe {
                    res_ptr.add(r_i).write($t);
                    r_i += 1;
                }
            };
        }
        macro_rules! success {
            () => {
                unsafe {
                    res.set_len(r_i);
                }
                return Ok(());
            };
        }
        macro_rules! update_char {
            () => {
                if i < structural_indexes.len() {
                    idx = *get!(structural_indexes, i) as usize;
                    i += 1;
                    c = *get!(input2, idx);
                } else {
                    fail!(ErrorType::Syntax);
                }
            };
        }

        macro_rules! goto {
            ($state:expr_2021) => {{
                state = $state;
                #[allow(clippy::needless_continue)]
                continue;
            }};
        }

        macro_rules! insert_str {
            () => {
                insert_res!(Node::String(s2try!(Self::parse_str_(
                    input.as_mut_ptr(),
                    &input2,
                    buffer,
                    idx
                ))));
            };
        }

        // The continue cases are the most frequently called onces it's
        // worth pulling them out into a macro (aka inlining them)
        // Since we don't have a 'gogo' in rust.
        macro_rules! array_continue {
            () => {{
                update_char!();
                match c {
                    b',' => {
                        cnt += 1;
                        update_char!();
                        goto!(MainArraySwitch);
                    }
                    b']' => {
                        goto!(ScopeEnd);
                    }
                    _c => {
                        fail!(ErrorType::ExpectedArrayContent);
                    }
                }
            }};
        }

        macro_rules! object_continue {
            () => {{
                update_char!();
                match c {
                    b',' => {
                        cnt += 1;
                        update_char!();
                        if c == b'"' {
                            insert_str!();
                            goto!(ObjectKey);
                        }
                        fail!(ErrorType::ExpectedObjectKey);
                    }
                    b'}' => {
                        goto!(ScopeEnd);
                    }
                    _ => {
                        fail!(ErrorType::ExpectedObjectContent);
                    }
                }
            }};
        }

        macro_rules! array_begin {
            () => {
                update_char!();
                if c == b']' {
                    cnt = 0;
                    goto!(ScopeEnd);
                }
                goto!(MainArraySwitch);
            };
        }

        macro_rules! object_begin {
            () => {{
                update_char!();
                match c {
                    b'"' => {
                        insert_str!();
                        goto!(ObjectKey)
                    }
                    b'}' => {
                        cnt = 0;
                        goto!(ScopeEnd);
                    }
                    _c => {
                        fail!(ErrorType::ExpectedObjectContent);
                    }
                }
            }};
        }

        macro_rules! fail {
            () => {
                // We need to ensure that rust doesn't
                // try to free strings that we never
                // allocated
                unsafe {
                    res.set_len(r_i);
                };
                return Err(Error::new_c(
                    idx,
                    c as char,
                    ErrorType::InternalError(InternalError::TapeError),
                ));
            };
            ($t:expr_2021) => {
                // We need to ensure that rust doesn't
                // try to free strings that we never
                // allocated
                unsafe {
                    res.set_len(r_i);
                };
                return Err(Error::new_c(idx, c as char, $t));
            };
        }
        // State start, we pull this outside of the
        // loop to reduce the number of required checks
        update_char!();
        match c {
            b'{' => {
                unsafe {
                    stack_ptr.add(depth).write(StackState::Start);
                }

                last_start = r_i;
                insert_res!(Node::Object { len: 0, count: 0 });

                depth += 1;
                cnt = 1;

                update_char!();
                match c {
                    b'"' => {
                        insert_str!();
                        state = State::ObjectKey;
                    }
                    b'}' => {
                        cnt = 0;
                        state = State::ScopeEnd;
                    }
                    _c => {
                        fail!(ErrorType::ExpectedObjectContent);
                    }
                }
            }
            b'[' => {
                unsafe {
                    stack_ptr.add(depth).write(StackState::Start);
                }

                last_start = r_i;
                insert_res!(Node::Array { len: 0, count: 0 });

                depth += 1;
                cnt = 1;

                update_char!();
                if c == b']' {
                    cnt = 0;
                    state = State::ScopeEnd;
                } else {
                    state = State::MainArraySwitch;
                }
            }
            b't' => {
                unsafe {
                    if !is_valid_true_atom(get!(input2, idx..)) {
                        fail!(ErrorType::ExpectedTrue);
                    }
                };
                insert_res!(Node::Static(StaticNode::Bool(true)));
                if i == structural_indexes.len() {
                    success!();
                }
                fail!(ErrorType::TrailingData);
            }
            b'f' => {
                unsafe {
                    if !is_valid_false_atom(get!(input2, idx..)) {
                        fail!(ErrorType::ExpectedFalse);
                    }
                };
                insert_res!(Node::Static(StaticNode::Bool(false)));
                if i == structural_indexes.len() {
                    success!();
                }
                fail!(ErrorType::TrailingData);
            }
            b'n' => {
                unsafe {
                    if !is_valid_null_atom(get!(input2, idx..)) {
                        fail!(ErrorType::ExpectedNull);
                    }
                };
                insert_res!(Node::Static(StaticNode::Null));
                if i == structural_indexes.len() {
                    success!();
                }
                fail!(ErrorType::TrailingData);
            }
            b'"' => {
                insert_str!();
                if i == structural_indexes.len() {
                    success!();
                }
                fail!(ErrorType::TrailingData);
            }
            b'-' => {
                insert_res!(Node::Static(s2try!(Self::parse_number(idx, input2, true))));

                if i == structural_indexes.len() {
                    success!();
                }
                fail!(ErrorType::TrailingData);
            }
            b'0'..=b'9' => {
                insert_res!(Node::Static(s2try!(Self::parse_number(idx, input2, false))));

                if i == structural_indexes.len() {
                    success!();
                }
                fail!(ErrorType::TrailingData);
            }
            _ => {
                fail!();
            }
        }

        loop {
            use self::State::{MainArraySwitch, ObjectKey, ScopeEnd};
            match state {
                ////////////////////////////// OBJECT STATES /////////////////////////////
                ObjectKey => {
                    update_char!();
                    if unlikely!(c != b':') {
                        fail!(ErrorType::ExpectedObjectColon);
                    }
                    update_char!();
                    match c {
                        b'"' => {
                            insert_str!();
                            object_continue!();
                        }
                        b't' => {
                            insert_res!(Node::Static(StaticNode::Bool(true)));
                            if !is_valid_true_atom(get!(input2, idx..)) {
                                fail!(ErrorType::ExpectedTrue);
                            }
                            object_continue!();
                        }
                        b'f' => {
                            insert_res!(Node::Static(StaticNode::Bool(false)));
                            if !is_valid_false_atom(get!(input2, idx..)) {
                                fail!(ErrorType::ExpectedFalse);
                            }
                            object_continue!();
                        }
                        b'n' => {
                            insert_res!(Node::Static(StaticNode::Null));
                            if !is_valid_null_atom(get!(input2, idx..)) {
                                fail!(ErrorType::ExpectedNull);
                            }
                            object_continue!();
                        }
                        b'-' => {
                            insert_res!(Node::Static(s2try!(Self::parse_number(
                                idx, input2, true
                            ))));

                            object_continue!();
                        }
                        b'0'..=b'9' => {
                            insert_res!(Node::Static(s2try!(Self::parse_number(
                                idx, input2, false
                            ))));

                            object_continue!();
                        }
                        b'{' => {
                            unsafe {
                                stack_ptr
                                    .add(depth)
                                    .write(StackState::Object { last_start, cnt });
                            }
                            last_start = r_i;
                            insert_res!(Node::Object { len: 0, count: 0 });
                            depth += 1;
                            cnt = 1;
                            object_begin!();
                        }
                        b'[' => {
                            unsafe {
                                stack_ptr
                                    .add(depth)
                                    .write(StackState::Object { last_start, cnt });
                            }
                            last_start = r_i;
                            insert_res!(Node::Array { len: 0, count: 0 });
                            depth += 1;
                            cnt = 1;
                            array_begin!();
                        }
                        _c => {
                            fail!();
                        }
                    }
                }
                ////////////////////////////// COMMON STATE /////////////////////////////
                ScopeEnd => {
                    if depth == 0 {
                        fail!(ErrorType::Syntax);
                    }
                    depth -= 1;
                    unsafe {
                        match *res_ptr.add(last_start) {
                            Node::Array {
                                ref mut len,
                                count: ref mut end,
                            }
                            | Node::Object {
                                ref mut len,
                                count: ref mut end,
                            } => {
                                *len = cnt;
                                *end = r_i - last_start - 1;
                            }
                            _ => unreachable!("scop end needs to be an array or object"),
                        }
                    }
                    unsafe {
                        match *stack_ptr.add(depth) {
                            StackState::Object {
                                last_start: l,
                                cnt: c,
                            } => {
                                last_start = l;
                                cnt = c;
                                object_continue!();
                            }
                            StackState::Array {
                                last_start: l,
                                cnt: c,
                            } => {
                                last_start = l;
                                cnt = c;
                                array_continue!();
                            }
                            StackState::Start => {
                                if i == structural_indexes.len() {
                                    success!();
                                }
                                fail!();
                            }
                        };
                    }
                }

                ////////////////////////////// ARRAY STATES /////////////////////////////
                MainArraySwitch => {
                    // we call update char on all paths in, so we can peek at c on the
                    // on paths that can accept a close square brace (post-, and at start)
                    match c {
                        b'"' => {
                            insert_str!();
                            array_continue!();
                        }
                        b't' => {
                            insert_res!(Node::Static(StaticNode::Bool(true)));
                            if !is_valid_true_atom(get!(input2, idx..)) {
                                fail!(ErrorType::ExpectedTrue);
                            }
                            array_continue!();
                        }
                        b'f' => {
                            insert_res!(Node::Static(StaticNode::Bool(false)));
                            if !is_valid_false_atom(get!(input2, idx..)) {
                                fail!(ErrorType::ExpectedFalse);
                            }
                            array_continue!();
                        }
                        b'n' => {
                            insert_res!(Node::Static(StaticNode::Null));
                            if !is_valid_null_atom(get!(input2, idx..)) {
                                fail!(ErrorType::ExpectedNull);
                            }
                            array_continue!();
                        }
                        b'-' => {
                            insert_res!(Node::Static(s2try!(Self::parse_number(
                                idx, input2, true
                            ))));

                            array_continue!();
                        }
                        b'0'..=b'9' => {
                            insert_res!(Node::Static(s2try!(Self::parse_number(
                                idx, input2, false
                            ))));

                            array_continue!();
                        }
                        b'{' => {
                            unsafe {
                                stack_ptr
                                    .add(depth)
                                    .write(StackState::Array { last_start, cnt });
                            }
                            last_start = r_i;
                            insert_res!(Node::Object { len: 0, count: 0 });
                            depth += 1;
                            cnt = 1;
                            object_begin!();
                        }
                        b'[' => {
                            unsafe {
                                stack_ptr
                                    .add(depth)
                                    .write(StackState::Array { last_start, cnt });
                            }
                            last_start = r_i;
                            insert_res!(Node::Array { len: 0, count: 0 });
                            depth += 1;
                            cnt = 1;
                            array_begin!();
                        }
                        _c => {
                            fail!();
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod test {
    use crate::SIMDJSON_PADDING;

    use super::*;

    #[test]
    fn true_atom() {
        assert!(is_valid_true_atom(b"true    "));
        assert!(!is_valid_true_atom(b"tru     "));
        assert!(!is_valid_true_atom(b" rue    "));
    }
    #[test]
    fn false_atom() {
        assert!(is_valid_false_atom(b"false   "));
        assert!(!is_valid_false_atom(b"falze   "));
        assert!(!is_valid_false_atom(b"falsy   "));
        assert!(!is_valid_false_atom(b"fals    "));
        assert!(!is_valid_false_atom(b" alse   "));

        //unsafe { assert!(!is_valid_false_atom(b"fals    " as *const u8)) }
        //        unsafe { assert!(!is_valid_false_atom(b"false   " as *const u8)) }
    }
    #[test]
    fn null_atom() {
        assert!(is_valid_null_atom(b"null    "));
        assert!(!is_valid_null_atom(b"nul     "));
        assert!(!is_valid_null_atom(b" ull    "));
    }

    #[cfg(feature = "serde_impl")]
    #[test]
    fn parsing_errors() {
        assert_eq!(
            crate::serde::from_slice::<bool>(&mut b"time".to_vec()),
            Err(Error::new_c(0, 't', ErrorType::ExpectedTrue))
        );
        assert_eq!(
            crate::serde::from_slice::<bool>(&mut b"falsy".to_vec()),
            Err(Error::new_c(0, 'f', ErrorType::ExpectedFalse))
        );
        assert_eq!(
            crate::serde::from_slice::<bool>(&mut b"new".to_vec()),
            Err(Error::new_c(0, 'n', ErrorType::ExpectedNull))
        );
        assert_eq!(
            crate::serde::from_slice::<bool>(&mut b"[true, time]".to_vec()),
            Err(Error::new_c(7, 't', ErrorType::ExpectedTrue))
        );
        assert_eq!(
            crate::serde::from_slice::<bool>(&mut b"[true, falsy]".to_vec()),
            Err(Error::new_c(7, 'f', ErrorType::ExpectedFalse))
        );
        assert_eq!(
            crate::serde::from_slice::<bool>(&mut b"[null, new]".to_vec()),
            Err(Error::new_c(7, 'n', ErrorType::ExpectedNull))
        );
        assert_eq!(
            crate::serde::from_slice::<bool>(&mut br#"{"1":time}"#.to_vec()),
            Err(Error::new_c(5, 't', ErrorType::ExpectedTrue))
        );
        assert_eq!(
            crate::serde::from_slice::<bool>(&mut br#"{"0":falsy}"#.to_vec()),
            Err(Error::new_c(5, 'f', ErrorType::ExpectedFalse))
        );
        assert_eq!(
            crate::serde::from_slice::<bool>(&mut br#"{"0":new}"#.to_vec()),
            Err(Error::new_c(5, 'n', ErrorType::ExpectedNull))
        );
    }

    #[test]
    fn parse_string() -> Result<()> {
        let mut input = Vec::from(&br#""{\"arg\":\"test\"}""#[..]);
        let mut input2 = input.clone();
        input2.append(vec![0; SIMDJSON_PADDING * 2].as_mut());
        let mut buffer = vec![0; 1024];

        let s = unsafe {
            Deserializer::parse_str_(input.as_mut_ptr(), &input2, buffer.as_mut_slice(), 0)?
        };
        assert_eq!(r#"{"arg":"test"}"#, s);
        Ok(())
    }
}
