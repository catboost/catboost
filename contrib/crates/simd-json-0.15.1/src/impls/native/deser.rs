use crate::{
    Deserializer, ErrorType, Result, SillyWrapper,
    safer_unchecked::GetSaferUnchecked,
    stringparse::{ESCAPE_MAP, get_unicode_codepoint},
};

#[allow(clippy::cast_possible_truncation)]
pub(crate) unsafe fn parse_str<'invoke, 'de>(
    input: SillyWrapper<'de>,
    data: &'invoke [u8],
    _buffer: &'invoke mut [u8],
    idx: usize,
) -> Result<&'de str> {
    use ErrorType::{InvalidEscape, InvalidUnicodeCodepoint};

    let input = input.input;
    // skip leading `"`
    let src: &[u8] = unsafe { data.get_kinda_unchecked(idx + 1..) };
    let input = unsafe { input.add(idx + 1) };

    let mut src_i = 0;
    let mut b = unsafe { *src.get_kinda_unchecked(src_i) };

    // quickly skip all the "good stuff"
    while b != b'"' && b != b'\\' {
        src_i += 1;
        b = unsafe { *src.get_kinda_unchecked(src_i) };
    }
    if b == b'"' {
        let v = unsafe { std::str::from_utf8_unchecked(std::slice::from_raw_parts(input, src_i)) };
        return Ok(v);
    }

    // we hit an escape sequence now it gets tricky
    // our destination index is idx +1 to skip the '"' + our current position
    let mut dst_i = src_i;
    while b != b'"' {
        if b == b'\\' {
            // don't advance i yet
            let escape_char = unsafe { *src.get_kinda_unchecked(src_i + 1) };
            if escape_char == b'u' {
                // got to reduce by 1 since we have to include the '\\' for get_unicode_codepoint
                let (cp, src_offset) =
                    unsafe { get_unicode_codepoint(src.get_kinda_unchecked(src_i..)) }.map_err(
                        |_| Deserializer::error_c(idx + 1 + src_i, 'u', InvalidUnicodeCodepoint),
                    )?;

                // from  codepoint_to_utf8 since we write directly to input
                unsafe {
                    if cp <= 0x7F {
                        input.add(dst_i).write(cp as u8);
                        dst_i += 1;
                    } else if cp <= 0x7FF {
                        input.add(dst_i).write(((cp >> 6) + 192) as u8);
                        dst_i += 1;
                        input.add(dst_i).write(((cp & 63) + 128) as u8);
                        dst_i += 1;
                    } else if cp <= 0xFFFF {
                        input.add(dst_i).write(((cp >> 12) + 224) as u8);
                        dst_i += 1;
                        input.add(dst_i).write((((cp >> 6) & 63) + 128) as u8);
                        dst_i += 1;
                        input.add(dst_i).write(((cp & 63) + 128) as u8);
                        dst_i += 1;
                    } else if cp <= 0x0010_FFFF {
                        input.add(dst_i).write(((cp >> 18) + 240) as u8);
                        dst_i += 1;
                        input.add(dst_i).write((((cp >> 12) & 63) + 128) as u8);
                        dst_i += 1;
                        input.add(dst_i).write((((cp >> 6) & 63) + 128) as u8);
                        dst_i += 1;
                        input.add(dst_i).write(((cp & 63) + 128) as u8);
                        dst_i += 1;
                    } else {
                        return Err(Deserializer::error_c(
                            idx + 1 + src_i,
                            'u',
                            InvalidUnicodeCodepoint,
                        ));
                    }
                }
                // We have to subtract one since we're already moving to the next character at the end of the loop
                src_i += src_offset - 1;
            } else {
                let escape_result: u8 =
                    unsafe { *ESCAPE_MAP.get_kinda_unchecked(escape_char as usize) };
                if escape_result == 0 {
                    return Err(Deserializer::error_c(
                        idx + 1 + src_i,
                        escape_char as char,
                        InvalidEscape,
                    ));
                }
                unsafe { input.add(dst_i).write(escape_result) };
                dst_i += 1;
                // move i for reading the escape char
                src_i += 1;
            }
        } else {
            unsafe { input.add(dst_i).write(b) };
            dst_i += 1;
        }
        src_i += 1;
        b = unsafe { *src.get_kinda_unchecked(src_i) };
    }
    unsafe {
        Ok(std::str::from_utf8_unchecked(std::slice::from_raw_parts(
            input, dst_i,
        )))
    }
}

#[cfg(test)]
mod test {
    use crate::SIMDJSON_PADDING;

    fn deser_str(input: &[u8]) -> Result<String> {
        let mut input = input.to_vec();
        let mut input2 = input.clone();
        input2.append(vec![0; SIMDJSON_PADDING * 2].as_mut());
        let mut buffer = vec![0; 1024];

        let r = unsafe {
            super::parse_str(input.as_mut_ptr().into(), &input2, buffer.as_mut_slice(), 0)?
        };
        Ok(String::from(r))
    }
    use super::*;
    #[test]
    fn easy_string() -> Result<()> {
        let s = deser_str(&br#""snot""#[..])?;
        assert_eq!("snot", s);
        Ok(())
    }

    #[test]
    fn string_with_quote() -> Result<()> {
        let s = deser_str(&br#""snot says:\n \"badger\"""#[..])?;
        assert_eq!("snot says:\n \"badger\"", s);
        Ok(())
    }

    #[test]
    fn string_with_utf8() -> Result<()> {
        let s = deser_str(&br#""\u000e""#[..])?;
        assert_eq!("\u{e}", s);
        Ok(())
    }
}
