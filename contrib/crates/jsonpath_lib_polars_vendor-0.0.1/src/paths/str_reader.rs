use std::result::Result;
use std::str::Chars;

#[derive(Debug, PartialEq)]
pub enum ReaderError {
    Eof,
}

#[derive(Debug, PartialEq, Clone)]
pub struct StrRange {
    pub pos: usize,
    pub offset: usize,
}

impl StrRange {
    pub fn new(pos: usize, offset: usize) -> Self {
        StrRange { pos, offset }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct StrReader<'a> {
    input: &'a str,
    pos: usize,
    chars: Chars<'a>,
    peeked: Option<Option<char>>,
}

impl<'a> StrReader<'a> {
    pub fn new(input: &'a str) -> Self {
        StrReader { input, pos: 0, chars: input.chars(), peeked: None }
    }

    pub fn peek_char(&mut self) -> Result<char, ReaderError> {
        let ch = self.peek().ok_or(ReaderError::Eof)?;
        Ok(*ch)
    }

    pub fn take_while<F>(&mut self, fun: F) -> Result<StrRange, ReaderError>
        where
            F: Fn(&char) -> bool,
    {
        let mut char_len: usize = 0;
        while let Some(c) = self.peek() {
            if !fun(c) {
                break;
            }
            match self.next() {
                Some(ch) => char_len += ch.len_utf8(),
                _ => return Err(ReaderError::Eof)
            }
        }

        let pos = self.pos;
        self.pos += char_len;
        Ok(StrRange::new(pos, char_len))
    }

    pub fn next_char(&mut self) -> Result<(StrRange, char), ReaderError> {
        let ch = self.next().ok_or(ReaderError::Eof)?;
        let pos = self.pos;
        let len = ch.len_utf8();
        self.pos += len;
        Ok((StrRange::new(pos, len), ch))
    }

    pub fn read(&self, span: &StrRange) -> &'a str {
        &self.input[span.pos..(span.pos + span.offset)]
    }

    pub fn current_pos(&self) -> usize {
        self.pos
    }

    pub fn origin_str(&self) -> &'a str {
        self.input
    }

    fn next(&mut self) -> Option<char> {
        match self.peeked.take() {
            Some(v) => v,
            None => self.chars.next(),
        }
    }

    fn peek(&mut self) -> Option<&char> {
        let chars = &mut self.chars;
        self.peeked.get_or_insert_with(|| chars.next()).as_ref()
    }
}
