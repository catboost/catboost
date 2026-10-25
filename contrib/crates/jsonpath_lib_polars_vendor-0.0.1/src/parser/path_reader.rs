use std::result::Result;

#[derive(Debug, PartialEq)]
pub enum ReaderError {
    Eof,
}

pub struct PathReader<'a> {
    input: &'a str,
    pos: usize,
}

impl<'a> PathReader<'a> {
    pub fn new(input: &'a str) -> Self {
        PathReader { input, pos: 0 }
    }

    pub fn peek_char(&self) -> Result<(usize, char), ReaderError> {
        let ch = self.input.chars().next().ok_or(ReaderError::Eof)?;
        Ok((self.pos + ch.len_utf8(), ch))
    }

    pub fn take_while<F>(&mut self, fun: F) -> Result<(usize, String), ReaderError>
    where
        F: Fn(&char) -> bool,
    {
        let mut char_len: usize = 0;
        let mut ret = String::new();
        for c in self.input.chars().by_ref() {
            if !fun(&c) {
                break;
            }
            char_len += c.len_utf8();
            ret.push(c);
        }

        self.pos += char_len;
        self.input = &self.input[char_len..];
        Ok((self.pos, ret))
    }

    pub fn next_char(&mut self) -> Result<(usize, char), ReaderError> {
        let (_, ch) = self.peek_char()?;
        self.input = &self.input[ch.len_utf8()..];
        let ret = Ok((self.pos, ch));
        self.pos += ch.len_utf8();
        ret
    }

    pub fn current_pos(&self) -> usize {
        self.pos
    }
}
