use std::result::Result;

use super::str_reader::{ReaderError, StrRange, StrReader};
use super::tokens::Token;

const CH_DOLLA: char = '$';
const CH_DOT: char = '.';
const CH_ASTERISK: char = '*';
const CH_LARRAY: char = '[';
const CH_RARRAY: char = ']';
const CH_LPAREN: char = '(';
const CH_RPAREN: char = ')';
const CH_AT: char = '@';
const CH_QUESTION: char = '?';
const CH_COMMA: char = ',';
const CH_SEMICOLON: char = ':';
const CH_EQUAL: char = '=';
const CH_AMPERSAND: char = '&';
const CH_PIPE: char = '|';
const CH_LITTLE: char = '<';
const CH_GREATER: char = '>';
const CH_EXCLAMATION: char = '!';
const CH_SINGLE_QUOTE: char = '\'';
const CH_DOUBLE_QUOTE: char = '"';

#[derive(Debug, Clone, PartialEq)]
pub enum TokenError {
    Eof,
    Position(usize),
}

fn to_token_error(read_err: ReaderError) -> TokenError {
    match read_err {
        ReaderError::Eof => TokenError::Eof,
    }
}

#[derive(Clone, Debug)]
pub(super) struct Tokenizer<'a> {
    input: StrReader<'a>,
}

impl<'a> Tokenizer<'a> {
    pub fn new(input: &'a str) -> Self {
        trace!("input: {}", input);
        Tokenizer {
            input: StrReader::new(input),
        }
    }

    fn dolla(&mut self) -> Result<Token, TokenError> {
        let fun = |c: &char| match c {
            &CH_DOT
            | &CH_ASTERISK
            | &CH_LARRAY
            | &CH_RARRAY
            | &CH_LPAREN
            | &CH_RPAREN
            | &CH_AT
            | &CH_QUESTION
            | &CH_COMMA
            | &CH_SEMICOLON
            | &CH_LITTLE
            | &CH_GREATER
            | &CH_EQUAL
            | &CH_AMPERSAND
            | &CH_PIPE
            | &CH_EXCLAMATION
            => false,
            _ => !c.is_whitespace(),
        };
        let read = self.input.take_while(fun).map_err(to_token_error)?;
        if read.offset == 0 {
            Ok(Token::Absolute(read))
        } else {
            Ok(Token::Key(read))
        }
    }

    fn quote(&mut self, ch: char) -> Result<StrRange, TokenError> {
        let span = self.input.take_while(|c| *c != ch).map_err(to_token_error)?;
        let val = self.input.read(&span);
        if let Some('\\') = val.chars().last() {
            self.input.next_char().map_err(to_token_error)?;
            let remain_span = self.input.take_while(|c| *c != ch).map_err(to_token_error)?;
            self.input.next_char().map_err(to_token_error)?;
            Ok(StrRange::new(span.pos, remain_span.offset))
        } else {
            self.input.next_char().map_err(to_token_error)?;
            Ok(span)
        }
    }

    fn single_quote(&mut self, ch: char) -> Result<Token, TokenError> {
        Ok(Token::SingleQuoted(self.quote(ch)?))
    }

    fn double_quote(&mut self, ch: char) -> Result<Token, TokenError> {
        Ok(Token::DoubleQuoted(self.quote(ch)?))
    }

    fn equal(&mut self, span: StrRange) -> Result<Token, TokenError> {
        let ch = self.input.peek_char().map_err(to_token_error)?;
        match ch {
            CH_EQUAL => {
                self.input.next_char().map_err(to_token_error)?;
                Ok(Token::Equal(span))
            }
            _ => Err(TokenError::Position(span.pos)),
        }
    }

    fn not_equal(&mut self, span: StrRange) -> Result<Token, TokenError> {
        let ch = self.input.peek_char().map_err(to_token_error)?;
        match ch {
            CH_EQUAL => {
                self.input.next_char().map_err(to_token_error)?;
                Ok(Token::NotEqual(span))
            }
            _ => Err(TokenError::Position(span.pos)),
        }
    }

    fn little(&mut self, span: StrRange) -> Result<Token, TokenError> {
        let ch = self.input.peek_char().map_err(to_token_error)?;
        match ch {
            CH_EQUAL => {
                self.input.next_char().map_err(to_token_error)?;
                Ok(Token::LittleOrEqual(span))
            }
            _ => Ok(Token::Little(span)),
        }
    }

    fn greater(&mut self, span: StrRange) -> Result<Token, TokenError> {
        let ch = self.input.peek_char().map_err(to_token_error)?;
        match ch {
            CH_EQUAL => {
                self.input.next_char().map_err(to_token_error)?;
                Ok(Token::GreaterOrEqual(span))
            }
            _ => Ok(Token::Greater(span)),
        }
    }

    fn and(&mut self, span: StrRange) -> Result<Token, TokenError> {
        let ch = self.input.peek_char().map_err(to_token_error)?;
        match ch {
            CH_AMPERSAND => {
                let _ = self.input.next_char().map_err(to_token_error);
                Ok(Token::And(span))
            }
            _ => Err(TokenError::Position(span.pos)),
        }
    }

    fn or(&mut self, span: StrRange) -> Result<Token, TokenError> {
        let ch = self.input.peek_char().map_err(to_token_error)?;
        match ch {
            CH_PIPE => {
                self.input.next_char().map_err(to_token_error)?;
                Ok(Token::Or(span))
            }
            _ => Err(TokenError::Position(span.pos)),
        }
    }

    fn whitespace(&mut self) -> Result<Token, TokenError> {
        let span = self
            .input
            .take_while(|c| c.is_whitespace())
            .map_err(to_token_error)?;
        Ok(Token::Whitespace(span))
    }

    fn other(&mut self) -> Result<Token, TokenError> {
        let fun = |c: &char| match c {
            &CH_DOLLA
            | &CH_DOT
            | &CH_ASTERISK
            | &CH_LARRAY
            | &CH_RARRAY
            | &CH_LPAREN
            | &CH_RPAREN
            | &CH_AT
            | &CH_QUESTION
            | &CH_COMMA
            | &CH_SEMICOLON
            | &CH_LITTLE
            | &CH_GREATER
            | &CH_EQUAL
            | &CH_AMPERSAND
            | &CH_PIPE
            | &CH_EXCLAMATION
            => false,
            _ => !c.is_whitespace(),
        };
        let span = self.input.take_while(fun).map_err(to_token_error)?;
        Ok(Token::Key(span))
    }

    fn read_token(&mut self, span: StrRange, ch: char) -> Result<Token, TokenError> {
        match ch {
            CH_DOLLA => self.dolla(),
            CH_DOT => Ok(Token::Dot(span)),
            CH_ASTERISK => Ok(Token::Asterisk(span)),
            CH_LARRAY => Ok(Token::OpenArray(span)),
            CH_RARRAY => Ok(Token::CloseArray(span)),
            CH_LPAREN => Ok(Token::OpenParenthesis(span)),
            CH_RPAREN => Ok(Token::CloseParenthesis(span)),
            CH_AT => Ok(Token::At(span)),
            CH_QUESTION => Ok(Token::Question(span)),
            CH_COMMA => Ok(Token::Comma(span)),
            CH_SEMICOLON => Ok(Token::Split(span)),
            CH_SINGLE_QUOTE => self.single_quote(ch),
            CH_DOUBLE_QUOTE => self.double_quote(ch),
            CH_EQUAL => self.equal(span),
            CH_GREATER => self.greater(span),
            CH_LITTLE => self.little(span),
            CH_AMPERSAND => self.and(span),
            CH_PIPE => self.or(span),
            CH_EXCLAMATION => self.not_equal(span),
            _ if ch.is_whitespace() => self.whitespace(),
            _ => self.other(),
        }
    }

    pub fn next_token(&mut self) -> Result<Token, TokenError> {
        let (span, ch) = self.input.next_char().map_err(to_token_error)?;
        match self.read_token(span, ch) {
            Ok(t) => Ok(t),
            Err(e) => Err(e),
        }
    }

    fn current_pos(&self) -> usize {
        self.input.current_pos()
    }

    fn read_span(&self, span: &StrRange) -> &'a str {
        self.input.read(span)
    }
}

#[derive(Clone, Debug)]
pub(super) struct TokenReader<'a> {
    tokenizer: Tokenizer<'a>,
    curr_pos: usize,
    err: Option<TokenError>,
    peeked: Option<Result<Token, TokenError>>,
}

impl<'a> TokenReader<'a> {
    pub fn new(input: &'a str) -> Self {
        TokenReader {
            tokenizer: Tokenizer::new(input),
            curr_pos: 0,
            err: None,
            peeked: None,
        }
    }

    pub fn read_value(&self, str_range: &StrRange) -> &'a str {
        self.tokenizer.read_span(str_range)
    }

    pub fn peek_token(&mut self) -> Result<&Token, &TokenError> {
        let tokenizer = &mut self.tokenizer;
        let prev_pos = self.curr_pos;
        let peeked = self.peeked.get_or_insert_with(|| {
            let mut token = tokenizer.next_token();
            if let Ok(token) = &mut token {
                let token = token.reset_span(StrRange::new(prev_pos, tokenizer.current_pos() - prev_pos));
                return Ok(token);
            }
            token
        });
        self.curr_pos = tokenizer.current_pos();
        peeked.as_ref()
    }

    pub fn next_token(&mut self) -> Result<Token, TokenError> {
        match self.peeked.take() {
            Some(v) => v,
            None => {
                let prev_pos = self.curr_pos;
                let tokenizer = &mut self.tokenizer;
                let mut token = tokenizer.next_token();
                if let Ok(token) = &mut token {
                    let current_pos = tokenizer.current_pos();
                    let token = token.reset_span(StrRange::new(prev_pos, current_pos - prev_pos));
                    self.curr_pos = current_pos;
                    return Ok(token);
                }
                token
            }
        }
    }

    pub fn to_error(&self) -> TokenError {
        let path = self.tokenizer.input.origin_str();
        let curr_pos = self.curr_pos;
        if path.len() == curr_pos {
            TokenError::Eof
        } else {
            TokenError::Position(curr_pos)
        }
    }
}

#[cfg(test)]
mod tokenizer_tests {
    use paths::str_reader::StrRange;
    use paths::tokenizer::{TokenError, TokenReader};
    use paths::tokens::Token;

    fn setup() {
        let _ = env_logger::try_init();
    }

    fn collect_token(input: &str) -> (Vec<Token>, Option<TokenError>) {
        let mut tokenizer = TokenReader::new(input);
        let mut vec = vec![];
        loop {
            match tokenizer.next_token() {
                Ok(t) => vec.push(t),
                Err(e) => return (vec, Some(e)),
            }
        }
    }

    fn run(input: &str, expected: (Vec<Token>, Option<TokenError>)) {
        let (vec, err) = collect_token(input);
        assert_eq!((vec, err), expected, "\"{}\"", input);
    }

    #[test]
    fn peek() {
        let mut tokenizer = TokenReader::new("$.a");
        match tokenizer.next_token() {
            Ok(t) => assert_eq!(Token::Absolute(StrRange::new(0, 1)), t),
            _ => panic!(),
        }

        match tokenizer.peek_token() {
            Ok(t) => assert_eq!(&Token::Dot(StrRange::new(1, 1)), t),
            _ => panic!(),
        }

        match tokenizer.peek_token() {
            Ok(t) => assert_eq!(&Token::Dot(StrRange::new(1, 1)), t),
            _ => panic!(),
        }

        match tokenizer.next_token() {
            Ok(t) => assert_eq!(Token::Dot(StrRange::new(1, 1)), t),
            _ => panic!(),
        }
    }

    #[test]
    fn token() {
        setup();

        run(
            "$.01.a",
            (
                vec![
                    Token::Absolute(StrRange::new(0, 1)),
                    Token::Dot(StrRange::new(1, 1)),
                    Token::Key(StrRange::new(2, 2)),
                    Token::Dot(StrRange::new(4, 1)),
                    Token::Key(StrRange::new(5, 1)),
                ],
                Some(TokenError::Eof),
            ),
        );

        run(
            "$.   []",
            (
                vec![
                    Token::Absolute(StrRange::new(0, 1)),
                    Token::Dot(StrRange::new(1, 1)),
                    Token::Whitespace(StrRange::new(2, 3)),
                    Token::OpenArray(StrRange::new(5, 1)),
                    Token::CloseArray(StrRange::new(6, 1)),
                ],
                Some(TokenError::Eof),
            ),
        );

        run(
            "$..",
            (
                vec![Token::Absolute(StrRange::new(0, 1)), Token::Dot(StrRange::new(1, 1)), Token::Dot(StrRange::new(2, 1))],
                Some(TokenError::Eof),
            ),
        );

        run(
            "$..ab",
            (
                vec![
                    Token::Absolute(StrRange::new(0, 1)),
                    Token::Dot(StrRange::new(1, 1)),
                    Token::Dot(StrRange::new(2, 1)),
                    Token::Key(StrRange::new(3, "ab".len())),
                ],
                Some(TokenError::Eof),
            ),
        );

        run(
            "$..가 [",
            (
                vec![
                    Token::Absolute(StrRange::new(0, 1)),
                    Token::Dot(StrRange::new(1, 1)),
                    Token::Dot(StrRange::new(2, 1)),
                    Token::Key(StrRange::new(3, '가'.len_utf8())),
                    Token::Whitespace(StrRange::new(6, 1)),
                    Token::OpenArray(StrRange::new(7, 1)),
                ],
                Some(TokenError::Eof),
            ),
        );

        run(
            "[-1, 2 ]",
            (
                vec![
                    Token::OpenArray(StrRange::new(0, 1)),
                    Token::Key(StrRange::new(1, "-1".len())),
                    Token::Comma(StrRange::new(3, 1)),
                    Token::Whitespace(StrRange::new(4, 1)),
                    Token::Key(StrRange::new(5, "2".len())),
                    Token::Whitespace(StrRange::new(6, 1)),
                    Token::CloseArray(StrRange::new(7, 1)),
                ],
                Some(TokenError::Eof),
            ),
        );

        run(
            "[ 1 2 , 3 \"abc\" : -10 ]",
            (
                vec![
                    Token::OpenArray(StrRange::new(0, 1)),
                    Token::Whitespace(StrRange::new(1, 1)),
                    Token::Key(StrRange::new(2, "1".len())),
                    Token::Whitespace(StrRange::new(3, 1)),
                    Token::Key(StrRange::new(4, "2".len())),
                    Token::Whitespace(StrRange::new(5, 1)),
                    Token::Comma(StrRange::new(6, 1)),
                    Token::Whitespace(StrRange::new(7, 1)),
                    Token::Key(StrRange::new(8, "3".len())),
                    Token::Whitespace(StrRange::new(9, 1)),
                    Token::DoubleQuoted(StrRange::new(10, "\"abc\"".len())),
                    Token::Whitespace(StrRange::new(15, 1)),
                    Token::Split(StrRange::new(16, 1)),
                    Token::Whitespace(StrRange::new(17, 1)),
                    Token::Key(StrRange::new(18, "-10".len())),
                    Token::Whitespace(StrRange::new(21, 1)),
                    Token::CloseArray(StrRange::new(22, 1)),
                ],
                Some(TokenError::Eof),
            ),
        );

        run(
            "?(@.a가 <41.01)",
            (
                vec![
                    Token::Question(StrRange::new(0, 1)),
                    Token::OpenParenthesis(StrRange::new(1, 1)),
                    Token::At(StrRange::new(2, 1)),
                    Token::Dot(StrRange::new(3, 1)),
                    Token::Key(StrRange::new(4, "a가".chars().map(|c| c.len_utf8()).sum())),
                    Token::Whitespace(StrRange::new(8, 1)),
                    Token::Little(StrRange::new(9, 1)),
                    Token::Key(StrRange::new(10, "41".len())),
                    Token::Dot(StrRange::new(12, 1)),
                    Token::Key(StrRange::new(13, "01".len())),
                    Token::CloseParenthesis(StrRange::new(15, 1)),
                ],
                Some(TokenError::Eof),
            ),
        );

        run(
            "?(@.a <4a.01)",
            (
                vec![
                    Token::Question(StrRange::new(0, 1)),
                    Token::OpenParenthesis(StrRange::new(1, 1)),
                    Token::At(StrRange::new(2, 1)),
                    Token::Dot(StrRange::new(3, 1)),
                    Token::Key(StrRange::new(4, "a".len())),
                    Token::Whitespace(StrRange::new(5, 1)),
                    Token::Little(StrRange::new(6, 1)),
                    Token::Key(StrRange::new(7, "4a".len())),
                    Token::Dot(StrRange::new(9, 1)),
                    Token::Key(StrRange::new(10, "01".len())),
                    Token::CloseParenthesis(StrRange::new(12, 1)),
                ],
                Some(TokenError::Eof),
            ),
        );

        run(
            "?($.c>@.d)",
            (
                vec![
                    Token::Question(StrRange::new(0, 1)),
                    Token::OpenParenthesis(StrRange::new(1, 1)),
                    Token::Absolute(StrRange::new(2, 1)),
                    Token::Dot(StrRange::new(3, 1)),
                    Token::Key(StrRange::new(4, 1)),
                    Token::Greater(StrRange::new(5, 1)),
                    Token::At(StrRange::new(6, 1)),
                    Token::Dot(StrRange::new(7, 1)),
                    Token::Key(StrRange::new(8, 1)),
                    Token::CloseParenthesis(StrRange::new(9, 1)),
                ],
                Some(TokenError::Eof),
            ),
        );

        run(
            "$[:]",
            (
                vec![
                    Token::Absolute(StrRange::new(0, 1)),
                    Token::OpenArray(StrRange::new(1, 1)),
                    Token::Split(StrRange::new(2, 1)),
                    Token::CloseArray(StrRange::new(3, 1)),
                ],
                Some(TokenError::Eof),
            ),
        );

        run(
            r#"$['single\'quote']"#,
            (
                vec![
                    Token::Absolute(StrRange::new(0, 1)),
                    Token::OpenArray(StrRange::new(1, 1)),
                    Token::SingleQuoted(StrRange::new(2, r#"'single\'quote'"#.len())),
                    Token::CloseArray(StrRange::new(17, 1)),
                ],
                Some(TokenError::Eof),
            ),
        );

        run(
            r#"$['single\'1','single\'2']"#,
            (
                vec![
                    Token::Absolute(StrRange::new(0, 1)),
                    Token::OpenArray(StrRange::new(1, 1)),
                    Token::SingleQuoted(StrRange::new(2, r#"'single\'1'"#.len())),
                    Token::Comma(StrRange::new(13, 1)),
                    Token::SingleQuoted(StrRange::new(14, r#"'single\'2'"#.len())),
                    Token::CloseArray(StrRange::new(25, 1)),
                ],
                Some(TokenError::Eof),
            ),
        );

        run(
            r#"$["double\"quote"]"#,
            (
                vec![
                    Token::Absolute(StrRange::new(0, 1)),
                    Token::OpenArray(StrRange::new(1, 1)),
                    Token::DoubleQuoted(StrRange::new(2, r#""double\"quote""#.len())),
                    Token::CloseArray(StrRange::new(17, 1)),
                ],
                Some(TokenError::Eof),
            ),
        );
    }
}