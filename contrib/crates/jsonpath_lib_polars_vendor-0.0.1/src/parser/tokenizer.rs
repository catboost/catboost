use std::result::Result;

use super::path_reader::{PathReader, ReaderError};

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

#[derive(Debug, PartialEq)]
pub enum Token {
    Absolute(usize),
    Dot(usize),
    At(usize),
    OpenArray(usize),
    CloseArray(usize),
    Asterisk(usize),
    Question(usize),
    Comma(usize),
    Split(usize),
    OpenParenthesis(usize),
    CloseParenthesis(usize),
    Key(usize, String),
    DoubleQuoted(usize, String),
    SingleQuoted(usize, String),
    Equal(usize),
    GreaterOrEqual(usize),
    Greater(usize),
    Little(usize),
    LittleOrEqual(usize),
    NotEqual(usize),
    And(usize),
    Or(usize),
    Whitespace(usize, usize),
}

impl Token {
    pub fn is_match_token_type(&self, other: Token) -> bool {
        match self {
            Token::Absolute(_) => matches!(other, Token::Absolute(_)),
            Token::Dot(_) => matches!(other, Token::Dot(_)),
            Token::At(_) => matches!(other, Token::At(_)),
            Token::OpenArray(_) => matches!(other, Token::OpenArray(_)),
            Token::CloseArray(_) => matches!(other, Token::CloseArray(_)),
            Token::Asterisk(_) => matches!(other, Token::Asterisk(_)),
            Token::Question(_) => matches!(other, Token::Question(_)),
            Token::Comma(_) => matches!(other, Token::Comma(_)),
            Token::Split(_) => matches!(other, Token::Split(_)),
            Token::OpenParenthesis(_) => matches!(other, Token::OpenParenthesis(_)),
            Token::CloseParenthesis(_) => matches!(other, Token::CloseParenthesis(_)),
            Token::Key(_, _) => matches!(other, Token::Key(_, _)),
            Token::DoubleQuoted(_, _) => matches!(other, Token::DoubleQuoted(_, _)),
            Token::SingleQuoted(_, _) => matches!(other, Token::SingleQuoted(_, _)),
            Token::Equal(_) => matches!(other, Token::Equal(_)),
            Token::GreaterOrEqual(_) => matches!(other, Token::GreaterOrEqual(_)),
            Token::Greater(_) => matches!(other, Token::Greater(_)),
            Token::Little(_) => matches!(other, Token::Little(_)),
            Token::LittleOrEqual(_) => matches!(other, Token::LittleOrEqual(_)),
            Token::NotEqual(_) => matches!(other, Token::NotEqual(_)),
            Token::And(_) => matches!(other, Token::And(_)),
            Token::Or(_) => matches!(other, Token::Or(_)),
            Token::Whitespace(_, _) => matches!(other, Token::Whitespace(_, _)),
        }
    }
}

pub struct Tokenizer<'a> {
    input: PathReader<'a>,
}

impl<'a> Tokenizer<'a> {
    pub fn new(input: &'a str) -> Self {
        trace!("input: {}", input);
        Tokenizer {
            input: PathReader::new(input),
        }
    }

    fn dolla(&mut self, pos: usize, ch: char) -> Result<Token, TokenError> {
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
        let (_, mut vec) = self.input.take_while(fun).map_err(to_token_error)?;
        vec.insert(0, ch);

        if vec.len() == 1 {
            Ok(Token::Absolute(pos))
        } else {
            Ok(Token::Key(pos, vec))
        }
    }

    fn quote(&mut self, ch: char) -> Result<String, TokenError> {
        let (_, mut val) = self
            .input
            .take_while(|c| *c != ch)
            .map_err(to_token_error)?;

        if let Some('\\') = val.chars().last() {
            self.input.next_char().map_err(to_token_error)?;
            let _ = val.pop();
            let (_, val_remain) = self
                .input
                .take_while(|c| *c != ch)
                .map_err(to_token_error)?;
            self.input.next_char().map_err(to_token_error)?;
            val.push(ch);
            val.push_str(val_remain.as_str());
        } else {
            self.input.next_char().map_err(to_token_error)?;
        }

        Ok(val)
    }

    fn single_quote(&mut self, pos: usize, ch: char) -> Result<Token, TokenError> {
        let val = self.quote(ch)?;
        Ok(Token::SingleQuoted(pos, val))
    }

    fn double_quote(&mut self, pos: usize, ch: char) -> Result<Token, TokenError> {
        let val = self.quote(ch)?;
        Ok(Token::DoubleQuoted(pos, val))
    }

    fn equal(&mut self, pos: usize, _: char) -> Result<Token, TokenError> {
        let (_, ch) = self.input.peek_char().map_err(to_token_error)?;
        match ch {
            CH_EQUAL => {
                self.input.next_char().map_err(to_token_error)?;
                Ok(Token::Equal(pos))
            }
            _ => Err(TokenError::Position(pos)),
        }
    }

    fn not_equal(&mut self, pos: usize, _: char) -> Result<Token, TokenError> {
        let (_, ch) = self.input.peek_char().map_err(to_token_error)?;
        match ch {
            CH_EQUAL => {
                self.input.next_char().map_err(to_token_error)?;
                Ok(Token::NotEqual(pos))
            }
            _ => Err(TokenError::Position(pos)),
        }
    }

    fn little(&mut self, pos: usize, _: char) -> Result<Token, TokenError> {
        let (_, ch) = self.input.peek_char().map_err(to_token_error)?;
        match ch {
            CH_EQUAL => {
                self.input.next_char().map_err(to_token_error)?;
                Ok(Token::LittleOrEqual(pos))
            }
            _ => Ok(Token::Little(pos)),
        }
    }

    fn greater(&mut self, pos: usize, _: char) -> Result<Token, TokenError> {
        let (_, ch) = self.input.peek_char().map_err(to_token_error)?;
        match ch {
            CH_EQUAL => {
                self.input.next_char().map_err(to_token_error)?;
                Ok(Token::GreaterOrEqual(pos))
            }
            _ => Ok(Token::Greater(pos)),
        }
    }

    fn and(&mut self, pos: usize, _: char) -> Result<Token, TokenError> {
        let (_, ch) = self.input.peek_char().map_err(to_token_error)?;
        match ch {
            CH_AMPERSAND => {
                let _ = self.input.next_char().map_err(to_token_error);
                Ok(Token::And(pos))
            }
            _ => Err(TokenError::Position(pos)),
        }
    }

    fn or(&mut self, pos: usize, _: char) -> Result<Token, TokenError> {
        let (_, ch) = self.input.peek_char().map_err(to_token_error)?;
        match ch {
            CH_PIPE => {
                self.input.next_char().map_err(to_token_error)?;
                Ok(Token::Or(pos))
            }
            _ => Err(TokenError::Position(pos)),
        }
    }

    fn whitespace(&mut self, pos: usize, _: char) -> Result<Token, TokenError> {
        let (_, vec) = self
            .input
            .take_while(|c| c.is_whitespace())
            .map_err(to_token_error)?;
        Ok(Token::Whitespace(pos, vec.len()))
    }

    fn other(&mut self, pos: usize, ch: char) -> Result<Token, TokenError> {
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
        let (_, mut vec) = self.input.take_while(fun).map_err(to_token_error)?;
        vec.insert(0, ch);
        Ok(Token::Key(pos, vec))
    }

    pub fn next_token(&mut self) -> Result<Token, TokenError> {
        let (pos, ch) = self.input.next_char().map_err(to_token_error)?;
        match ch {
            CH_DOLLA => self.dolla(pos, ch),
            CH_DOT => Ok(Token::Dot(pos)),
            CH_ASTERISK => Ok(Token::Asterisk(pos)),
            CH_LARRAY => Ok(Token::OpenArray(pos)),
            CH_RARRAY => Ok(Token::CloseArray(pos)),
            CH_LPAREN => Ok(Token::OpenParenthesis(pos)),
            CH_RPAREN => Ok(Token::CloseParenthesis(pos)),
            CH_AT => Ok(Token::At(pos)),
            CH_QUESTION => Ok(Token::Question(pos)),
            CH_COMMA => Ok(Token::Comma(pos)),
            CH_SEMICOLON => Ok(Token::Split(pos)),
            CH_SINGLE_QUOTE => self.single_quote(pos, ch),
            CH_DOUBLE_QUOTE => self.double_quote(pos, ch),
            CH_EQUAL => self.equal(pos, ch),
            CH_GREATER => self.greater(pos, ch),
            CH_LITTLE => self.little(pos, ch),
            CH_AMPERSAND => self.and(pos, ch),
            CH_PIPE => self.or(pos, ch),
            CH_EXCLAMATION => self.not_equal(pos, ch),
            _ if ch.is_whitespace() => self.whitespace(pos, ch),
            _ => self.other(pos, ch),
        }
    }

    fn current_pos(&self) -> usize {
        self.input.current_pos()
    }
}

pub struct TokenReader<'a> {
    origin_input: &'a str,
    err: TokenError,
    err_pos: usize,
    tokens: Vec<(usize, Token)>,
    curr_pos: Option<usize>,
}

impl<'a> TokenReader<'a> {
    pub fn new(input: &'a str) -> Self {
        let mut tokenizer = Tokenizer::new(input);
        let mut tokens = vec![];
        loop {
            match tokenizer.next_token() {
                Ok(t) => {
                    tokens.insert(0, (tokenizer.current_pos(), t));
                }
                Err(e) => {
                    return TokenReader {
                        origin_input: input,
                        err: e,
                        err_pos: tokenizer.current_pos(),
                        tokens,
                        curr_pos: None,
                    };
                }
            }
        }
    }

    pub fn peek_token(&self) -> Result<&Token, TokenError> {
        match self.tokens.last() {
            Some((_, t)) => {
                trace!("%{:?}", t);
                Ok(t)
            }
            _ => {
                trace!("%{:?}", self.err);
                Err(self.err.clone())
            }
        }
    }

    pub fn next_token(&mut self) -> Result<Token, TokenError> {
        match self.tokens.pop() {
            Some((pos, t)) => {
                self.curr_pos = Some(pos);
                trace!("@{:?}", t);
                Ok(t)
            }
            _ => {
                trace!("@{:?}", self.err);
                Err(self.err.clone())
            }
        }
    }

    pub fn err_msg_with_pos(&self, pos: usize) -> String {
        format!("{}\n{}", self.origin_input, "^".repeat(pos))
    }

    pub fn err_msg(&self) -> String {
        match self.curr_pos {
            Some(pos) => self.err_msg_with_pos(pos),
            _ => self.err_msg_with_pos(self.err_pos),
        }
    }
}
