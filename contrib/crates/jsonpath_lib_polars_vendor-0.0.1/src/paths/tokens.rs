use super::str_reader::StrRange;

#[derive(Debug, PartialEq, Clone)]
pub enum Token {
    Absolute(StrRange),
    Dot(StrRange),
    At(StrRange),
    OpenArray(StrRange),
    CloseArray(StrRange),
    Asterisk(StrRange),
    Question(StrRange),
    Comma(StrRange),
    Split(StrRange),
    OpenParenthesis(StrRange),
    CloseParenthesis(StrRange),
    Key(StrRange),
    DoubleQuoted(StrRange),
    SingleQuoted(StrRange),
    Equal(StrRange),
    GreaterOrEqual(StrRange),
    Greater(StrRange),
    Little(StrRange),
    LittleOrEqual(StrRange),
    NotEqual(StrRange),
    And(StrRange),
    Or(StrRange),
    Whitespace(StrRange),
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
            Token::Key(_) => matches!(other, Token::Key(_)),
            Token::DoubleQuoted(_) => matches!(other, Token::DoubleQuoted(_)),
            Token::SingleQuoted(_) => matches!(other, Token::SingleQuoted(_)),
            Token::Equal(_) => matches!(other, Token::Equal(_)),
            Token::GreaterOrEqual(_) => matches!(other, Token::GreaterOrEqual(_)),
            Token::Greater(_) => matches!(other, Token::Greater(_)),
            Token::Little(_) => matches!(other, Token::Little(_)),
            Token::LittleOrEqual(_) => matches!(other, Token::LittleOrEqual(_)),
            Token::NotEqual(_) => matches!(other, Token::NotEqual(_)),
            Token::And(_) => matches!(other, Token::And(_)),
            Token::Or(_) => matches!(other, Token::Or(_)),
            Token::Whitespace(_) => matches!(other, Token::Whitespace(_)),
        }
    }

    pub fn reset_span(&mut self, new_span: StrRange) -> Token {
        match self {
            Token::Absolute(_) => Token::Absolute(new_span),
            Token::Dot(_) => Token::Dot(new_span),
            Token::At(_) => Token::At(new_span),
            Token::OpenArray(_) => Token::OpenArray(new_span),
            Token::CloseArray(_) => Token::CloseArray(new_span),
            Token::Asterisk(_) => Token::Asterisk(new_span),
            Token::Question(_) => Token::Question(new_span),
            Token::Comma(_) => Token::Comma(new_span),
            Token::Split(_) => Token::Split(new_span),
            Token::OpenParenthesis(_) => Token::OpenParenthesis(new_span),
            Token::CloseParenthesis(_) => Token::CloseParenthesis(new_span),
            Token::Key(_) => Token::Key(new_span),
            Token::DoubleQuoted(_) => Token::DoubleQuoted(new_span),
            Token::SingleQuoted(_) => Token::SingleQuoted(new_span),
            Token::Equal(_) => Token::Equal(new_span),
            Token::GreaterOrEqual(_) => Token::GreaterOrEqual(new_span),
            Token::Greater(_) => Token::Greater(new_span),
            Token::Little(_) => Token::Little(new_span),
            Token::LittleOrEqual(_) => Token::LittleOrEqual(new_span),
            Token::NotEqual(_) => Token::NotEqual(new_span),
            Token::And(_) => Token::And(new_span),
            Token::Or(_) => Token::Or(new_span),
            Token::Whitespace(_) => Token::Whitespace(new_span),
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum ParseToken {
    // '$'
    Absolute,
    // '@'
    Relative,
    // '.'
    In,
    // '..'
    Leaves,
    // '*'
    All,

    Key(StrRange),
    Keys(Vec<StrRange>),
    // []
    Array,
    // 메타토큰
    ArrayEof,
    // ?( filter )
    Filter(FilterToken),
    // 1 : 2
    Range(Option<isize>, Option<isize>, Option<usize>),
    // 1, 2, 3
    Union(Vec<isize>),

    Number(f64),

    Bool(bool),

    Eof,
}

#[derive(Debug, PartialEq, Clone)]
pub enum FilterToken {
    Equal,
    NotEqual,
    Little,
    LittleOrEqual,
    Greater,
    GreaterOrEqual,
    And,
    Or,
}