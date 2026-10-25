mod path_reader;
mod tokenizer;

use std::str::FromStr;

use self::tokenizer::*;

const DUMMY: usize = 0;

type ParseResult<T> = Result<T, String>;

mod utils {
    use std::str::FromStr;

    pub fn string_to_num<F, S: FromStr>(string: &str, msg_handler: F) -> Result<S, String>
        where
            F: Fn() -> String,
    {
        match string.parse() {
            Ok(n) => Ok(n),
            _ => Err(msg_handler()),
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

    Key(String),
    Keys(Vec<String>),
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

#[deprecated(since = "0.4.0", note = "Please use `paths::PathParser`")]
#[derive(Debug, Clone)]
pub struct Node {
    left: Option<Box<Node>>,
    right: Option<Box<Node>>,
    token: ParseToken,
}

#[deprecated(since = "0.4.0", note = "Please use `paths::PathParser`")]
pub struct Parser;

#[allow(deprecated)]
impl Parser {
    pub fn compile(input: &str) -> ParseResult<Node> {
        let mut tokenizer = TokenReader::new(input);
        Self::json_path(&mut tokenizer)
    }

    fn json_path(tokenizer: &mut TokenReader) -> ParseResult<Node> {
        debug!("#json_path");
        match tokenizer.next_token() {
            Ok(Token::Absolute(_)) => {
                let node = Self::node(ParseToken::Absolute);
                Self::paths(node, tokenizer)
            }
            _ => Err(tokenizer.err_msg()),
        }
    }

    fn paths(prev: Node, tokenizer: &mut TokenReader) -> ParseResult<Node> {
        debug!("#paths");
        match tokenizer.peek_token() {
            Ok(Token::Dot(_)) => {
                Self::eat_token(tokenizer);
                Self::paths_dot(prev, tokenizer)
            }
            Ok(Token::OpenArray(_)) => {
                Self::eat_token(tokenizer);
                Self::eat_whitespace(tokenizer);
                let node = Self::array(prev, tokenizer)?;
                Self::paths(node, tokenizer)
            }
            _ => Ok(prev),
        }
    }

    fn paths_dot(prev: Node, tokenizer: &mut TokenReader) -> ParseResult<Node> {
        debug!("#paths_dot");
        let node = Self::path(prev, tokenizer)?;
        Self::paths(node, tokenizer)
    }

    fn path(prev: Node, tokenizer: &mut TokenReader) -> ParseResult<Node> {
        debug!("#path");
        match tokenizer.peek_token() {
            Ok(Token::Dot(_)) => Self::path_leaves(prev, tokenizer),
            Ok(Token::Asterisk(_)) => Self::path_in_all(prev, tokenizer),
            Ok(Token::Key(_, _)) => Self::path_in_key(prev, tokenizer),
            Ok(Token::OpenArray(_)) => {
                Self::eat_token(tokenizer);
                Self::array(prev, tokenizer)
            }
            _ => Err(tokenizer.err_msg()),
        }
    }

    fn path_leaves(prev: Node, tokenizer: &mut TokenReader) -> ParseResult<Node> {
        debug!("#path_leaves");
        Self::eat_token(tokenizer);
        match tokenizer.peek_token() {
            Ok(Token::Asterisk(_)) => Self::path_leaves_all(prev, tokenizer),
            Ok(Token::OpenArray(_)) => {
                let mut leaves_node = Self::node(ParseToken::Leaves);
                leaves_node.left = Some(Box::new(prev));
                Ok(Self::paths(leaves_node, tokenizer)?)
            }
            _ => Self::path_leaves_key(prev, tokenizer),
        }
    }

    #[allow(clippy::unnecessary_wraps)]
    fn path_leaves_key(prev: Node, tokenizer: &mut TokenReader) -> ParseResult<Node> {
        debug!("#path_leaves_key");
        Ok(Node {
            token: ParseToken::Leaves,
            left: Some(Box::new(prev)),
            right: Some(Box::new(Self::key(tokenizer)?)),
        })
    }

    #[allow(clippy::unnecessary_wraps)]
    fn path_leaves_all(prev: Node, tokenizer: &mut TokenReader) -> ParseResult<Node> {
        debug!("#path_leaves_all");
        Self::eat_token(tokenizer);
        Ok(Node {
            token: ParseToken::Leaves,
            left: Some(Box::new(prev)),
            right: Some(Box::new(Self::node(ParseToken::All))),
        })
    }

    #[allow(clippy::unnecessary_wraps)]
    fn path_in_all(prev: Node, tokenizer: &mut TokenReader) -> ParseResult<Node> {
        debug!("#path_in_all");
        Self::eat_token(tokenizer);
        Ok(Node {
            token: ParseToken::In,
            left: Some(Box::new(prev)),
            right: Some(Box::new(Self::node(ParseToken::All))),
        })
    }

    #[allow(clippy::unnecessary_wraps)]
    fn path_in_key(prev: Node, tokenizer: &mut TokenReader) -> ParseResult<Node> {
        debug!("#path_in_key");
        Ok(Node {
            token: ParseToken::In,
            left: Some(Box::new(prev)),
            right: Some(Box::new(Self::key(tokenizer)?)),
        })
    }

    fn key(tokenizer: &mut TokenReader) -> ParseResult<Node> {
        debug!("#key");
        match tokenizer.next_token() {
            Ok(Token::Key(_, v)) => Ok(Self::node(ParseToken::Key(v))),
            _ => Err(tokenizer.err_msg()),
        }
    }

    fn boolean(tokenizer: &mut TokenReader) -> ParseResult<Node> {
        debug!("#boolean");

        fn validation_bool_value(v: &str) -> bool {
            let b = v.as_bytes();
            !b.is_empty() && (b[0] == b't' || b[0] == b'T' || b[0] == b'f' || b[0] == b'F')
        }

        match tokenizer.next_token() {
            Ok(Token::Key(_, ref v)) if validation_bool_value(v) => {
                Ok(Self::node(ParseToken::Bool(v.eq_ignore_ascii_case("true"))))
            }
            _ => Err(tokenizer.err_msg()),
        }
    }

    fn array_keys(tokenizer: &mut TokenReader, first_key: String) -> ParseResult<Node> {
        let mut keys = vec![first_key];

        while let Ok(Token::Comma(_)) = tokenizer.peek_token() {
            Self::eat_token(tokenizer);
            Self::eat_whitespace(tokenizer);

            match tokenizer.next_token() {
                Ok(Token::SingleQuoted(_, val)) | Ok(Token::DoubleQuoted(_, val)) => {
                    keys.push(val);
                }
                _ => return Err(tokenizer.err_msg()),
            }

            Self::eat_whitespace(tokenizer);
        }

        Ok(Self::node(ParseToken::Keys(keys)))
    }

    fn array_quote_value(tokenizer: &mut TokenReader) -> ParseResult<Node> {
        debug!("#array_quote_value");
        match tokenizer.next_token() {
            Ok(Token::SingleQuoted(_, val)) | Ok(Token::DoubleQuoted(_, val)) => {
                if let Ok(Token::Comma(_)) = tokenizer.peek_token() {
                    Self::array_keys(tokenizer, val)
                } else {
                    Ok(Self::node(ParseToken::Key(val)))
                }
            }
            _ => Err(tokenizer.err_msg()),
        }
    }

    fn array_start(prev: Node, tokenizer: &mut TokenReader) -> ParseResult<Node> {
        debug!("#array_start");
        match tokenizer.peek_token() {
            Ok(Token::Question(_)) => {
                Self::eat_token(tokenizer);
                Ok(Node {
                    token: ParseToken::Array,
                    left: Some(Box::new(prev)),
                    right: Some(Box::new(Self::filter(tokenizer)?)),
                })
            }
            Ok(Token::Asterisk(_)) => {
                Self::eat_token(tokenizer);
                Ok(Node {
                    token: ParseToken::Array,
                    left: Some(Box::new(prev)),
                    right: Some(Box::new(Self::node(ParseToken::All))),
                })
            }
            _ => Ok(Node {
                token: ParseToken::Array,
                left: Some(Box::new(prev)),
                right: Some(Box::new(Self::array_value(tokenizer)?)),
            }),
        }
    }

    fn array(prev: Node, tokenizer: &mut TokenReader) -> ParseResult<Node> {
        debug!("#array");
        let ret = Self::array_start(prev, tokenizer)?;
        Self::eat_whitespace(tokenizer);
        Self::close_token(ret, Token::CloseArray(DUMMY), tokenizer)
    }

    fn array_value_key(tokenizer: &mut TokenReader) -> ParseResult<Node> {
        debug!("#array_value_key");
        match tokenizer.next_token() {
            Ok(Token::Key(pos, ref val)) => {
                let digit = utils::string_to_num(val, || tokenizer.err_msg_with_pos(pos))?;
                Self::eat_whitespace(tokenizer);

                match tokenizer.peek_token() {
                    Ok(Token::Comma(_)) => Self::union(digit, tokenizer),
                    Ok(Token::Split(_)) => Self::range_from(digit, tokenizer),
                    _ => Ok(Self::node(ParseToken::Number(digit as f64))),
                }
            }
            _ => Err(tokenizer.err_msg()),
        }
    }

    fn array_value(tokenizer: &mut TokenReader) -> ParseResult<Node> {
        debug!("#array_value");
        match tokenizer.peek_token() {
            Ok(Token::Key(_, _)) => Self::array_value_key(tokenizer),
            Ok(Token::Split(_)) => {
                Self::eat_token(tokenizer);
                Self::range_to(tokenizer)
            }
            Ok(Token::DoubleQuoted(_, _)) | Ok(Token::SingleQuoted(_, _)) => {
                Self::array_quote_value(tokenizer)
            }
            Err(TokenError::Eof) => Ok(Self::node(ParseToken::Eof)),
            _ => {
                Self::eat_token(tokenizer);
                Err(tokenizer.err_msg())
            }
        }
    }

    fn union(num: isize, tokenizer: &mut TokenReader) -> ParseResult<Node> {
        debug!("#union");
        let mut values = vec![num];
        while matches!(tokenizer.peek_token(), Ok(Token::Comma(_))) {
            Self::eat_token(tokenizer);
            Self::eat_whitespace(tokenizer);
            match tokenizer.next_token() {
                Ok(Token::Key(pos, ref val)) => {
                    let digit = utils::string_to_num(val, || tokenizer.err_msg_with_pos(pos))?;
                    values.push(digit);
                }
                _ => {
                    return Err(tokenizer.err_msg());
                }
            }
        }
        Ok(Self::node(ParseToken::Union(values)))
    }

    fn range_value<S: FromStr>(tokenizer: &mut TokenReader) -> Result<Option<S>, String> {
        Self::eat_whitespace(tokenizer);

        match tokenizer.peek_token() {
            Ok(Token::Split(_)) => {
                Self::eat_token(tokenizer);
                Self::eat_whitespace(tokenizer);
            }
            _ => {
                return Ok(None);
            }
        }

        match tokenizer.peek_token() {
            Ok(Token::Key(_, _)) => {}
            _ => {
                return Ok(None);
            }
        }

        match tokenizer.next_token() {
            Ok(Token::Key(pos, str_step)) => {
                match utils::string_to_num(&str_step, || tokenizer.err_msg_with_pos(pos)) {
                    Ok(step) => Ok(Some(step)),
                    Err(e) => Err(e),
                }
            }
            _ => {
                unreachable!();
            }
        }
    }

    fn range_from(from: isize, tokenizer: &mut TokenReader) -> ParseResult<Node> {
        debug!("#range_from");
        Self::eat_token(tokenizer);
        Self::eat_whitespace(tokenizer);

        match tokenizer.peek_token() {
            Ok(Token::Key(_, _)) => Self::range(from, tokenizer),
            Ok(Token::Split(_)) => match Self::range_value(tokenizer)? {
                Some(step) => Ok(Self::node(ParseToken::Range(Some(from), None, Some(step)))),
                _ => Ok(Self::node(ParseToken::Range(Some(from), None, None))),
            },
            _ => Ok(Self::node(ParseToken::Range(Some(from), None, None))),
        }
    }

    fn range_to(tokenizer: &mut TokenReader) -> ParseResult<Node> {
        debug!("#range_to");

        if let Some(step) = Self::range_value(tokenizer)? {
            return Ok(Self::node(ParseToken::Range(None, None, Some(step))));
        }

        if let Ok(Token::CloseArray(_)) = tokenizer.peek_token() {
            return Ok(Self::node(ParseToken::Range(None, None, None)));
        }

        match tokenizer.next_token() {
            Ok(Token::Key(pos, ref to_str)) => {
                let to = utils::string_to_num(to_str, || tokenizer.err_msg_with_pos(pos))?;
                let step = Self::range_value(tokenizer)?;
                Ok(Self::node(ParseToken::Range(None, Some(to), step)))
            }
            _ => Err(tokenizer.err_msg()),
        }
    }

    fn range(from: isize, tokenizer: &mut TokenReader) -> ParseResult<Node> {
        debug!("#range");
        match tokenizer.next_token() {
            Ok(Token::Key(pos, ref str_to)) => {
                let to = utils::string_to_num(str_to, || tokenizer.err_msg_with_pos(pos))?;
                let step = Self::range_value(tokenizer)?;
                Ok(Self::node(ParseToken::Range(Some(from), Some(to), step)))
            }
            _ => Err(tokenizer.err_msg()),
        }
    }

    fn filter(tokenizer: &mut TokenReader) -> ParseResult<Node> {
        debug!("#filter");
        match tokenizer.next_token() {
            Ok(Token::OpenParenthesis(_)) => {
                let ret = Self::exprs(tokenizer)?;
                Self::eat_whitespace(tokenizer);
                Self::close_token(ret, Token::CloseParenthesis(DUMMY), tokenizer)
            }
            _ => Err(tokenizer.err_msg()),
        }
    }

    fn exprs(tokenizer: &mut TokenReader) -> ParseResult<Node> {
        Self::eat_whitespace(tokenizer);
        debug!("#exprs");
        let node = match tokenizer.peek_token() {
            Ok(Token::OpenParenthesis(_)) => {
                Self::eat_token(tokenizer);
                trace!("\t-exprs - open_parenthesis");
                let ret = Self::exprs(tokenizer)?;
                Self::eat_whitespace(tokenizer);
                Self::close_token(ret, Token::CloseParenthesis(DUMMY), tokenizer)?
            }
            _ => {
                trace!("\t-exprs - else");
                Self::expr(tokenizer)?
            }
        };
        Self::eat_whitespace(tokenizer);
        Self::condition_expr(node, tokenizer)
    }

    fn condition_expr(prev: Node, tokenizer: &mut TokenReader) -> ParseResult<Node> {
        debug!("#condition_expr");
        match tokenizer.peek_token() {
            Ok(Token::And(_)) => {
                Self::eat_token(tokenizer);
                Ok(Node {
                    token: ParseToken::Filter(FilterToken::And),
                    left: Some(Box::new(prev)),
                    right: Some(Box::new(Self::exprs(tokenizer)?)),
                })
            }
            Ok(Token::Or(_)) => {
                Self::eat_token(tokenizer);
                Ok(Node {
                    token: ParseToken::Filter(FilterToken::Or),
                    left: Some(Box::new(prev)),
                    right: Some(Box::new(Self::exprs(tokenizer)?)),
                })
            }
            _ => Ok(prev),
        }
    }

    fn expr(tokenizer: &mut TokenReader) -> ParseResult<Node> {
        debug!("#expr");

        let has_prop_candidate = matches!(tokenizer.peek_token(), Ok(Token::At(_)));

        let node = Self::term(tokenizer)?;
        Self::eat_whitespace(tokenizer);

        if matches!(tokenizer.peek_token(),
            Ok(Token::Equal(_))
            | Ok(Token::NotEqual(_))
            | Ok(Token::Little(_))
            | Ok(Token::LittleOrEqual(_))
            | Ok(Token::Greater(_))
            | Ok(Token::GreaterOrEqual(_)))
        {
            Self::op(node, tokenizer)
        } else if has_prop_candidate {
            Ok(node)
        } else {
            Err(tokenizer.err_msg())
        }
    }

    fn term_num(tokenizer: &mut TokenReader) -> ParseResult<Node> {
        debug!("#term_num");
        match tokenizer.next_token() {
            Ok(Token::Key(pos, val)) => match tokenizer.peek_token() {
                Ok(Token::Dot(_)) => Self::term_num_float(val.as_str(), tokenizer),
                _ => {
                    let number = utils::string_to_num(&val, || tokenizer.err_msg_with_pos(pos))?;
                    Ok(Self::node(ParseToken::Number(number)))
                }
            },
            _ => Err(tokenizer.err_msg()),
        }
    }

    fn term_num_float(num: &str, tokenizer: &mut TokenReader) -> ParseResult<Node> {
        debug!("#term_num_float");
        Self::eat_token(tokenizer);
        match tokenizer.next_token() {
            Ok(Token::Key(pos, frac)) => {
                let mut f = String::new();
                f.push_str(num);
                f.push('.');
                f.push_str(frac.as_str());
                let number = utils::string_to_num(&f, || tokenizer.err_msg_with_pos(pos))?;
                Ok(Self::node(ParseToken::Number(number)))
            }
            _ => Err(tokenizer.err_msg()),
        }
    }

    fn term(tokenizer: &mut TokenReader) -> ParseResult<Node> {
        debug!("#term");

        match tokenizer.peek_token() {
            Ok(Token::At(_)) => {
                Self::eat_token(tokenizer);
                let node = Self::node(ParseToken::Relative);

                match tokenizer.peek_token() {
                    Ok(Token::Whitespace(_, _)) => {
                        Self::eat_whitespace(tokenizer);
                        Ok(node)
                    }
                    _ => Self::paths(node, tokenizer),
                }
            }
            Ok(Token::Absolute(_)) => {
                Self::json_path(tokenizer)
            }
            Ok(Token::DoubleQuoted(_, _)) | Ok(Token::SingleQuoted(_, _)) => {
                Self::array_quote_value(tokenizer)
            }
            Ok(Token::Key(_, key)) => {
                match key.as_bytes()[0] {
                    b'-' | b'0'..=b'9' => Self::term_num(tokenizer),
                    _ => Self::boolean(tokenizer),
                }
            }
            _ => {
                Err(tokenizer.err_msg())
            }
        }
    }

    fn op(prev: Node, tokenizer: &mut TokenReader) -> ParseResult<Node> {
        debug!("#op");
        let token = match tokenizer.next_token() {
            Ok(Token::Equal(_)) => ParseToken::Filter(FilterToken::Equal),
            Ok(Token::NotEqual(_)) => ParseToken::Filter(FilterToken::NotEqual),
            Ok(Token::Little(_)) => ParseToken::Filter(FilterToken::Little),
            Ok(Token::LittleOrEqual(_)) => ParseToken::Filter(FilterToken::LittleOrEqual),
            Ok(Token::Greater(_)) => ParseToken::Filter(FilterToken::Greater),
            Ok(Token::GreaterOrEqual(_)) => ParseToken::Filter(FilterToken::GreaterOrEqual),
            _ => {
                return Err(tokenizer.err_msg());
            }
        };

        Self::eat_whitespace(tokenizer);

        Ok(Node {
            token,
            left: Some(Box::new(prev)),
            right: Some(Box::new(Self::term(tokenizer)?)),
        })
    }

    fn eat_whitespace(tokenizer: &mut TokenReader) {
        while let Ok(Token::Whitespace(_, _)) = tokenizer.peek_token() {
            let _ = tokenizer.next_token();
        }
    }

    fn eat_token(tokenizer: &mut TokenReader) {
        let _ = tokenizer.next_token();
    }

    fn node(token: ParseToken) -> Node {
        Node {
            left: None,
            right: None,
            token,
        }
    }

    fn close_token(ret: Node, token: Token, tokenizer: &mut TokenReader) -> ParseResult<Node> {
        debug!("#close_token");
        match tokenizer.next_token() {
            Ok(ref t) if t.is_match_token_type(token) => Ok(ret),
            _ => Err(tokenizer.err_msg()),
        }
    }
}

#[allow(deprecated)]
pub trait NodeVisitor {
    fn visit(&mut self, node: &Node) {
        match &node.token {
            ParseToken::Absolute
            | ParseToken::Relative
            | ParseToken::All
            | ParseToken::Key(_)
            | ParseToken::Keys(_)
            | ParseToken::Range(_, _, _)
            | ParseToken::Union(_)
            | ParseToken::Number(_)
            | ParseToken::Bool(_) => {
                self.visit_token(&node.token);
            }
            ParseToken::In | ParseToken::Leaves => {
                if let Some(n) = &node.left {
                    self.visit(&*n);
                }

                self.visit_token(&node.token);

                if let Some(n) = &node.right {
                    self.visit(&*n);
                }
            }
            ParseToken::Array => {
                if let Some(n) = &node.left {
                    self.visit(&*n);
                }

                self.visit_token(&node.token);

                if let Some(n) = &node.right {
                    self.visit(&*n);
                }

                self.visit_token(&ParseToken::ArrayEof);
            }
            ParseToken::Filter(FilterToken::And) | ParseToken::Filter(FilterToken::Or) => {
                if let Some(n) = &node.left {
                    self.visit(&*n);
                }

                if let Some(n) = &node.right {
                    self.visit(&*n);
                }

                self.visit_token(&node.token);
            }
            ParseToken::Filter(_) => {
                if let Some(n) = &node.left {
                    self.visit(&*n);
                }

                self.end_term();

                if let Some(n) = &node.right {
                    self.visit(&*n);
                }

                self.end_term();

                self.visit_token(&node.token);
            }
            _ => {}
        }
    }

    fn visit_token(&mut self, token: &ParseToken);
    fn end_term(&mut self) {}
}

// #[cfg(test)]
// mod parser_tests {
//     use parser::{FilterToken, NodeVisitor, ParseToken, Parser};
//
//     struct NodeVisitorTestImpl<'a> {
//         input: &'a str,
//         stack: Vec<ParseToken>,
//     }
//
//     impl<'a> NodeVisitorTestImpl<'a> {
//         fn new(input: &'a str) -> Self {
//             NodeVisitorTestImpl {
//                 input,
//                 stack: Vec::new(),
//             }
//         }
//
//         fn start(&mut self) -> Result<Vec<ParseToken>, String> {
//             let node = Parser::compile(self.input)?;
//             self.visit(&node);
//             Ok(self.stack.split_off(0))
//         }
//     }
//
//     impl<'a> NodeVisitor for NodeVisitorTestImpl<'a> {
//         fn visit_token(&mut self, token: &ParseToken) {
//             self.stack.push(token.clone());
//         }
//     }
//
//     fn setup() {
//         let _ = env_logger::try_init();
//     }
//
//     fn run(input: &str) -> Result<Vec<ParseToken>, String> {
//         let mut interpreter = NodeVisitorTestImpl::new(input);
//         interpreter.start()
//     }
//
//     #[test]
//     fn parse_error() {
//         setup();
//
//         fn invalid(path: &str) {
//             assert!(run(path).is_err());
//         }
//
//         invalid("$[]");
//         invalid("$[a]");
//         invalid("$[?($.a)]");
//         invalid("$[?(@.a > @.b]");
//         invalid("$[?(@.a < @.b&&(@.c < @.d)]");
//         invalid("@.");
//         invalid("$..[?(a <= @.a)]"); // invalid term value
//         invalid("$['a', b]");
//         invalid("$[0, >=]");
//         invalid("$[a:]");
//         invalid("$[:a]");
//         invalid("$[::a]");
//         invalid("$[:>]");
//         invalid("$[1:>]");
//         invalid("$[1,,]");
//         invalid("$[?]");
//         invalid("$[?(1 = 1)]");
//         invalid("$[?(1 = >)]");
//     }
//
//     #[test]
//     fn parse_path() {
//         setup();
//
//         assert_eq!(
//             run("$.aa"),
//             Ok(vec![
//                 ParseToken::Absolute,
//                 ParseToken::In,
//                 ParseToken::Key("aa".to_owned())
//             ])
//         );
//
//         assert_eq!(
//             run("$.00.a"),
//             Ok(vec![
//                 ParseToken::Absolute,
//                 ParseToken::In,
//                 ParseToken::Key("00".to_owned()),
//                 ParseToken::In,
//                 ParseToken::Key("a".to_owned())
//             ])
//         );
//
//         assert_eq!(
//             run("$.00.韓창.seok"),
//             Ok(vec![
//                 ParseToken::Absolute,
//                 ParseToken::In,
//                 ParseToken::Key("00".to_owned()),
//                 ParseToken::In,
//                 ParseToken::Key("韓창".to_owned()),
//                 ParseToken::In,
//                 ParseToken::Key("seok".to_owned())
//             ])
//         );
//
//         assert_eq!(
//             run("$.*"),
//             Ok(vec![ParseToken::Absolute, ParseToken::In, ParseToken::All])
//         );
//
//         assert_eq!(
//             run("$..*"),
//             Ok(vec![
//                 ParseToken::Absolute,
//                 ParseToken::Leaves,
//                 ParseToken::All
//             ])
//         );
//
//         assert_eq!(
//             run("$..[0]"),
//             Ok(vec![
//                 ParseToken::Absolute,
//                 ParseToken::Leaves,
//                 ParseToken::Array,
//                 ParseToken::Number(0.0),
//                 ParseToken::ArrayEof
//             ])
//         );
//
//         assert_eq!(
//             run("$.$a"),
//             Ok(vec![
//                 ParseToken::Absolute,
//                 ParseToken::In,
//                 ParseToken::Key("$a".to_owned())
//             ])
//         );
//
//         assert_eq!(
//             run("$.['$a']"),
//             Ok(vec![
//                 ParseToken::Absolute,
//                 ParseToken::Array,
//                 ParseToken::Key("$a".to_owned()),
//                 ParseToken::ArrayEof,
//             ])
//         );
//
//         if run("$.").is_ok() {
//             panic!();
//         }
//
//         if run("$..").is_ok() {
//             panic!();
//         }
//
//         if run("$. a").is_ok() {
//             panic!();
//         }
//     }
//
//     #[test]
//     fn parse_array_syntax() {
//         setup();
//
//         assert_eq!(
//             run("$.book[?(@.isbn)]"),
//             Ok(vec![
//                 ParseToken::Absolute,
//                 ParseToken::In,
//                 ParseToken::Key("book".to_string()),
//                 ParseToken::Array,
//                 ParseToken::Relative,
//                 ParseToken::In,
//                 ParseToken::Key("isbn".to_string()),
//                 ParseToken::ArrayEof
//             ])
//         );
//
//         //
//         // Array도 컨텍스트 In으로 간주 할거라서 중첩되면 하나만
//         //
//         assert_eq!(
//             run("$.[*]"),
//             Ok(vec![
//                 ParseToken::Absolute,
//                 ParseToken::Array,
//                 ParseToken::All,
//                 ParseToken::ArrayEof
//             ])
//         );
//
//         assert_eq!(
//             run("$.a[*]"),
//             Ok(vec![
//                 ParseToken::Absolute,
//                 ParseToken::In,
//                 ParseToken::Key("a".to_owned()),
//                 ParseToken::Array,
//                 ParseToken::All,
//                 ParseToken::ArrayEof
//             ])
//         );
//
//         assert_eq!(
//             run("$.a[*].가"),
//             Ok(vec![
//                 ParseToken::Absolute,
//                 ParseToken::In,
//                 ParseToken::Key("a".to_owned()),
//                 ParseToken::Array,
//                 ParseToken::All,
//                 ParseToken::ArrayEof,
//                 ParseToken::In,
//                 ParseToken::Key("가".to_owned())
//             ])
//         );
//
//         assert_eq!(
//             run("$.a[0][1]"),
//             Ok(vec![
//                 ParseToken::Absolute,
//                 ParseToken::In,
//                 ParseToken::Key("a".to_owned()),
//                 ParseToken::Array,
//                 ParseToken::Number(0_f64),
//                 ParseToken::ArrayEof,
//                 ParseToken::Array,
//                 ParseToken::Number(1_f64),
//                 ParseToken::ArrayEof
//             ])
//         );
//
//         assert_eq!(
//             run("$.a[1,2]"),
//             Ok(vec![
//                 ParseToken::Absolute,
//                 ParseToken::In,
//                 ParseToken::Key("a".to_owned()),
//                 ParseToken::Array,
//                 ParseToken::Union(vec![1, 2]),
//                 ParseToken::ArrayEof
//             ])
//         );
//
//         assert_eq!(
//             run("$.a[10:]"),
//             Ok(vec![
//                 ParseToken::Absolute,
//                 ParseToken::In,
//                 ParseToken::Key("a".to_owned()),
//                 ParseToken::Array,
//                 ParseToken::Range(Some(10), None, None),
//                 ParseToken::ArrayEof
//             ])
//         );
//
//         assert_eq!(
//             run("$.a[:11]"),
//             Ok(vec![
//                 ParseToken::Absolute,
//                 ParseToken::In,
//                 ParseToken::Key("a".to_owned()),
//                 ParseToken::Array,
//                 ParseToken::Range(None, Some(11), None),
//                 ParseToken::ArrayEof
//             ])
//         );
//
//         assert_eq!(
//             run("$.a[-12:13]"),
//             Ok(vec![
//                 ParseToken::Absolute,
//                 ParseToken::In,
//                 ParseToken::Key("a".to_owned()),
//                 ParseToken::Array,
//                 ParseToken::Range(Some(-12), Some(13), None),
//                 ParseToken::ArrayEof
//             ])
//         );
//
//         assert_eq!(
//             run(r#"$[0:3:2]"#),
//             Ok(vec![
//                 ParseToken::Absolute,
//                 ParseToken::Array,
//                 ParseToken::Range(Some(0), Some(3), Some(2)),
//                 ParseToken::ArrayEof
//             ])
//         );
//
//         assert_eq!(
//             run(r#"$[:3:2]"#),
//             Ok(vec![
//                 ParseToken::Absolute,
//                 ParseToken::Array,
//                 ParseToken::Range(None, Some(3), Some(2)),
//                 ParseToken::ArrayEof
//             ])
//         );
//
//         assert_eq!(
//             run(r#"$[:]"#),
//             Ok(vec![
//                 ParseToken::Absolute,
//                 ParseToken::Array,
//                 ParseToken::Range(None, None, None),
//                 ParseToken::ArrayEof
//             ])
//         );
//
//         assert_eq!(
//             run(r#"$[::]"#),
//             Ok(vec![
//                 ParseToken::Absolute,
//                 ParseToken::Array,
//                 ParseToken::Range(None, None, None),
//                 ParseToken::ArrayEof
//             ])
//         );
//
//         assert_eq!(
//             run(r#"$[::2]"#),
//             Ok(vec![
//                 ParseToken::Absolute,
//                 ParseToken::Array,
//                 ParseToken::Range(None, None, Some(2)),
//                 ParseToken::ArrayEof
//             ])
//         );
//
//         assert_eq!(
//             run(r#"$["a", 'b']"#),
//             Ok(vec![
//                 ParseToken::Absolute,
//                 ParseToken::Array,
//                 ParseToken::Keys(vec!["a".to_string(), "b".to_string()]),
//                 ParseToken::ArrayEof
//             ])
//         );
//
//         assert_eq!(
//             run("$.a[?(1>2)]"),
//             Ok(vec![
//                 ParseToken::Absolute,
//                 ParseToken::In,
//                 ParseToken::Key("a".to_owned()),
//                 ParseToken::Array,
//                 ParseToken::Number(1_f64),
//                 ParseToken::Number(2_f64),
//                 ParseToken::Filter(FilterToken::Greater),
//                 ParseToken::ArrayEof
//             ])
//         );
//
//         assert_eq!(
//             run("$.a[?($.b>3)]"),
//             Ok(vec![
//                 ParseToken::Absolute,
//                 ParseToken::In,
//                 ParseToken::Key("a".to_owned()),
//                 ParseToken::Array,
//                 ParseToken::Absolute,
//                 ParseToken::In,
//                 ParseToken::Key("b".to_owned()),
//                 ParseToken::Number(3_f64),
//                 ParseToken::Filter(FilterToken::Greater),
//                 ParseToken::ArrayEof
//             ])
//         );
//
//         assert_eq!(
//             run("$[?($.c>@.d && 1==2)]"),
//             Ok(vec![
//                 ParseToken::Absolute,
//                 ParseToken::Array,
//                 ParseToken::Absolute,
//                 ParseToken::In,
//                 ParseToken::Key("c".to_owned()),
//                 ParseToken::Relative,
//                 ParseToken::In,
//                 ParseToken::Key("d".to_owned()),
//                 ParseToken::Filter(FilterToken::Greater),
//                 ParseToken::Number(1_f64),
//                 ParseToken::Number(2_f64),
//                 ParseToken::Filter(FilterToken::Equal),
//                 ParseToken::Filter(FilterToken::And),
//                 ParseToken::ArrayEof
//             ])
//         );
//
//         assert_eq!(
//             run("$[?($.c>@.d&&(1==2||3>=4))]"),
//             Ok(vec![
//                 ParseToken::Absolute,
//                 ParseToken::Array,
//                 ParseToken::Absolute,
//                 ParseToken::In,
//                 ParseToken::Key("c".to_owned()),
//                 ParseToken::Relative,
//                 ParseToken::In,
//                 ParseToken::Key("d".to_owned()),
//                 ParseToken::Filter(FilterToken::Greater),
//                 ParseToken::Number(1_f64),
//                 ParseToken::Number(2_f64),
//                 ParseToken::Filter(FilterToken::Equal),
//                 ParseToken::Number(3_f64),
//                 ParseToken::Number(4_f64),
//                 ParseToken::Filter(FilterToken::GreaterOrEqual),
//                 ParseToken::Filter(FilterToken::Or),
//                 ParseToken::Filter(FilterToken::And),
//                 ParseToken::ArrayEof
//             ])
//         );
//
//         assert_eq!(
//             run("$[?(@.a<@.b)]"),
//             Ok(vec![
//                 ParseToken::Absolute,
//                 ParseToken::Array,
//                 ParseToken::Relative,
//                 ParseToken::In,
//                 ParseToken::Key("a".to_owned()),
//                 ParseToken::Relative,
//                 ParseToken::In,
//                 ParseToken::Key("b".to_owned()),
//                 ParseToken::Filter(FilterToken::Little),
//                 ParseToken::ArrayEof
//             ])
//         );
//
//         assert_eq!(
//             run("$[*][*][*]"),
//             Ok(vec![
//                 ParseToken::Absolute,
//                 ParseToken::Array,
//                 ParseToken::All,
//                 ParseToken::ArrayEof,
//                 ParseToken::Array,
//                 ParseToken::All,
//                 ParseToken::ArrayEof,
//                 ParseToken::Array,
//                 ParseToken::All,
//                 ParseToken::ArrayEof
//             ])
//         );
//
//         assert_eq!(
//             run("$['a']['bb']"),
//             Ok(vec![
//                 ParseToken::Absolute,
//                 ParseToken::Array,
//                 ParseToken::Key("a".to_string()),
//                 ParseToken::ArrayEof,
//                 ParseToken::Array,
//                 ParseToken::Key("bb".to_string()),
//                 ParseToken::ArrayEof
//             ])
//         );
//
//         assert_eq!(
//             run("$.a[?(@.e==true)]"),
//             Ok(vec![
//                 ParseToken::Absolute,
//                 ParseToken::In,
//                 ParseToken::Key("a".to_string()),
//                 ParseToken::Array,
//                 ParseToken::Relative,
//                 ParseToken::In,
//                 ParseToken::Key("e".to_string()),
//                 ParseToken::Bool(true),
//                 ParseToken::Filter(FilterToken::Equal),
//                 ParseToken::ArrayEof
//             ])
//         );
//
//         assert_eq!(
//             run(r#"$[?(@ > 1)]"#),
//             Ok(vec![
//                 ParseToken::Absolute,
//                 ParseToken::Array,
//                 ParseToken::Relative,
//                 ParseToken::Number(1_f64),
//                 ParseToken::Filter(FilterToken::Greater),
//                 ParseToken::ArrayEof
//             ])
//         );
//
//         assert_eq!(
//             run("$[:]"),
//             Ok(vec![
//                 ParseToken::Absolute,
//                 ParseToken::Array,
//                 ParseToken::Range(None, None, None),
//                 ParseToken::ArrayEof
//             ])
//         );
//
//         assert_eq!(
//             run(r#"$['single\'quote']"#),
//             Ok(vec![
//                 ParseToken::Absolute,
//                 ParseToken::Array,
//                 ParseToken::Key("single'quote".to_string()),
//                 ParseToken::ArrayEof
//             ])
//         );
//
//         assert_eq!(
//             run(r#"$["single\"quote"]"#),
//             Ok(vec![
//                 ParseToken::Absolute,
//                 ParseToken::Array,
//                 ParseToken::Key(r#"single"quote"#.to_string()),
//                 ParseToken::ArrayEof
//             ])
//         );
//     }
//
//     #[test]
//     fn parse_array_float() {
//         setup();
//
//         assert_eq!(
//             run("$[?(1.1<2.1)]"),
//             Ok(vec![
//                 ParseToken::Absolute,
//                 ParseToken::Array,
//                 ParseToken::Number(1.1),
//                 ParseToken::Number(2.1),
//                 ParseToken::Filter(FilterToken::Little),
//                 ParseToken::ArrayEof
//             ])
//         );
//
//         if run("$[1.1]").is_ok() {
//             panic!();
//         }
//
//         if run("$[?(1.1<.2)]").is_ok() {
//             panic!();
//         }
//
//         if run("$[?(1.1<2.)]").is_ok() {
//             panic!();
//         }
//
//         if run("$[?(1.1<2.a)]").is_ok() {
//             panic!();
//         }
//     }
// }
//
// #[cfg(test)]
// mod tokenizer_tests {
//     use parser::tokenizer::{Token, TokenError, TokenReader, Tokenizer};
//
//     fn setup() {
//         let _ = env_logger::try_init();
//     }
//
//     fn collect_token(input: &str) -> (Vec<Token>, Option<TokenError>) {
//         let mut tokenizer = Tokenizer::new(input);
//         let mut vec = vec![];
//         loop {
//             match tokenizer.next_token() {
//                 Ok(t) => vec.push(t),
//                 Err(e) => return (vec, Some(e)),
//             }
//         }
//     }
//
//     fn run(input: &str, expected: (Vec<Token>, Option<TokenError>)) {
//         let (vec, err) = collect_token(input);
//         assert_eq!((vec, err), expected, "\"{}\"", input);
//     }
//
//     #[test]
//     fn peek() {
//         let mut tokenizer = TokenReader::new("$.a");
//         match tokenizer.next_token() {
//             Ok(t) => assert_eq!(Token::Absolute(0), t),
//             _ => panic!(),
//         }
//
//         match tokenizer.peek_token() {
//             Ok(t) => assert_eq!(&Token::Dot(1), t),
//             _ => panic!(),
//         }
//
//         match tokenizer.peek_token() {
//             Ok(t) => assert_eq!(&Token::Dot(1), t),
//             _ => panic!(),
//         }
//
//         match tokenizer.next_token() {
//             Ok(t) => assert_eq!(Token::Dot(1), t),
//             _ => panic!(),
//         }
//     }
//
//     #[test]
//     fn token() {
//         setup();
//
//         run(
//             "$.01.a",
//             (
//                 vec![
//                     Token::Absolute(0),
//                     Token::Dot(1),
//                     Token::Key(2, "01".to_string()),
//                     Token::Dot(4),
//                     Token::Key(5, "a".to_string()),
//                 ],
//                 Some(TokenError::Eof),
//             ),
//         );
//
//         run(
//             "$.   []",
//             (
//                 vec![
//                     Token::Absolute(0),
//                     Token::Dot(1),
//                     Token::Whitespace(2, 2),
//                     Token::OpenArray(5),
//                     Token::CloseArray(6),
//                 ],
//                 Some(TokenError::Eof),
//             ),
//         );
//
//         run(
//             "$..",
//             (
//                 vec![Token::Absolute(0), Token::Dot(1), Token::Dot(2)],
//                 Some(TokenError::Eof),
//             ),
//         );
//
//         run(
//             "$..ab",
//             (
//                 vec![
//                     Token::Absolute(0),
//                     Token::Dot(1),
//                     Token::Dot(2),
//                     Token::Key(3, "ab".to_string()),
//                 ],
//                 Some(TokenError::Eof),
//             ),
//         );
//
//         run(
//             "$..가 [",
//             (
//                 vec![
//                     Token::Absolute(0),
//                     Token::Dot(1),
//                     Token::Dot(2),
//                     Token::Key(3, "가".to_string()),
//                     Token::Whitespace(6, 0),
//                     Token::OpenArray(7),
//                 ],
//                 Some(TokenError::Eof),
//             ),
//         );
//
//         run(
//             "[-1, 2 ]",
//             (
//                 vec![
//                     Token::OpenArray(0),
//                     Token::Key(1, "-1".to_string()),
//                     Token::Comma(3),
//                     Token::Whitespace(4, 0),
//                     Token::Key(5, "2".to_string()),
//                     Token::Whitespace(6, 0),
//                     Token::CloseArray(7),
//                 ],
//                 Some(TokenError::Eof),
//             ),
//         );
//
//         run(
//             "[ 1 2 , 3 \"abc\" : -10 ]",
//             (
//                 vec![
//                     Token::OpenArray(0),
//                     Token::Whitespace(1, 0),
//                     Token::Key(2, "1".to_string()),
//                     Token::Whitespace(3, 0),
//                     Token::Key(4, "2".to_string()),
//                     Token::Whitespace(5, 0),
//                     Token::Comma(6),
//                     Token::Whitespace(7, 0),
//                     Token::Key(8, "3".to_string()),
//                     Token::Whitespace(9, 0),
//                     Token::DoubleQuoted(10, "abc".to_string()),
//                     Token::Whitespace(15, 0),
//                     Token::Split(16),
//                     Token::Whitespace(17, 0),
//                     Token::Key(18, "-10".to_string()),
//                     Token::Whitespace(21, 0),
//                     Token::CloseArray(22),
//                 ],
//                 Some(TokenError::Eof),
//             ),
//         );
//
//         run(
//             "?(@.a가 <41.01)",
//             (
//                 vec![
//                     Token::Question(0),
//                     Token::OpenParenthesis(1),
//                     Token::At(2),
//                     Token::Dot(3),
//                     Token::Key(4, "a가".to_string()),
//                     Token::Whitespace(8, 0),
//                     Token::Little(9),
//                     Token::Key(10, "41".to_string()),
//                     Token::Dot(12),
//                     Token::Key(13, "01".to_string()),
//                     Token::CloseParenthesis(15),
//                 ],
//                 Some(TokenError::Eof),
//             ),
//         );
//
//         run(
//             "?(@.a <4a.01)",
//             (
//                 vec![
//                     Token::Question(0),
//                     Token::OpenParenthesis(1),
//                     Token::At(2),
//                     Token::Dot(3),
//                     Token::Key(4, "a".to_string()),
//                     Token::Whitespace(5, 0),
//                     Token::Little(6),
//                     Token::Key(7, "4a".to_string()),
//                     Token::Dot(9),
//                     Token::Key(10, "01".to_string()),
//                     Token::CloseParenthesis(12),
//                 ],
//                 Some(TokenError::Eof),
//             ),
//         );
//
//         run(
//             "?($.c>@.d)",
//             (
//                 vec![
//                     Token::Question(0),
//                     Token::OpenParenthesis(1),
//                     Token::Absolute(2),
//                     Token::Dot(3),
//                     Token::Key(4, "c".to_string()),
//                     Token::Greater(5),
//                     Token::At(6),
//                     Token::Dot(7),
//                     Token::Key(8, "d".to_string()),
//                     Token::CloseParenthesis(9),
//                 ],
//                 Some(TokenError::Eof),
//             ),
//         );
//
//         run(
//             "$[:]",
//             (
//                 vec![
//                     Token::Absolute(0),
//                     Token::OpenArray(1),
//                     Token::Split(2),
//                     Token::CloseArray(3),
//                 ],
//                 Some(TokenError::Eof),
//             ),
//         );
//
//         run(
//             r#"$['single\'quote']"#,
//             (
//                 vec![
//                     Token::Absolute(0),
//                     Token::OpenArray(1),
//                     Token::SingleQuoted(2, "single\'quote".to_string()),
//                     Token::CloseArray(17),
//                 ],
//                 Some(TokenError::Eof),
//             ),
//         );
//
//         run(
//             r#"$['single\'1','single\'2']"#,
//             (
//                 vec![
//                     Token::Absolute(0),
//                     Token::OpenArray(1),
//                     Token::SingleQuoted(2, "single\'1".to_string()),
//                     Token::Comma(13),
//                     Token::SingleQuoted(14, "single\'2".to_string()),
//                     Token::CloseArray(25),
//                 ],
//                 Some(TokenError::Eof),
//             ),
//         );
//
//         run(
//             r#"$["double\"quote"]"#,
//             (
//                 vec![
//                     Token::Absolute(0),
//                     Token::OpenArray(1),
//                     Token::DoubleQuoted(2, "double\"quote".to_string()),
//                     Token::CloseArray(17),
//                 ],
//                 Some(TokenError::Eof),
//             ),
//         );
//     }
// }
