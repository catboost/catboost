use std::str::FromStr;

use super::parser_token_handler::ParserTokenHandler;
use super::parser_node_visitor::ParserNodeVisitor;
use super::str_reader::StrRange;
use super::tokenizer::{TokenError, TokenReader};
use super::tokens::{FilterToken, ParseToken, Token};

#[derive(Clone, Debug)]
pub struct PathParser<'a> {
    parser: ParserImpl<'a>,
}

impl<'a> PathParser<'a> {
    pub fn compile(input: &'a str) -> Result<Self, TokenError> {
        let mut parser = ParserImpl::new(input);
        parser.compile()?;
        Ok(PathParser { parser })
    }

    pub(crate) fn parse<F>(&self, parse_token_handler: &mut F) -> Result<(), String>
        where
            F: ParserTokenHandler<'a>,
    {
        if self.parser.parse_node.is_none() {
            unreachable!()
        }

        let token_reader = &self.parser.token_reader;
        if let Some(parse_node) = self.parser.parse_node.as_ref() {
            self.visit(parse_node, parse_token_handler, &|s| {
                token_reader.read_value(s)
            });
        }

        Ok(())
    }
}

impl<'a> ParserNodeVisitor<'a> for PathParser<'a> {}

#[derive(Clone, Debug)]
struct ParserImpl<'a> {
    token_reader: TokenReader<'a>,
    parse_node: Option<ParserNode>,
}

impl<'a> ParserImpl<'a> {
    pub fn new(input: &'a str) -> Self {
        ParserImpl {
            token_reader: TokenReader::new(input),
            parse_node: None,
        }
    }

    fn string_to_num<F, S: FromStr>(string: &str, msg_handler: F) -> Result<S, TokenError>
        where
            F: Fn() -> TokenError,
    {
        match string.parse() {
            Ok(n) => Ok(n),
            _ => Err(msg_handler()),
        }
    }

    pub fn compile(&mut self) -> Result<&mut Self, TokenError> {
        self.parse_node = Some(self.json_path()?);
        Ok(self)
    }

    fn json_path(&mut self) -> Result<ParserNode, TokenError> {
        debug!("#json_path");
        match self.token_reader.next_token() {
            Ok(Token::Absolute(_)) => {
                let node = self.create_node(ParseToken::Absolute);
                self.paths(node)
            }
            _ => Err(self.token_reader.to_error()),
        }
    }

    fn paths(&mut self, prev: ParserNode) -> Result<ParserNode, TokenError> {
        debug!("#paths");
        match self.token_reader.peek_token() {
            Ok(Token::Dot(_)) => {
                self.eat_token();
                self.paths_dot(prev)
            }
            Ok(Token::OpenArray(_)) => {
                self.eat_token();
                self.eat_whitespace();
                let node = self.array(prev)?;
                self.paths(node)
            }
            _ => Ok(prev),
        }
    }

    fn paths_dot(&mut self, prev: ParserNode) -> Result<ParserNode, TokenError> {
        debug!("#paths_dot");
        let node = self.path(prev)?;
        self.paths(node)
    }

    fn path(&mut self, prev: ParserNode) -> Result<ParserNode, TokenError> {
        debug!("#path");
        match self.token_reader.peek_token() {
            Ok(Token::Dot(_)) => self.path_leaves(prev),
            Ok(Token::Asterisk(_)) => self.path_in_all(prev),
            Ok(Token::Key(_)) => self.path_in_key(prev),
            Ok(Token::OpenArray(_)) => {
                self.eat_token();
                self.array(prev)
            }
            _ => Err(self.token_reader.to_error()),
        }
    }

    fn path_leaves(&mut self, prev: ParserNode) -> Result<ParserNode, TokenError> {
        debug!("#path_leaves");
        self.eat_token();
        match self.token_reader.peek_token() {
            Ok(Token::Asterisk(_)) => self.path_leaves_all(prev),
            Ok(Token::OpenArray(_)) => {
                let mut leaves_node = self.create_node(ParseToken::Leaves);
                leaves_node.left = Some(Box::new(prev));
                Ok(self.paths(leaves_node)?)
            }
            _ => self.path_leaves_key(prev),
        }
    }

    #[allow(clippy::unnecessary_wraps)]
    fn path_leaves_key(&mut self, prev: ParserNode) -> Result<ParserNode, TokenError> {
        debug!("#path_leaves_key");
        Ok(ParserNode {
            token: ParseToken::Leaves,
            left: Some(Box::new(prev)),
            right: Some(Box::new(self.key()?)),
        })
    }

    #[allow(clippy::unnecessary_wraps)]
    fn path_leaves_all(&mut self, prev: ParserNode) -> Result<ParserNode, TokenError> {
        debug!("#path_leaves_all");
        self.eat_token();
        Ok(ParserNode {
            token: ParseToken::Leaves,
            left: Some(Box::new(prev)),
            right: Some(Box::new(self.create_node(ParseToken::All))),
        })
    }

    #[allow(clippy::unnecessary_wraps)]
    fn path_in_all(&mut self, prev: ParserNode) -> Result<ParserNode, TokenError> {
        debug!("#path_in_all");
        self.eat_token();
        Ok(ParserNode {
            token: ParseToken::In,
            left: Some(Box::new(prev)),
            right: Some(Box::new(self.create_node(ParseToken::All))),
        })
    }

    #[allow(clippy::unnecessary_wraps)]
    fn path_in_key(&mut self, prev: ParserNode) -> Result<ParserNode, TokenError> {
        debug!("#path_in_key");
        Ok(ParserNode {
            token: ParseToken::In,
            left: Some(Box::new(prev)),
            right: Some(Box::new(self.key()?)),
        })
    }

    fn key(&mut self) -> Result<ParserNode, TokenError> {
        debug!("#key");
        match self.token_reader.next_token() {
            Ok(Token::Key(s)) => Ok(self.create_node(ParseToken::Key(s))),
            _ => Err(self.token_reader.to_error()),
        }
    }

    fn boolean(&mut self) -> Result<ParserNode, TokenError> {
        debug!("#boolean");

        fn validation_bool_value(v: &str) -> bool {
            let b = v.as_bytes();
            !b.is_empty() && (b[0] == b't' || b[0] == b'T' || b[0] == b'f' || b[0] == b'F')
        }

        if let Ok(Token::Key(s)) = self.token_reader.next_token() {
            let v = self.token_reader.read_value(&s);
            if validation_bool_value(v) {
                return Ok(self.create_node(ParseToken::Bool(v.eq_ignore_ascii_case("true"))));
            }
        }

        Err(self.token_reader.to_error())
    }

    fn array_keys(&mut self, first_key: StrRange) -> Result<ParserNode, TokenError> {
        let mut keys = vec![first_key];

        while let Ok(Token::Comma(_)) = self.token_reader.peek_token() {
            self.eat_token();
            self.eat_whitespace();

            match self.token_reader.next_token() {
                Ok(Token::SingleQuoted(s)) | Ok(Token::DoubleQuoted(s)) => {
                    keys.push(s);
                }
                _ => return Err(self.token_reader.to_error()),
            }

            self.eat_whitespace();
        }

        Ok(self.create_node(ParseToken::Keys(keys)))
    }

    fn array_quote_value(&mut self) -> Result<ParserNode, TokenError> {
        debug!("#array_quote_value");
        let next = self.token_reader.next_token();
        match next {
            Ok(Token::SingleQuoted(s)) | Ok(Token::DoubleQuoted(s)) => {
                if let Ok(Token::Comma(_)) = self.token_reader.peek_token() {
                    self.array_keys(s)
                } else {
                    Ok(self.create_node(ParseToken::Key(s)))
                }
            }
            _ => Err(self.token_reader.to_error()),
        }
    }

    fn array_start(&mut self, prev: ParserNode) -> Result<ParserNode, TokenError> {
        debug!("#array_start");
        match self.token_reader.peek_token() {
            Ok(Token::Question(_)) => {
                self.eat_token();
                Ok(ParserNode {
                    token: ParseToken::Array,
                    left: Some(Box::new(prev)),
                    right: Some(Box::new(self.filter()?)),
                })
            }
            Ok(Token::Asterisk(_)) => {
                self.eat_token();
                Ok(ParserNode {
                    token: ParseToken::Array,
                    left: Some(Box::new(prev)),
                    right: Some(Box::new(self.create_node(ParseToken::All))),
                })
            }
            _ => Ok(ParserNode {
                token: ParseToken::Array,
                left: Some(Box::new(prev)),
                right: Some(Box::new(self.array_value()?)),
            }),
        }
    }

    fn array(&mut self, prev: ParserNode) -> Result<ParserNode, TokenError> {
        debug!("#array");
        let ret = self.array_start(prev)?;
        self.eat_whitespace();
        self.close_token(ret, Token::CloseArray(StrRange::new(0, 0)))
    }

    fn array_value_key(&mut self) -> Result<ParserNode, TokenError> {
        debug!("#array_value_key");

        if let Ok(Token::Key(s)) = self.token_reader.next_token() {
            let val = self.token_reader.read_value(&s);
            let digit = Self::string_to_num(val, || self.token_reader.to_error())?;
            self.eat_whitespace();

            match self.token_reader.peek_token() {
                Ok(Token::Comma(_)) => self.union(digit),
                Ok(Token::Split(_)) => self.range_from(digit),
                _ => Ok(self.create_node(ParseToken::Number(digit as f64))),
            }
        } else {
            Err(self.token_reader.to_error())
        }
    }

    fn array_value(&mut self) -> Result<ParserNode, TokenError> {
        debug!("#array_value");
        match self.token_reader.peek_token() {
            Ok(Token::Key(_)) => self.array_value_key(),
            Ok(Token::Split(_)) => {
                self.eat_token();
                self.range_to()
            }
            Ok(Token::DoubleQuoted(_)) | Ok(Token::SingleQuoted(_)) => {
                self.array_quote_value()
            }
            Err(TokenError::Eof) => Ok(self.create_node(ParseToken::Eof)),
            _ => {
                self.eat_token();
                Err(self.token_reader.to_error())
            }
        }
    }

    fn union(&mut self, num: isize) -> Result<ParserNode, TokenError> {
        debug!("#union");
        let mut values = vec![num];
        while matches!(self.token_reader.peek_token(), Ok(Token::Comma(_))) {
            self.eat_token();
            self.eat_whitespace();

            match self.token_reader.next_token() {
                Ok(Token::Key(s)) => {
                    let val = self.token_reader.read_value(&s);
                    let digit = Self::string_to_num(val, || self.token_reader.to_error())?;
                    values.push(digit);
                }
                _ => {
                    return Err(self.token_reader.to_error());
                }
            }
        }
        Ok(self.create_node(ParseToken::Union(values)))
    }

    fn range_value<S: FromStr>(&mut self) -> Result<Option<S>, TokenError> {
        self.eat_whitespace();

        match self.token_reader.peek_token() {
            Ok(Token::Split(_)) => {
                self.eat_token();
                self.eat_whitespace();
            }
            _ => {
                return Ok(None);
            }
        }

        match self.token_reader.peek_token() {
            Ok(Token::Key(_)) => {}
            _ => {
                return Ok(None);
            }
        }

        match self.token_reader.next_token() {
            Ok(Token::Key(s)) => {
                let str_step = self.token_reader.read_value(&s);
                match Self::string_to_num(str_step, || self.token_reader.to_error()) {
                    Ok(step) => Ok(Some(step)),
                    Err(e) => Err(e),
                }
            }
            _ => {
                unreachable!();
            }
        }
    }

    fn range_from(&mut self, from: isize) -> Result<ParserNode, TokenError> {
        debug!("#range_from");
        self.eat_token();
        self.eat_whitespace();

        match self.token_reader.peek_token() {
            Ok(Token::Key(_)) => self.range(from),
            Ok(Token::Split(_)) => match self.range_value()? {
                Some(step) => Ok(self.create_node(ParseToken::Range(Some(from), None, Some(step)))),
                _ => Ok(self.create_node(ParseToken::Range(Some(from), None, None))),
            },
            _ => Ok(self.create_node(ParseToken::Range(Some(from), None, None))),
        }
    }

    fn range_to(&mut self) -> Result<ParserNode, TokenError> {
        debug!("#range_to");

        if let Some(step) = self.range_value()? {
            return Ok(self.create_node(ParseToken::Range(None, None, Some(step))));
        }

        if let Ok(Token::CloseArray(_)) = self.token_reader.peek_token() {
            return Ok(self.create_node(ParseToken::Range(None, None, None)));
        }

        match self.token_reader.next_token() {
            Ok(Token::Key(s)) => {
                let to_str = self.token_reader.read_value(&s);
                let to = Self::string_to_num(to_str, || self.token_reader.to_error())?;
                let step = self.range_value()?;
                Ok(self.create_node(ParseToken::Range(None, Some(to), step)))
            }
            _ => Err(self.token_reader.to_error()),
        }
    }

    fn range(&mut self, from: isize) -> Result<ParserNode, TokenError> {
        debug!("#range");
        match self.token_reader.next_token() {
            Ok(Token::Key(s)) => {
                let str_to = self.token_reader.read_value(&s);
                let to = Self::string_to_num(str_to, || self.token_reader.to_error())?;
                let step = self.range_value()?;
                Ok(self.create_node(ParseToken::Range(Some(from), Some(to), step)))
            }
            _ => Err(self.token_reader.to_error()),
        }
    }

    fn filter(&mut self) -> Result<ParserNode, TokenError> {
        debug!("#filter");
        match self.token_reader.next_token() {
            Ok(Token::OpenParenthesis(_)) => {
                let ret = self.exprs()?;
                self.eat_whitespace();
                self.close_token(ret, Token::CloseParenthesis(StrRange::new(0, 0)))
            }
            _ => Err(self.token_reader.to_error()),
        }
    }

    fn exprs(&mut self) -> Result<ParserNode, TokenError> {
        self.eat_whitespace();
        debug!("#exprs");
        let node = match self.token_reader.peek_token() {
            Ok(Token::OpenParenthesis(_)) => {
                self.eat_token();
                trace!("\t-exprs - open_parenthesis");
                let ret = self.exprs()?;
                self.eat_whitespace();
                self.close_token(ret, Token::CloseParenthesis(StrRange::new(0, 0)))?
            }
            _ => {
                trace!("\t-exprs - else");
                self.expr()?
            }
        };
        self.eat_whitespace();
        self.condition_expr(node)
    }

    fn condition_expr(&mut self, prev: ParserNode) -> Result<ParserNode, TokenError> {
        debug!("#condition_expr");
        match self.token_reader.peek_token() {
            Ok(Token::And(_)) => {
                self.eat_token();
                Ok(ParserNode {
                    token: ParseToken::Filter(FilterToken::And),
                    left: Some(Box::new(prev)),
                    right: Some(Box::new(self.exprs()?)),
                })
            }
            Ok(Token::Or(_)) => {
                self.eat_token();
                Ok(ParserNode {
                    token: ParseToken::Filter(FilterToken::Or),
                    left: Some(Box::new(prev)),
                    right: Some(Box::new(self.exprs()?)),
                })
            }
            _ => Ok(prev),
        }
    }

    fn expr(&mut self) -> Result<ParserNode, TokenError> {
        debug!("#expr");

        let has_prop_candidate = matches!(self.token_reader.peek_token(), Ok(Token::At(_)));

        let node = self.term()?;
        self.eat_whitespace();

        if matches!(self.token_reader.peek_token(),
            Ok(Token::Equal(_))
            | Ok(Token::NotEqual(_))
            | Ok(Token::Little(_))
            | Ok(Token::LittleOrEqual(_))
            | Ok(Token::Greater(_))
            | Ok(Token::GreaterOrEqual(_)))
        {
            self.op(node)
        } else if has_prop_candidate {
            Ok(node)
        } else {
            Err(self.token_reader.to_error())
        }
    }

    fn term_num(&mut self) -> Result<ParserNode, TokenError> {
        debug!("#term_num");
        match self.token_reader.next_token() {
            Ok(Token::Key(s)) => {
                let val = self.token_reader.read_value(&s);
                match self.token_reader.peek_token() {
                    Ok(Token::Dot(_)) => self.term_num_float(val),
                    _ => {
                        let number = Self::string_to_num(val, || self.token_reader.to_error())?;
                        Ok(self.create_node(ParseToken::Number(number)))
                    }
                }
            }
            _ => Err(self.token_reader.to_error()),
        }
    }

    fn term_num_float(&mut self, num: &'a str) -> Result<ParserNode, TokenError> {
        debug!("#term_num_float");
        self.eat_token();
        match self.token_reader.next_token() {
            Ok(Token::Key(s)) => {
                let frac = self.token_reader.read_value(&s);
                let number = Self::string_to_num(&[num, ".", frac].concat(), || self.token_reader.to_error())?;
                Ok(self.create_node(ParseToken::Number(number)))
            }
            _ => Err(self.token_reader.to_error()),
        }
    }

    fn term(&mut self) -> Result<ParserNode, TokenError> {
        debug!("#term");

        if self.token_reader.peek_token().is_err() {
            return Err(self.token_reader.to_error());
        }

        let has_term_key = if let Ok(Token::Key(s)) = self.token_reader.peek_token() {
            Some(s.clone())
        } else {
            None
        };

        if let Some(s) = has_term_key {
            let key = self.token_reader.read_value(&s);
            return match key.as_bytes()[0] {
                b'-' | b'0'..=b'9' => self.term_num(),
                _ => self.boolean(),
            };
        }

        match self.token_reader.peek_token() {
            Ok(Token::At(_)) => {
                self.eat_token();

                let node = self.create_node(ParseToken::Relative);
                match self.token_reader.peek_token() {
                    Ok(Token::Whitespace(_)) => {
                        self.eat_whitespace();
                        Ok(node)
                    }
                    _ => self.paths(node),
                }
            }
            Ok(Token::Absolute(_)) => {
                self.json_path()
            }
            Ok(Token::DoubleQuoted(_)) | Ok(Token::SingleQuoted(_)) => {
                self.array_quote_value()
            }
            _ => {
                Err(self.token_reader.to_error())
            }
        }
    }

    fn op(&mut self, prev: ParserNode) -> Result<ParserNode, TokenError> {
        debug!("#op");
        let token = match self.token_reader.next_token() {
            Ok(Token::Equal(_)) => ParseToken::Filter(FilterToken::Equal),
            Ok(Token::NotEqual(_)) => ParseToken::Filter(FilterToken::NotEqual),
            Ok(Token::Little(_)) => ParseToken::Filter(FilterToken::Little),
            Ok(Token::LittleOrEqual(_)) => ParseToken::Filter(FilterToken::LittleOrEqual),
            Ok(Token::Greater(_)) => ParseToken::Filter(FilterToken::Greater),
            Ok(Token::GreaterOrEqual(_)) => ParseToken::Filter(FilterToken::GreaterOrEqual),
            _ => {
                return Err(self.token_reader.to_error());
            }
        };

        self.eat_whitespace();

        Ok(ParserNode {
            token,
            left: Some(Box::new(prev)),
            right: Some(Box::new(self.term()?)),
        })
    }

    fn eat_whitespace(&mut self) {
        while let Ok(Token::Whitespace(_)) = self.token_reader.peek_token() {
            let _ = self.token_reader.next_token();
        }
    }

    fn eat_token(&mut self) {
        let _ = self.token_reader.next_token();
    }

    fn close_token(&mut self, ret: ParserNode, token: Token) -> Result<ParserNode, TokenError> {
        debug!("#close_token");
        match self.token_reader.next_token() {
            Ok(ref t) if t.is_match_token_type(token) => Ok(ret),
            _ => Err(self.token_reader.to_error()),
        }
    }

    fn create_node(&mut self, token: ParseToken) -> ParserNode {
        ParserNode {
            left: None,
            right: None,
            token,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ParserNode {
    pub left: Option<Box<ParserNode>>,
    pub right: Option<Box<ParserNode>>,
    pub token: ParseToken,
}

#[cfg(test)]
mod path_parser_tests {
    use paths::ParserTokenHandler;
    use paths::path_parser::PathParser;
    use paths::str_reader::StrRange;
    use paths::tokens::{FilterToken, ParseToken};

    struct NodeVisitorTestImpl<'a> {
        input: &'a str,
        stack: Vec<ParseToken>,
    }

    impl<'a> NodeVisitorTestImpl<'a> {
        fn new(input: &'a str) -> Self {
            NodeVisitorTestImpl {
                input,
                stack: Vec::new(),
            }
        }

        fn start(&mut self) -> Result<Vec<ParseToken>, String> {
            let parser = PathParser::compile(self.input).map_err(|_| "Token Error")?;
            let _ = parser.parse(self);
            Ok(self.stack.split_off(0))
        }
    }

    impl<'a> ParserTokenHandler<'a> for NodeVisitorTestImpl<'a> {
        fn handle<F>(&mut self, token: &ParseToken, _: &F)
            where
                F: Fn(&StrRange) -> &'a str
        {
            trace!("handle {:?}", token);
            self.stack.push(token.clone());
        }
    }

    fn setup() {
        let _ = env_logger::try_init();
    }

    fn run(input: &str) -> Result<Vec<ParseToken>, String> {
        let mut interpreter = NodeVisitorTestImpl::new(input);
        interpreter.start()
    }

    #[test]
    fn parse_error() {
        setup();

        fn invalid(path: &str) {
            assert!(run(path).is_err());
        }

        invalid("$[]");
        invalid("$[a]");
        invalid("$[?($.a)]");
        invalid("$[?(@.a > @.b]");
        invalid("$[?(@.a < @.b&&(@.c < @.d)]");
        invalid("@.");
        invalid("$..[?(a <= @.a)]"); // invalid term value
        invalid("$['a', b]");
        invalid("$[0, >=]");
        invalid("$[a:]");
        invalid("$[:a]");
        invalid("$[::a]");
        invalid("$[:>]");
        invalid("$[1:>]");
        invalid("$[1,,]");
        invalid("$[?]");
        invalid("$[?(1 = 1)]");
        invalid("$[?(1 = >)]");
    }

    #[test]
    fn parse_path() {
        setup();

        assert_eq!(
            run("$.aa"),
            Ok(vec![
                ParseToken::Absolute,
                ParseToken::In,
                ParseToken::Key(StrRange::new(2, "aa".len()))
            ])
        );

        assert_eq!(
            run("$.00.a"),
            Ok(vec![
                ParseToken::Absolute,
                ParseToken::In,
                ParseToken::Key(StrRange::new(2, "00".len())),
                ParseToken::In,
                ParseToken::Key(StrRange::new(5, "a".len()))
            ])
        );

        assert_eq!(
            run("$.00.韓창.seok"),
            Ok(vec![
                ParseToken::Absolute,
                ParseToken::In,
                ParseToken::Key(StrRange::new(2, "00".len())),
                ParseToken::In,
                ParseToken::Key(StrRange::new(5, "韓창".chars().map(|c| c.len_utf8()).sum())),
                ParseToken::In,
                ParseToken::Key(StrRange::new(12, "seok".len()))
            ])
        );

        assert_eq!(
            run("$.*"),
            Ok(vec![ParseToken::Absolute, ParseToken::In, ParseToken::All])
        );

        assert_eq!(
            run("$..*"),
            Ok(vec![
                ParseToken::Absolute,
                ParseToken::Leaves,
                ParseToken::All
            ])
        );

        assert_eq!(
            run("$..[0]"),
            Ok(vec![
                ParseToken::Absolute,
                ParseToken::Leaves,
                ParseToken::Array,
                ParseToken::Number(0.0),
                ParseToken::ArrayEof
            ])
        );

        assert_eq!(
            run("$.$a"),
            Ok(vec![
                ParseToken::Absolute,
                ParseToken::In,
                ParseToken::Key(StrRange::new(2, "$a".len()))
            ])
        );

        assert_eq!(
            run("$.['$a']"),
            Ok(vec![
                ParseToken::Absolute,
                ParseToken::Array,
                ParseToken::Key(StrRange::new(3, "'$a'".len())),
                ParseToken::ArrayEof,
            ])
        );

        if run("$.").is_ok() {
            panic!();
        }

        if run("$..").is_ok() {
            panic!();
        }

        if run("$. a").is_ok() {
            panic!();
        }
    }

    #[test]
    fn parse_array_syntax() {
        setup();

        assert_eq!(
            run("$.book[?(@.isbn)]"),
            Ok(vec![
                ParseToken::Absolute,
                ParseToken::In,
                ParseToken::Key(StrRange::new(2, "book".len())),
                ParseToken::Array,
                ParseToken::Relative,
                ParseToken::In,
                ParseToken::Key(StrRange::new(11, "isbn".len())),
                ParseToken::ArrayEof
            ])
        );

        //
        // Array도 컨텍스트 In으로 간주 할거라서 중첩되면 하나만
        //
        assert_eq!(
            run("$.[*]"),
            Ok(vec![
                ParseToken::Absolute,
                ParseToken::Array,
                ParseToken::All,
                ParseToken::ArrayEof
            ])
        );

        assert_eq!(
            run("$.a[*]"),
            Ok(vec![
                ParseToken::Absolute,
                ParseToken::In,
                ParseToken::Key(StrRange::new(2, "a".len())),
                ParseToken::Array,
                ParseToken::All,
                ParseToken::ArrayEof
            ])
        );

        assert_eq!(
            run("$.a[*].가"),
            Ok(vec![
                ParseToken::Absolute,
                ParseToken::In,
                ParseToken::Key(StrRange::new(2, "a".len())),
                ParseToken::Array,
                ParseToken::All,
                ParseToken::ArrayEof,
                ParseToken::In,
                ParseToken::Key(StrRange::new(7, '가'.len_utf8()))
            ])
        );

        assert_eq!(
            run("$.a[0][1]"),
            Ok(vec![
                ParseToken::Absolute,
                ParseToken::In,
                ParseToken::Key(StrRange::new(2, "a".len())),
                ParseToken::Array,
                ParseToken::Number(0_f64),
                ParseToken::ArrayEof,
                ParseToken::Array,
                ParseToken::Number(1_f64),
                ParseToken::ArrayEof
            ])
        );

        assert_eq!(
            run("$.a[1,2]"),
            Ok(vec![
                ParseToken::Absolute,
                ParseToken::In,
                ParseToken::Key(StrRange::new(2, "a".len())),
                ParseToken::Array,
                ParseToken::Union(vec![1, 2]),
                ParseToken::ArrayEof
            ])
        );

        assert_eq!(
            run("$.a[10:]"),
            Ok(vec![
                ParseToken::Absolute,
                ParseToken::In,
                ParseToken::Key(StrRange::new(2, "a".len())),
                ParseToken::Array,
                ParseToken::Range(Some(10), None, None),
                ParseToken::ArrayEof
            ])
        );

        assert_eq!(
            run("$.a[:11]"),
            Ok(vec![
                ParseToken::Absolute,
                ParseToken::In,
                ParseToken::Key(StrRange::new(2, "a".len())),
                ParseToken::Array,
                ParseToken::Range(None, Some(11), None),
                ParseToken::ArrayEof
            ])
        );

        assert_eq!(
            run("$.a[-12:13]"),
            Ok(vec![
                ParseToken::Absolute,
                ParseToken::In,
                ParseToken::Key(StrRange::new(2, "a".len())),
                ParseToken::Array,
                ParseToken::Range(Some(-12), Some(13), None),
                ParseToken::ArrayEof
            ])
        );

        assert_eq!(
            run(r#"$[0:3:2]"#),
            Ok(vec![
                ParseToken::Absolute,
                ParseToken::Array,
                ParseToken::Range(Some(0), Some(3), Some(2)),
                ParseToken::ArrayEof
            ])
        );

        assert_eq!(
            run(r#"$[:3:2]"#),
            Ok(vec![
                ParseToken::Absolute,
                ParseToken::Array,
                ParseToken::Range(None, Some(3), Some(2)),
                ParseToken::ArrayEof
            ])
        );

        assert_eq!(
            run(r#"$[:]"#),
            Ok(vec![
                ParseToken::Absolute,
                ParseToken::Array,
                ParseToken::Range(None, None, None),
                ParseToken::ArrayEof
            ])
        );

        assert_eq!(
            run(r#"$[::]"#),
            Ok(vec![
                ParseToken::Absolute,
                ParseToken::Array,
                ParseToken::Range(None, None, None),
                ParseToken::ArrayEof
            ])
        );

        assert_eq!(
            run(r#"$[::2]"#),
            Ok(vec![
                ParseToken::Absolute,
                ParseToken::Array,
                ParseToken::Range(None, None, Some(2)),
                ParseToken::ArrayEof
            ])
        );

        assert_eq!(
            run(r#"$["a", 'b']"#),
            Ok(vec![
                ParseToken::Absolute,
                ParseToken::Array,
                ParseToken::Keys(vec![StrRange::new(2, "\"a\"".len()), StrRange::new(7, "'b'".len())]),
                ParseToken::ArrayEof
            ])
        );

        assert_eq!(
            run("$.a[?(1>2)]"),
            Ok(vec![
                ParseToken::Absolute,
                ParseToken::In,
                ParseToken::Key(StrRange::new(2, "a".len())),
                ParseToken::Array,
                ParseToken::Number(1_f64),
                ParseToken::Number(2_f64),
                ParseToken::Filter(FilterToken::Greater),
                ParseToken::ArrayEof
            ])
        );

        assert_eq!(
            run("$.a[?($.b>3)]"),
            Ok(vec![
                ParseToken::Absolute,
                ParseToken::In,
                ParseToken::Key(StrRange::new(2, "a".len())),
                ParseToken::Array,
                ParseToken::Absolute,
                ParseToken::In,
                ParseToken::Key(StrRange::new(8, "b".len())),
                ParseToken::Number(3_f64),
                ParseToken::Filter(FilterToken::Greater),
                ParseToken::ArrayEof
            ])
        );

        assert_eq!(
            run("$[?($.c>@.d && 1==2)]"),
            Ok(vec![
                ParseToken::Absolute,
                ParseToken::Array,
                ParseToken::Absolute,
                ParseToken::In,
                ParseToken::Key(StrRange::new(6, "c".len())),
                ParseToken::Relative,
                ParseToken::In,
                ParseToken::Key(StrRange::new(10, "c".len())),
                ParseToken::Filter(FilterToken::Greater),
                ParseToken::Number(1_f64),
                ParseToken::Number(2_f64),
                ParseToken::Filter(FilterToken::Equal),
                ParseToken::Filter(FilterToken::And),
                ParseToken::ArrayEof
            ])
        );

        assert_eq!(
            run("$[?($.c>@.d&&(1==2||3>=4))]"),
            Ok(vec![
                ParseToken::Absolute,
                ParseToken::Array,
                ParseToken::Absolute,
                ParseToken::In,
                ParseToken::Key(StrRange::new(6, "c".len())),
                ParseToken::Relative,
                ParseToken::In,
                ParseToken::Key(StrRange::new(10, "d".len())),
                ParseToken::Filter(FilterToken::Greater),
                ParseToken::Number(1_f64),
                ParseToken::Number(2_f64),
                ParseToken::Filter(FilterToken::Equal),
                ParseToken::Number(3_f64),
                ParseToken::Number(4_f64),
                ParseToken::Filter(FilterToken::GreaterOrEqual),
                ParseToken::Filter(FilterToken::Or),
                ParseToken::Filter(FilterToken::And),
                ParseToken::ArrayEof
            ])
        );

        assert_eq!(
            run("$[?(@.a<@.b)]"),
            Ok(vec![
                ParseToken::Absolute,
                ParseToken::Array,
                ParseToken::Relative,
                ParseToken::In,
                ParseToken::Key(StrRange::new(6, "a".len())),
                ParseToken::Relative,
                ParseToken::In,
                ParseToken::Key(StrRange::new(10, "b".len())),
                ParseToken::Filter(FilterToken::Little),
                ParseToken::ArrayEof
            ])
        );

        assert_eq!(
            run("$[*][*][*]"),
            Ok(vec![
                ParseToken::Absolute,
                ParseToken::Array,
                ParseToken::All,
                ParseToken::ArrayEof,
                ParseToken::Array,
                ParseToken::All,
                ParseToken::ArrayEof,
                ParseToken::Array,
                ParseToken::All,
                ParseToken::ArrayEof
            ])
        );

        assert_eq!(
            run("$['a']['bb']"),
            Ok(vec![
                ParseToken::Absolute,
                ParseToken::Array,
                ParseToken::Key(StrRange::new(2, "'a'".len())),
                ParseToken::ArrayEof,
                ParseToken::Array,
                ParseToken::Key(StrRange::new(7, "'bb'".len())),
                ParseToken::ArrayEof
            ])
        );

        assert_eq!(
            run("$.a[?(@.e==true)]"),
            Ok(vec![
                ParseToken::Absolute,
                ParseToken::In,
                ParseToken::Key(StrRange::new(2, "a".len())),
                ParseToken::Array,
                ParseToken::Relative,
                ParseToken::In,
                ParseToken::Key(StrRange::new(8, "e".len())),
                ParseToken::Bool(true),
                ParseToken::Filter(FilterToken::Equal),
                ParseToken::ArrayEof
            ])
        );

        assert_eq!(
            run(r#"$[?(@ > 1)]"#),
            Ok(vec![
                ParseToken::Absolute,
                ParseToken::Array,
                ParseToken::Relative,
                ParseToken::Number(1_f64),
                ParseToken::Filter(FilterToken::Greater),
                ParseToken::ArrayEof
            ])
        );

        assert_eq!(
            run("$[:]"),
            Ok(vec![
                ParseToken::Absolute,
                ParseToken::Array,
                ParseToken::Range(None, None, None),
                ParseToken::ArrayEof
            ])
        );

        assert_eq!(
            run(r#"$['single\'quote']"#),
            Ok(vec![
                ParseToken::Absolute,
                ParseToken::Array,
                ParseToken::Key(StrRange::new(2, r#"'single\'quote'"#.len())),
                ParseToken::ArrayEof
            ])
        );

        assert_eq!(
            run(r#"$["single\"quote"]"#),
            Ok(vec![
                ParseToken::Absolute,
                ParseToken::Array,
                ParseToken::Key(StrRange::new(2, r#""single\"quote""#.len())),
                ParseToken::ArrayEof
            ])
        );
    }

    #[test]
    fn parse_array_float() {
        setup();

        assert_eq!(
            run("$[?(1.1<2.1)]"),
            Ok(vec![
                ParseToken::Absolute,
                ParseToken::Array,
                ParseToken::Number(1.1),
                ParseToken::Number(2.1),
                ParseToken::Filter(FilterToken::Little),
                ParseToken::ArrayEof
            ])
        );

        if run("$[1.1]").is_ok() {
            panic!();
        }

        if run("$[?(1.1<.2)]").is_ok() {
            panic!();
        }

        if run("$[?(1.1<2.)]").is_ok() {
            panic!();
        }

        if run("$[?(1.1<2.a)]").is_ok() {
            panic!();
        }
    }
}
