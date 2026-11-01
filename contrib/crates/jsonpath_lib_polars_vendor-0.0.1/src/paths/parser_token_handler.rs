use super::str_reader::StrRange;
use super::tokens::ParseToken;

pub trait ParserTokenHandler<'a> {
    fn handle<F>(&mut self, token: &ParseToken, parse_value_reader: &F)
        where
            F: Fn(&StrRange) -> &'a str;
}