use std::collections::HashSet;
use std::rc::Rc;

use serde_json::{Number, Value};
use serde_json::map::Entry;

use JsonPathError;
use paths::{ParserTokenHandler, PathParser, StrRange, tokens::*};
use super::utils;

use super::terms::*;

#[derive(Debug, Default)]
pub struct JsonSelector<'a> {
    parser: Option<Rc<PathParser<'a>>>,
    value: Option<&'a Value>,
    tokens: Vec<ParseToken>,
    current: Option<Vec<&'a Value>>,
    selectors: Vec<JsonSelector<'a>>,
    selector_filter: FilterTerms<'a>,
}

impl<'a> JsonSelector<'a> {
    pub fn new(parser: PathParser<'a>) -> Self {
        JsonSelector {
            parser: Some(Rc::new(parser)),
            value: None,
            tokens: Vec::new(),
            current: None,
            selectors: Vec::new(),
            selector_filter: FilterTerms(Vec::new()),
        }
    }

    pub fn new_ref(parser: Rc<PathParser<'a>>) -> Self {
        JsonSelector {
            parser: Some(parser),
            value: None,
            tokens: Vec::new(),
            current: None,
            selectors: Vec::new(),
            selector_filter: FilterTerms(Vec::new()),
        }
    }

    pub fn reset_parser(&mut self, parser: PathParser<'a>) -> &mut Self {
        self.parser = Some(Rc::new(parser));
        self
    }

    pub fn reset_parser_ref(&mut self, parser: Rc<PathParser<'a>>) -> &mut Self {
        self.parser = Some(parser);
        self
    }

    pub fn reset_value(&mut self) -> &mut Self {
        self.current = None;
        self
    }

    pub fn value(&mut self, v: &'a Value) -> &mut Self {
        self.value = Some(v);
        self
    }

    fn _select(&mut self) -> Result<(), JsonPathError> {
        let parser = self.parser.take();
        if let Some(parser) = parser.as_ref() {
            let _ = parser.parse(self);
        }
        self.parser = parser;

        Ok(())
    }

    pub fn select_as<T: serde::de::DeserializeOwned>(&mut self) -> Result<Vec<T>, JsonPathError> {
        self._select()?;

        match &self.current {
            Some(vec) => {
                let mut ret = Vec::new();
                for v in vec {
                    match T::deserialize(*v) {
                        Ok(v) => ret.push(v),
                        Err(e) => return Err(JsonPathError::Serde(e.to_string())),
                    }
                }
                Ok(ret)
            }
            _ => Err(JsonPathError::EmptyValue),
        }
    }

    pub fn select_as_str(&mut self) -> Result<String, JsonPathError> {
        self._select()?;

        match &self.current {
            Some(r) => {
                Ok(serde_json::to_string(r).map_err(|e| JsonPathError::Serde(e.to_string()))?)
            }
            _ => Err(JsonPathError::EmptyValue),
        }
    }

    pub fn select(&mut self) -> Result<Vec<&'a Value>, JsonPathError> {
        self._select()?;

        match &self.current {
            Some(r) => Ok(r.to_vec()),
            _ => Err(JsonPathError::EmptyValue),
        }
    }

    fn compute_absolute_path_filter<F>(&mut self, token: &ParseToken, parse_value_reader: &F) -> bool
        where
            F: Fn(&StrRange) -> &'a str
    {
        if !self.selectors.is_empty() {
            match token {
                ParseToken::Absolute | ParseToken::Relative | ParseToken::Filter(_) => {
                    let selector = self.selectors.pop().unwrap();

                    if let Some(current) = &selector.current {
                        let term = current.into();

                        if let Some(s) = self.selectors.last_mut() {
                            s.selector_filter.push_term(Some(term));
                        } else {
                            self.selector_filter.push_term(Some(term));
                        }
                    } else {
                        unreachable!()
                    }
                }
                _ => {}
            }
        }

        if self.selectors.is_empty() {
            return false;
        }

        self.selectors.last_mut().unwrap().handle(token, parse_value_reader);
        true
    }
}

impl<'a> JsonSelector<'a> {
    fn visit_absolute(&mut self) {
        if self.current.is_some() {
            if let Some(value) = self.value {
                let selector = JsonSelector {
                    parser: None,
                    value: Some(value),
                    tokens: Vec::new(),
                    current: Some(vec![value]),
                    selectors: Vec::new(),
                    selector_filter: FilterTerms(Vec::new()),
                };
                self.selectors.push(selector);
            }
            return;
        }

        if let Some(v) = &self.value {
            self.current = Some(vec![v]);
        }
    }

    fn visit_relative(&mut self) {
        if let Some(ParseToken::Array) = self.tokens.last() {
            let array_token = self.tokens.pop();
            if let Some(ParseToken::Leaves) = self.tokens.last() {
                self.tokens.pop();
                self.current = self.selector_filter.collect_all(self.current.take());
            }
            self.tokens.push(array_token.unwrap());
        }
        self.selector_filter.new_filter_context();
    }

    fn visit_array_eof(&mut self) {
        if self.is_last_before_token_match(ParseToken::Array) {
            if let Some(Some(e)) = self.selector_filter.pop_term() {
                if let ExprTerm::String(key) = e {
                    self.current = self.selector_filter.filter_next_with_str(self.current.take(), key);
                    self.tokens.pop();
                    return;
                }

                self.selector_filter.push_term(Some(e));
            }
        }

        if self.is_last_before_token_match(ParseToken::Leaves) {
            self.tokens.pop();
            self.tokens.pop();
            if let Some(Some(e)) = self.selector_filter.pop_term() {
                let selector_filter_consumed = match e {
                    ExprTerm::Number(n) => {
                        self.current = self.selector_filter.collect_all_with_num(self.current.take(), utils::to_f64(&n));
                        self.selector_filter.pop_term();
                        true
                    }
                    ExprTerm::String(key) => {
                        self.current = self.selector_filter.collect_all_with_str(self.current.take(), key);
                        self.selector_filter.pop_term();
                        true
                    }
                    _ => {
                        self.selector_filter.push_term(Some(e));
                        false
                    }
                };

                if selector_filter_consumed {
                    return;
                }
            }
        }

        if let Some(Some(e)) = self.selector_filter.pop_term() {
            match e {
                ExprTerm::Number(n) => {
                    self.current = self.selector_filter.collect_next_with_num(self.current.take(), utils::to_f64(&n));
                }
                ExprTerm::String(key) => {
                    self.current = self.selector_filter.collect_next_with_str(self.current.take(), &[key]);
                }
                ExprTerm::Json(rel, _, v) => {
                    if v.is_empty() {
                        self.current = Some(Vec::new());
                    } else if let Some(vec) = rel {
                        self.current = Some(vec);
                    } else {
                        self.current = Some(v);
                    }
                }
                ExprTerm::Bool(false) => {
                    self.current = Some(vec![]);
                }
                _ => {}
            }
        }

        self.tokens.pop();
    }

    fn is_last_before_token_match(&mut self, token: ParseToken) -> bool {
        if self.tokens.len() > 1 {
            return token == self.tokens[self.tokens.len() - 2];
        }

        false
    }

    fn visit_all(&mut self) {
        if let Some(ParseToken::Array) = self.tokens.last() {
            self.tokens.pop();
        }

        match self.tokens.last() {
            Some(ParseToken::Leaves) => {
                self.tokens.pop();
                self.current = self.selector_filter.collect_all(self.current.take());
            }
            Some(ParseToken::In) => {
                self.tokens.pop();
                self.current = self.selector_filter.collect_next_all(self.current.take());
            }
            _ => {
                self.current = self.selector_filter.collect_next_all(self.current.take());
            }
        }
    }

    fn visit_key(&mut self, key: &'a str) {
        if let Some(ParseToken::Array) = self.tokens.last() {
            self.selector_filter.push_term(Some(ExprTerm::String(key)));
            return;
        }

        if let Some(t) = self.tokens.pop() {
            if self.selector_filter.is_term_empty() {
                match t {
                    ParseToken::Leaves => {
                        self.current = self.selector_filter.collect_all_with_str(self.current.take(), key)
                    }
                    ParseToken::In => {
                        self.current = self.selector_filter.collect_next_with_str(self.current.take(), &[key])
                    }
                    _ => {}
                }
            } else {
                match t {
                    ParseToken::Leaves => {
                        self.current = self.selector_filter.filter_all_with_str(self.current.take(), key);
                    }
                    ParseToken::In => {
                        self.current = self.selector_filter.filter_next_with_str(self.current.take(), key);
                    }
                    _ => {}
                }
            }
        }
    }

    fn visit_keys(&mut self, keys: &[&'a str]) {
        if !self.selector_filter.is_term_empty() {
            unimplemented!("keys in filter");
        }

        if let Some(ParseToken::Array) = self.tokens.pop() {
            self.current = self.selector_filter.collect_next_with_str(self.current.take(), keys);
        } else {
            unreachable!();
        }
    }

    fn visit_filter(&mut self, ft: &FilterToken) {
        let right = match self.selector_filter.pop_term() {
            Some(Some(right)) => right,
            Some(None) => ExprTerm::Json(
                None,
                None,
                match &self.current {
                    Some(current) => current.to_vec(),
                    _ => unreachable!(),
                },
            ),
            _ => panic!("empty term right"),
        };

        let mut left = match self.selector_filter.pop_term() {
            Some(Some(left)) => left,
            Some(None) => ExprTerm::Json(
                None,
                None,
                match &self.current {
                    Some(current) => current.to_vec(),
                    _ => unreachable!(),
                },
            ),
            _ => panic!("empty term left"),
        };

        let expr = match ft {
            FilterToken::Equal => left.eq_(right),
            FilterToken::NotEqual => left.ne_(right),
            FilterToken::Greater => left.gt(right),
            FilterToken::GreaterOrEqual => left.ge(right),
            FilterToken::Little => left.lt(right),
            FilterToken::LittleOrEqual => left.le(right),
            FilterToken::And => left.and(right),
            FilterToken::Or => left.or(right),
        };

        self.selector_filter.push_term(Some(expr));
    }

    fn visit_range(&mut self, from: &Option<isize>, to: &Option<isize>, step: &Option<usize>) {
        if !self.selector_filter.is_term_empty() {
            unimplemented!("range syntax in filter");
        }

        if let Some(ParseToken::Array) = self.tokens.pop() {
            let mut tmp = Vec::new();
            if let Some(current) = &self.current {
                for v in current {
                    if let Value::Array(vec) = v {
                        let from = if let Some(from) = from {
                            utils::abs_index(*from, vec.len())
                        } else {
                            0
                        };

                        let to = if let Some(to) = to {
                            utils::abs_index(*to, vec.len())
                        } else {
                            vec.len()
                        };

                        for i in (from..to).step_by(match step {
                            Some(step) => *step,
                            _ => 1,
                        }) {
                            if let Some(v) = vec.get(i) {
                                tmp.push(v);
                            }
                        }
                    }
                }
            }
            self.current = Some(tmp);
        } else {
            unreachable!();
        }
    }

    fn visit_union(&mut self, indices: &[isize]) {
        if !self.selector_filter.is_term_empty() {
            unimplemented!("union syntax in filter");
        }

        if let Some(ParseToken::Array) = self.tokens.pop() {
            let mut tmp = Vec::new();
            if let Some(current) = &self.current {
                for v in current {
                    if let Value::Array(vec) = v {
                        for i in indices {
                            if let Some(v) = vec.get(utils::abs_index(*i, vec.len())) {
                                tmp.push(v);
                            }
                        }
                    }
                }
            }

            self.current = Some(tmp);
        } else {
            unreachable!();
        }
    }
}

impl<'a> ParserTokenHandler<'a> for JsonSelector<'a> {
    fn handle<F>(&mut self, token: &ParseToken, parse_value_reader: &F)
        where
            F: Fn(&StrRange) -> &'a str
    {
        debug!("token: {:?}, stack: {:?}", token, self.tokens);

        if self.compute_absolute_path_filter(token, parse_value_reader) {
            return;
        }

        match token {
            ParseToken::Absolute => self.visit_absolute(),
            ParseToken::Relative => self.visit_relative(),
            ParseToken::In | ParseToken::Leaves | ParseToken::Array => {
                self.tokens.push(token.clone());
            }
            ParseToken::ArrayEof => self.visit_array_eof(),
            ParseToken::All => self.visit_all(),
            ParseToken::Bool(b) => {
                self.selector_filter.push_term(Some(ExprTerm::Bool(*b)));
            }
            ParseToken::Key(s) => {
                let key = parse_value_reader(s);
                self.visit_key(key);
            }
            ParseToken::Keys(keys) => {
                let keys: Vec<&str> = keys.iter().map(|s| { parse_value_reader(s) }).collect();
                self.visit_keys(&keys)
            }
            ParseToken::Number(v) => {
                self.selector_filter.push_term(Some(ExprTerm::Number(Number::from_f64(*v).unwrap())));
            }
            ParseToken::Filter(ref ft) => self.visit_filter(ft),
            ParseToken::Range(from, to, step) => self.visit_range(from, to, step),
            ParseToken::Union(indices) => self.visit_union(indices),
            ParseToken::Eof => {
                debug!("visit_token eof");
            }
        }
    }
}

#[derive(Default)]
pub struct JsonSelectorMut<'a> {
    value: Option<Value>,
    parser: Option<Rc<PathParser<'a>>>,
}

impl<'a> JsonSelectorMut<'a> {
    pub fn new(parser: PathParser<'a>) -> Self {
        Self::new_ref(Rc::new(parser))
    }

    pub fn new_ref(parser: Rc<PathParser<'a>>) -> Self {
        JsonSelectorMut {
            value: None,
            parser: Some(parser),
        }
    }

    pub fn reset_parser(&mut self, parser: PathParser<'a>) -> &mut Self {
        self.parser = Some(Rc::new(parser));
        self
    }

    pub fn reset_parser_ref(&mut self, parser: Rc<PathParser<'a>>) -> &mut Self {
        self.parser = Some(parser);
        self
    }

    pub fn value(&mut self, value: Value) -> &mut Self {
        self.value = Some(value);
        self
    }

    pub fn take(&mut self) -> Option<Value> {
        self.value.take()
    }

    pub fn delete(&mut self) -> Result<&mut Self, JsonPathError> {
        self.replace_with(&mut |_| Some(Value::Null))
    }

    pub fn remove(&mut self) -> Result<&mut Self, JsonPathError> {
        self.replace_with(&mut |_| None)
    }

    fn select(&self) -> Result<Vec<&Value>, JsonPathError> {
        let mut selector = JsonSelector::default();

        if let Some(parser) = self.parser.as_ref() {
            selector.reset_parser_ref(Rc::clone(parser));
        } else {
            return Err(JsonPathError::EmptyPath);
        }

        if let Some(value) = self.value.as_ref() {
            selector.value(value);
        } else {
            return Err(JsonPathError::EmptyValue);
        }

        selector.select()
    }

    pub fn replace_with<F>(&mut self, fun: &mut F) -> Result<&mut Self, JsonPathError>
        where
            F: FnMut(Value) -> Option<Value>,
    {
        let result = self.select()?;
        let paths = self.compute_paths(result);

        if let Some(ref mut value) = &mut self.value {
            for tokens in paths {
                Self::replace_value(tokens, value, fun);
            }
        }

        Ok(self)
    }

    fn replace_value<F>(mut tokens: Vec<String>, value: &mut Value, fun: &mut F)
        where
            F: FnMut(Value) -> Option<Value>
    {
        let mut target = value;

        let last_index = tokens.len().saturating_sub(1);
        for (i, token) in tokens.drain(..).enumerate() {
            let target_once = target;
            let is_last = i == last_index;
            let target_opt = match *target_once {
                Value::Object(ref mut map) => {
                    if is_last {
                        if let Entry::Occupied(mut e) = map.entry(token) {
                            let v = e.insert(Value::Null);
                            if let Some(res) = fun(v) {
                                e.insert(res);
                            } else {
                                e.remove();
                            }
                        }
                        return;
                    }
                    map.get_mut(&token)
                }
                Value::Array(ref mut vec) => {
                    if let Ok(x) = token.parse::<usize>() {
                        if is_last {
                            if x < vec.len() {
                                let v = std::mem::replace(&mut vec[x], Value::Null);
                                if let Some(res) = fun(v) {
                                    vec[x] = res;
                                } else {
                                    vec.remove(x);
                                }
                            }
                            return;
                        }
                        vec.get_mut(x)
                    } else {
                        None
                    }
                }
                _ => None,
            };

            if let Some(t) = target_opt {
                target = t;
            } else {
                break;
            }
        }
    }

    fn compute_paths(&self, mut result: Vec<&Value>) -> Vec<Vec<String>> {
        let mut visited = HashSet::new();
        let mut visited_order = Vec::new();

        if let Some(origin) = &self.value {
            let mut tokens = Vec::new();
            Self::walk(
                origin,
                &mut result,
                &mut tokens,
                &mut visited,
                &mut visited_order,
            );
        }

        visited_order
    }

    fn walk(
        origin: &Value,
        target: &mut Vec<&Value>,
        tokens: &mut Vec<String>,
        visited: &mut HashSet<*const Value>,
        visited_order: &mut Vec<Vec<String>>,
    ) -> bool {
        trace!("{:?}, {:?}", target, tokens);

        if target.is_empty() {
            return true;
        }

        target.retain(|t| {
            if std::ptr::eq(origin, *t) {
                if visited.insert(*t) {
                    visited_order.push(tokens.to_vec());
                }
                false
            } else {
                true
            }
        });

        match origin {
            Value::Array(vec) => {
                for (i, v) in vec.iter().enumerate() {
                    tokens.push(i.to_string());
                    if Self::walk(v, target, tokens, visited, visited_order) {
                        return true;
                    }
                    tokens.pop();
                }
            }
            Value::Object(map) => {
                for (k, v) in map {
                    tokens.push(k.clone());
                    if Self::walk(v, target, tokens, visited, visited_order) {
                        return true;
                    }
                    tokens.pop();
                }
            }
            _ => {}
        }

        false
    }
}