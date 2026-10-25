use std::collections::HashSet;

use serde_json::{Number, Value};

use super::cmp::*;
use super::utils;
use super::value_walker::ValueWalker;

#[derive(Debug, PartialEq)]
pub enum ExprTerm<'a> {
    String(&'a str),
    Number(Number),
    Bool(bool),
    Json(Option<Vec<&'a Value>>, Option<FilterKey<'a>>, Vec<&'a Value>),
}

impl<'a> ExprTerm<'a> {
    fn cmp_string<C>(s1: &str, other: &mut ExprTerm<'a>, cmp_fn: &C) -> ExprTerm<'a>
        where
            C: Cmp,
    {
        match other {
            ExprTerm::String(s2) => {
                let p1 = utils::to_path_str(s1);
                let p2 = utils::to_path_str(s2);
                ExprTerm::Bool(cmp_fn.cmp_string(p1.get_key(), p2.get_key()))
            }
            ExprTerm::Json(_, _, _) => unreachable!(),
            _ => ExprTerm::Bool(cmp_fn.default()),
        }
    }

    fn cmp_number<C>(n1: &Number, other: &mut ExprTerm<'a>, cmp_fn: &C) -> ExprTerm<'a>
        where
            C: Cmp,
    {
        match other {
            ExprTerm::Number(n2) => ExprTerm::Bool(cmp_fn.cmp_f64(utils::to_f64(n1), utils::to_f64(n2))),
            ExprTerm::Json(_, _, _) => unreachable!(),
            _ => ExprTerm::Bool(cmp_fn.default()),
        }
    }

    fn cmp_bool<C>(b1: &bool, other: &mut ExprTerm<'a>, cmp_fn: &C) -> ExprTerm<'a>
        where
            C: Cmp,
    {
        match other {
            ExprTerm::Bool(b2) => ExprTerm::Bool(cmp_fn.cmp_bool(*b1, *b2)),
            ExprTerm::Json(_, _, _) => unreachable!(),
            _ => ExprTerm::Bool(cmp_fn.default()),
        }
    }

    fn cmp_json_string<C>(s2: &str,
                          fk1: &Option<FilterKey>,
                          vec1: &[&'a Value],
                          cmp_fn: &C) -> Vec<&'a Value>
        where
            C: Cmp
    {
        let path_str = utils::to_path_str(s2);
        vec1.iter().filter(|v1| match v1 {
            Value::String(s1) => {
                cmp_fn.cmp_string(s1, path_str.get_key())
            }
            Value::Object(map1) => {
                if let Some(FilterKey::String(k)) = fk1 {
                    if let Some(Value::String(s1)) = map1.get(*k) {
                        return cmp_fn.cmp_string(s1, path_str.get_key());
                    }
                }
                cmp_fn.default()
            }
            _ => cmp_fn.default(),
        }).copied().collect()
    }

    fn cmp_json_number<C>(n2: &Number,
                          fk1: &Option<FilterKey>,
                          vec1: &[&'a Value],
                          cmp_fn: &C) -> Vec<&'a Value>
        where
            C: Cmp
    {
        let n2 = utils::to_f64(n2);
        vec1.iter().filter(|v1| match v1 {
            Value::Number(n1) => cmp_fn.cmp_f64(utils::to_f64(n1), n2),
            Value::Object(map1) => {
                if let Some(FilterKey::String(k)) = fk1 {
                    if let Some(Value::Number(n1)) = map1.get(*k) {
                        return cmp_fn.cmp_f64(utils::to_f64(n1), n2);
                    }
                }
                cmp_fn.default()
            }
            _ => cmp_fn.default(),
        }).copied().collect()
    }

    fn cmp_json_bool<C1>(b2: &bool,
                         fk1: &Option<FilterKey>,
                         vec1: &[&'a Value],
                         cmp_fn: &C1) -> Vec<&'a Value>
        where
            C1: Cmp
    {
        vec1.iter().filter(|v1| match v1 {
            Value::Bool(b1) => cmp_fn.cmp_bool(*b1, *b2),
            Value::Object(map1) => {
                if let Some(FilterKey::String(k)) = fk1 {
                    if let Some(Value::Bool(b1)) = map1.get(*k) {
                        return cmp_fn.cmp_bool(*b1, *b2);
                    }
                }
                cmp_fn.default()
            }
            _ => cmp_fn.default(),
        }).copied().collect()
    }

    fn cmp_json_json<C1>(rel: &Option<Vec<&'a Value>>,
                         parent: &Option<Vec<&'a Value>>,
                         vec1: &[&'a Value],
                         vec2: &[&'a Value],
                         cmp_fn: &C1) -> Vec<&'a Value>
        where
            C1: Cmp
    {
        if let Some(vec1) = rel {
            if let Some(vec2) = parent {
                cmp_fn.cmp_json(vec1, vec2)
            } else {
                cmp_fn.cmp_json(vec1, vec2)
            }
        } else if let Some(vec2) = parent {
            cmp_fn.cmp_json(vec1, vec2)
        } else {
            cmp_fn.cmp_json(vec1, vec2)
        }
    }

    fn cmp_json<C1>(rel: Option<Vec<&'a Value>>,
                    fk1: Option<FilterKey<'a>>,
                    vec1: &mut Vec<&'a Value>,
                    other: &mut ExprTerm<'a>,
                    cmp_fn: &C1) -> ExprTerm<'a>
        where
            C1: Cmp
    {
        let ret: Vec<&Value> = match other {
            ExprTerm::String(s2) => Self::cmp_json_string(s2, &fk1, vec1, cmp_fn),
            ExprTerm::Number(n2) => Self::cmp_json_number(n2, &fk1, vec1, cmp_fn),
            ExprTerm::Bool(b2) => Self::cmp_json_bool(b2, &fk1, vec1, cmp_fn),
            ExprTerm::Json(parent, _, vec2) => {
                Self::cmp_json_json(&rel, parent, vec1, vec2, cmp_fn)
            }
        };

        if ret.is_empty() {
            return ExprTerm::Bool(cmp_fn.default());
        }

        if rel.is_none() {
            return ExprTerm::Json(None, None, ret);
        }

        if rel.is_some() {
            if let ExprTerm::Json(_, _, _) = &other {
                if let Some(rel) = rel {
                    return ExprTerm::Json(Some(rel), None, ret);
                }
            }
        }

        let rel = rel.unwrap();
        let mut object_exist = false;
        for v in &rel {
            if v.is_object() {
                object_exist = true;
                break;
            }
        }

        if !object_exist {
            return ExprTerm::Json(Some(Vec::new()), None, ret);
        }

        let ret_set: HashSet<*const Value> = ret.iter()
            .fold(HashSet::new(), |mut acc, v| {
                let ptr = *v as *const Value;
                acc.insert(ptr);
                acc
            });

        let mut tmp = Vec::new();
        for rv in rel {
            if let Value::Object(map) = rv {
                for map_value in map.values() {
                    let ptr = map_value as *const Value;
                    if ret_set.contains(&ptr) {
                        tmp.push(rv);
                    }
                }
            }
        }

        ExprTerm::Json(Some(tmp), None, ret)
    }

    fn cmp<C1, C2>(&mut self, other: &mut Self, cmp_fn: &C1, rev_cmp_fn: &C2) -> ExprTerm<'a>
        where
            C1: Cmp,
            C2: Cmp
    {
        if let ExprTerm::Json(_, _, _) = other {
            if let ExprTerm::Json(_, _, _) = &self {
                //
            } else {
                return other.cmp(self, rev_cmp_fn, cmp_fn);
            }
        }

        match self {
            ExprTerm::String(s1) => Self::cmp_string(s1, other, cmp_fn),
            ExprTerm::Number(n1) => Self::cmp_number(n1, other, cmp_fn),
            ExprTerm::Bool(b1) => Self::cmp_bool(b1, other, cmp_fn),
            ExprTerm::Json(rel, fk1, vec1) =>
                Self::cmp_json(rel.take(), fk1.take(), vec1, other, cmp_fn)
        }
    }

    pub fn eq_(&mut self, mut other: Self) -> ExprTerm<'a> {
        debug!("eq - {:?} : {:?}", &self, &other);
        let expr = self.cmp(&mut other, &CmpEq, &CmpEq);
        debug!("eq = {:?}", expr);
        expr
    }

    pub fn ne_(&mut self, mut other: Self) -> ExprTerm<'a> {
        debug!("ne - {:?} : {:?}", &self, &other);
        let expr = self.cmp(&mut other, &CmpNe, &CmpNe);
        debug!("ne = {:?}", expr);
        expr
    }

    pub fn gt(&mut self, mut other: Self) -> ExprTerm<'a> {
        debug!("gt - {:?} : {:?}", &self, &other);
        let expr = self.cmp(&mut other, &CmpGt, &CmpLt);
        debug!("gt = {:?}", expr);
        expr
    }

    pub fn ge(&mut self, mut other: Self) -> ExprTerm<'a> {
        debug!("ge - {:?} : {:?}", &self, &other);
        let expr = self.cmp(&mut other, &CmpGe, &CmpLe);
        debug!("ge = {:?}", expr);
        expr
    }

    pub fn lt(&mut self, mut other: Self) -> ExprTerm<'a> {
        debug!("lt - {:?} : {:?}", &self, &other);
        let expr = self.cmp(&mut other, &CmpLt, &CmpGt);
        debug!("lt = {:?}", expr);
        expr
    }

    pub fn le(&mut self, mut other: Self) -> ExprTerm<'a> {
        debug!("le - {:?} : {:?}", &self, &other);
        let expr = self.cmp(&mut other, &CmpLe, &CmpGe);
        debug!("le = {:?}", expr);
        expr
    }

    pub fn and(&mut self, mut other: Self) -> ExprTerm<'a> {
        debug!("and - {:?} : {:?}", &self, &other);
        let expr = self.cmp(&mut other, &CmpAnd, &CmpAnd);
        debug!("and = {:?}", expr);
        expr
    }

    pub fn or(&mut self, mut other: Self) -> ExprTerm<'a> {
        debug!("or - {:?} : {:?}", &self, &other);
        let expr = self.cmp(&mut other, &CmpOr, &CmpOr);
        debug!("or = {:?}", expr);
        expr
    }
}

impl<'a> From<&Vec<&'a Value>> for ExprTerm<'a> {
    fn from(vec: &Vec<&'a Value>) -> Self {
        if vec.len() == 1 {
            match &vec[0] {
                Value::Number(v) => return ExprTerm::Number(v.clone()),
                Value::String(v) => return ExprTerm::String(v.as_str()),
                Value::Bool(v) => return ExprTerm::Bool(*v),
                _ => {}
            }
        }

        ExprTerm::Json(None, None, vec.to_vec())
    }
}

#[derive(Debug, PartialEq)]
pub enum FilterKey<'a> {
    String(&'a str),
    All,
}

struct FilterResult<'a> {
    key: FilterKey<'a>,
    collected: Vec<&'a Value>
}

#[derive(Debug, Default)]
pub struct FilterTerms<'a>(pub Vec<Option<ExprTerm<'a>>>);

impl<'a> FilterTerms<'a> {
    pub fn new_filter_context(&mut self) {
        self.0.push(None);
        debug!("new_filter_context: {:?}", self.0);
    }

    pub fn is_term_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn push_term(&mut self, term: Option<ExprTerm<'a>>) {
        self.0.push(term);
    }

    #[allow(clippy::option_option)]
    pub fn pop_term(&mut self) -> Option<Option<ExprTerm<'a>>> {
        self.0.pop()
    }

    fn filter_json_term<F>(&mut self, e: ExprTerm<'a>, fun: F)
        where
            F: Fn(&Vec<&'a Value>, &mut Option<HashSet<usize>>) -> FilterResult<'a>,
    {
        debug!("filter_json_term: {:?}", e);

        if let ExprTerm::Json(rel, fk, vec) = e {
            let mut not_matched = Some(HashSet::new());
            let filter_result = if let Some(FilterKey::String(key)) = fk {
                fun(&ValueWalker::next_with_str(&vec, key), &mut not_matched)
            } else {
                fun(&vec, &mut not_matched)
            };

            if rel.is_some() {
                self.push_term(Some(ExprTerm::Json(
                    rel,
                    Some(filter_result.key),
                    filter_result.collected)));
            } else {
                let not_matched = not_matched.unwrap();
                let filtered = vec.iter().enumerate()
                    .filter(|(idx, _)| !not_matched.contains(idx))
                    .map(|(_, v)| *v).collect();
                self.push_term(Some(ExprTerm::Json(
                    Some(filtered),
                    Some(filter_result.key),
                    filter_result.collected)));
            }
        } else {
            unreachable!("unexpected: ExprTerm: {:?}", e);
        }
    }

    fn push_json_term<F>(&mut self, current: Option<Vec<&'a Value>>, fun: F) -> Option<Vec<&'a Value>>
        where
            F: Fn(&Vec<&'a Value>, &mut Option<HashSet<usize>>) -> FilterResult<'a>,
    {
        debug!("push_json_term: {:?}", &current);

        if let Some(current) = &current {
            let filter_result = fun(current, &mut None);
            self.push_term(Some(ExprTerm::Json(
                None,
                Some(filter_result.key),
                filter_result.collected)));
        }

        current
    }

    fn filter<F>(&mut self, current: Option<Vec<&'a Value>>, fun: F) -> Option<Vec<&'a Value>>
        where
            F: Fn(&Vec<&'a Value>, &mut Option<HashSet<usize>>) -> FilterResult<'a>,
    {
        let peek = self.pop_term();

        if let Some(None) = peek {
            return self.push_json_term(current, fun);
        }

        if let Some(Some(e)) = peek {
            self.filter_json_term(e, fun);
        }

        current
    }

    pub fn filter_all_with_str(&mut self, current: Option<Vec<&'a Value>>, key: &'a str) -> Option<Vec<&'a Value>> {
        let current = self.filter(current, |vec, _| {
            FilterResult {
                key: FilterKey::All,
                collected: ValueWalker::all_with_str(vec, key)
            }
        });

        debug!("filter_all_with_str : {}, {:?}", key, self.0);
        current
    }

    pub fn filter_next_with_str(&mut self, current: Option<Vec<&'a Value>>, key: &'a str) -> Option<Vec<&'a Value>> {
        let current = self.filter(current, |vec, not_matched| {
            let mut visited = HashSet::new();
            let mut acc = Vec::new();

            let path_key = &utils::to_path_str(key);

            ValueWalker::walk_dedup_all(vec,
                                        path_key.get_key(),
                                        &mut visited,
                                        &mut |v| {
                                            acc.push(v);
                                        },
                                        &mut |idx| {
                                            if let Some(set) = not_matched {
                                                set.insert(idx);
                                            }
                                        },
                                        0
            );

            FilterResult {
                key: FilterKey::String(path_key.get_origin_key()),
                collected: acc
            }
        });

        debug!("filter_next_with_str : {}, {:?}", key, self.0);
        current
    }

    pub fn collect_next_with_num(&mut self, current: Option<Vec<&'a Value>>, index: f64) -> Option<Vec<&'a Value>> {
        if current.is_none() {
            debug!("collect_next_with_num : {:?}, {:?}", &index, &current);
            return current;
        }

        if let Some(Some(e)) = self.pop_term() {
            match e {
                ExprTerm::Json(rel, _, vec) => {
                    return if vec.is_empty() {
                        Some(Vec::new())
                    } else if let Some(vec) = rel {
                        let index = utils::abs_index(index as isize, vec.len());
                        let ret = vec.get(index).map_or(Vec::new(), |v| vec![*v]);
                        Some(ret)
                    } else {
                        let index = utils::abs_index(index as isize, vec.len());
                        let ret = vec.get(index).map_or(Vec::new(), |v| vec![*v]);
                        Some(ret)
                    };
                }
                _ => {
                    self.push_term(Some(e));
                }
            }
        }

        let acc = ValueWalker::next_with_num(&current.unwrap(), index);

        if acc.is_empty() {
            self.pop_term();
        }

        Some(acc)
    }

    pub fn collect_next_with_str(&mut self, current: Option<Vec<&'a Value>>, keys: &[&'a str]) -> Option<Vec<&'a Value>> {
        if current.is_none() {
            debug!(
                "collect_next_with_str : {:?}, {:?}",
                keys, &current
            );
            return current;
        }

        let acc = ValueWalker::all_with_strs(current.as_ref().unwrap(), keys);

        if acc.is_empty() {
            self.pop_term();
        }

        Some(acc)
    }

    pub fn collect_next_all(&mut self, current: Option<Vec<&'a Value>>) -> Option<Vec<&'a Value>> {
        if current.is_none() {
            debug!("collect_next_all : {:?}", &current);
            return current;
        }

        Some(ValueWalker::next_all(&current.unwrap()))
    }

    pub fn collect_all(&mut self, current: Option<Vec<&'a Value>>) -> Option<Vec<&'a Value>> {
        if current.is_none() {
            debug!("collect_all: {:?}", &current);
            return current;
        }

        Some(ValueWalker::all(current.as_ref().unwrap()))
    }

    pub fn collect_all_with_str(&mut self, current: Option<Vec<&'a Value>>, key: &'a str) -> Option<Vec<&'a Value>> {
        if current.is_none() {
            debug!("collect_all_with_str: {}, {:?}", key, &current);
            return current;
        }

        let ret = ValueWalker::all_with_str(current.as_ref().unwrap(), key);
        Some(ret)
    }

    pub fn collect_all_with_num(&mut self, mut current: Option<Vec<&'a Value>>, index: f64) -> Option<Vec<&'a Value>> {
        if let Some(current) = current.take() {
            let ret = ValueWalker::all_with_num(&current, index);
            if !ret.is_empty() {
                return Some(ret);
            }
        }

        debug!("collect_all_with_num: {}, {:?}", index, &current);
        None
    }
}

#[cfg(test)]
mod expr_term_inner_tests {
    use serde_json::{Number, Value};

    use selector::terms::ExprTerm;

    #[test]
    fn value_vec_into() {
        let v = Value::Bool(true);
        let vec = &vec![&v];
        let term: ExprTerm = vec.into();
        assert_eq!(term, ExprTerm::Bool(true));

        let v = Value::String("a".to_string());
        let vec = &vec![&v];
        let term: ExprTerm = vec.into();
        assert_eq!(term, ExprTerm::String("a"));

        let v = serde_json::from_str("1.0").unwrap();
        let vec = &vec![&v];
        let term: ExprTerm = vec.into();
        assert_eq!(term, ExprTerm::Number(Number::from_f64(1.0).unwrap()));
    }
}
