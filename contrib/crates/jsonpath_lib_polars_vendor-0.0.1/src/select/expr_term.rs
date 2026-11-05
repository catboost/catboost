use serde_json::{Number, Value};
use select::cmp::*;
use select::{FilterKey, to_f64};

#[derive(Debug, PartialEq)]
pub(super) enum ExprTerm<'a> {
    String(String),
    Number(Number),
    Bool(bool),
    Json(Option<Vec<&'a Value>>, Option<FilterKey>, Vec<&'a Value>),
}

impl<'a> ExprTerm<'a> {
    fn cmp<C1: Cmp, C2: Cmp>(
        &self,
        other: &Self,
        cmp_fn: &C1,
        reverse_cmp_fn: &C2,
    ) -> ExprTerm<'a> {
        match &self {
            ExprTerm::String(s1) => match &other {
                ExprTerm::String(s2) => ExprTerm::Bool(cmp_fn.cmp_string(s1, s2)),
                ExprTerm::Json(_, _, _) => other.cmp(self, reverse_cmp_fn, cmp_fn),
                _ => ExprTerm::Bool(cmp_fn.default()),
            },
            ExprTerm::Number(n1) => match &other {
                ExprTerm::Number(n2) => ExprTerm::Bool(cmp_fn.cmp_f64(to_f64(n1), to_f64(n2))),
                ExprTerm::Json(_, _, _) => other.cmp(self, reverse_cmp_fn, cmp_fn),
                _ => ExprTerm::Bool(cmp_fn.default()),
            },
            ExprTerm::Bool(b1) => match &other {
                ExprTerm::Bool(b2) => ExprTerm::Bool(cmp_fn.cmp_bool(*b1, *b2)),
                ExprTerm::Json(_, _, _) => other.cmp(self, reverse_cmp_fn, cmp_fn),
                _ => ExprTerm::Bool(cmp_fn.default()),
            },
            ExprTerm::Json(rel, fk1, vec1) => {
                let ret: Vec<&Value> = match &other {
                    ExprTerm::String(s2) => vec1
                        .iter()
                        .filter(|v1| match v1 {
                            Value::String(s1) => cmp_fn.cmp_string(s1, s2),
                            Value::Object(map1) => {
                                if let Some(FilterKey::String(k)) = fk1 {
                                    if let Some(Value::String(s1)) = map1.get(k) {
                                        return cmp_fn.cmp_string(s1, s2);
                                    }
                                }
                                cmp_fn.default()
                            }
                            _ => cmp_fn.default(),
                        })
                        .cloned()
                        .collect(),
                    ExprTerm::Number(n2) => vec1
                        .iter()
                        .filter(|v1| match v1 {
                            Value::Number(n1) => cmp_fn.cmp_f64(to_f64(n1), to_f64(n2)),
                            Value::Object(map1) => {
                                if let Some(FilterKey::String(k)) = fk1 {
                                    if let Some(Value::Number(n1)) = map1.get(k) {
                                        return cmp_fn.cmp_f64(to_f64(n1), to_f64(n2));
                                    }
                                }
                                cmp_fn.default()
                            }
                            _ => cmp_fn.default(),
                        })
                        .cloned()
                        .collect(),
                    ExprTerm::Bool(b2) => vec1
                        .iter()
                        .filter(|v1| match v1 {
                            Value::Bool(b1) => cmp_fn.cmp_bool(*b1, *b2),
                            Value::Object(map1) => {
                                if let Some(FilterKey::String(k)) = fk1 {
                                    if let Some(Value::Bool(b1)) = map1.get(k) {
                                        return cmp_fn.cmp_bool(*b1, *b2);
                                    }
                                }
                                cmp_fn.default()
                            }
                            _ => cmp_fn.default(),
                        })
                        .cloned()
                        .collect(),
                    ExprTerm::Json(parent, _, vec2) => {
                        if let Some(vec1) = rel {
                            cmp_fn.cmp_json(vec1, vec2)
                        } else if let Some(vec2) = parent {
                            cmp_fn.cmp_json(vec1, vec2)
                        } else {
                            cmp_fn.cmp_json(vec1, vec2)
                        }
                    }
                };

                if ret.is_empty() {
                    ExprTerm::Bool(cmp_fn.default())
                } else if let Some(rel) = rel {
                    if let ExprTerm::Json(_, _, _) = &other {
                        ExprTerm::Json(Some(rel.to_vec()), None, ret)
                    } else {
                        let mut tmp = Vec::new();
                        for rel_value in rel {
                            if let Value::Object(map) = rel_value {
                                for map_value in map.values() {
                                    for result_value in &ret {
                                        if map_value.eq(*result_value) {
                                            tmp.push(*rel_value);
                                        }
                                    }
                                }
                            }
                        }
                        ExprTerm::Json(Some(tmp), None, ret)
                    }
                } else {
                    ExprTerm::Json(None, None, ret)
                }
            }
        }
    }

    pub fn eq(&self, other: &Self, ret: &mut Option<ExprTerm<'a>>) {
        debug!("eq - {:?} : {:?}", &self, &other);
        let _ = ret.take();
        let tmp = self.cmp(other, &CmpEq, &CmpEq);
        debug!("eq = {:?}", tmp);
        *ret = Some(tmp);
    }

    pub fn ne(&self, other: &Self, ret: &mut Option<ExprTerm<'a>>) {
        debug!("ne - {:?} : {:?}", &self, &other);
        let _ = ret.take();
        let tmp = self.cmp(other, &CmpNe, &CmpNe);
        debug!("ne = {:?}", tmp);
        *ret = Some(tmp);
    }

    pub fn gt(&self, other: &Self, ret: &mut Option<ExprTerm<'a>>) {
        debug!("gt - {:?} : {:?}", &self, &other);
        let _ = ret.take();
        let tmp = self.cmp(other, &CmpGt, &CmpLt);
        debug!("gt = {:?}", tmp);
        *ret = Some(tmp);
    }

    pub fn ge(&self, other: &Self, ret: &mut Option<ExprTerm<'a>>) {
        debug!("ge - {:?} : {:?}", &self, &other);
        let _ = ret.take();
        let tmp = self.cmp(other, &CmpGe, &CmpLe);
        debug!("ge = {:?}", tmp);
        *ret = Some(tmp);
    }

    pub fn lt(&self, other: &Self, ret: &mut Option<ExprTerm<'a>>) {
        debug!("lt - {:?} : {:?}", &self, &other);
        let _ = ret.take();
        let tmp = self.cmp(other, &CmpLt, &CmpGt);
        debug!("lt = {:?}", tmp);
        *ret = Some(tmp);
    }

    pub fn le(&self, other: &Self, ret: &mut Option<ExprTerm<'a>>) {
        debug!("le - {:?} : {:?}", &self, &other);
        let _ = ret.take();
        let tmp = self.cmp(other, &CmpLe, &CmpGe);
        debug!("le = {:?}", tmp);
        *ret = Some(tmp);
    }

    pub fn and(&self, other: &Self, ret: &mut Option<ExprTerm<'a>>) {
        debug!("and - {:?} : {:?}", &self, &other);
        let _ = ret.take();
        let tmp = self.cmp(other, &CmpAnd, &CmpAnd);
        debug!("and = {:?}", tmp);
        *ret = Some(tmp);
    }

    pub fn or(&self, other: &Self, ret: &mut Option<ExprTerm<'a>>) {
        debug!("or - {:?} : {:?}", &self, &other);
        let _ = ret.take();
        let tmp = self.cmp(other, &CmpOr, &CmpOr);
        debug!("or = {:?}", tmp);
        *ret = Some(tmp);
    }
}

impl<'a> From<&Vec<&'a Value>> for ExprTerm<'a> {
    fn from(vec: &Vec<&'a Value>) -> Self {
        if vec.len() == 1 {
            match &vec[0] {
                Value::Number(v) => return ExprTerm::Number(v.clone()),
                Value::String(v) => return ExprTerm::String(v.clone()),
                Value::Bool(v) => return ExprTerm::Bool(*v),
                _ => {}
            }
        }

        ExprTerm::Json(None, None, vec.to_vec())
    }
}

// #[cfg(test)]
// mod expr_term_inner_tests {
//     use serde_json::{Number, Value};
//     use select::expr_term::ExprTerm;
//
//     #[test]
//     fn value_vec_into() {
//         let v = Value::Bool(true);
//         let vec = &vec![&v];
//         let term: ExprTerm = vec.into();
//         assert_eq!(term, ExprTerm::Bool(true));
//
//         let v = Value::String("a".to_string());
//         let vec = &vec![&v];
//         let term: ExprTerm = vec.into();
//         assert_eq!(term, ExprTerm::String("a".to_string()));
//
//         let v = serde_json::from_str("1.0").unwrap();
//         let vec = &vec![&v];
//         let term: ExprTerm = vec.into();
//         assert_eq!(term, ExprTerm::Number(Number::from_f64(1.0).unwrap()));
//     }
// }
