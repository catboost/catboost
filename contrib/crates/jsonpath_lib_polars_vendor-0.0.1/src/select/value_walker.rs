use serde_json::Value;
use std::collections::HashSet;

pub(super) struct ValueWalker;

impl<'a> ValueWalker {
    pub fn all_with_num(vec: &[&'a Value], tmp: &mut Vec<&'a Value>, index: f64) {
        Self::walk(vec, tmp, &|v| if v.is_array() {
            v.get(index as usize).map(|item| vec![item])
        } else {
            None
        });
    }

    pub fn all_with_str(vec: &[&'a Value], tmp: &mut Vec<&'a Value>, key: &str, is_filter: bool) {
        if is_filter {
            Self::walk(vec, tmp, &|v| match v {
                Value::Object(map) if map.contains_key(key) => Some(vec![v]),
                _ => None,
            });
        } else {
            Self::walk(vec, tmp, &|v| match v {
                Value::Object(map) => map.get(key).map(|v| vec![v]),
                _ => None,
            });
        }
    }

    pub fn all(vec: &[&'a Value], tmp: &mut Vec<&'a Value>) {
        Self::walk(vec, tmp, &|v| match v {
            Value::Array(vec) => Some(vec.iter().collect()),
            Value::Object(map) => {
                let mut tmp = Vec::new();
                for (_, v) in map {
                    tmp.push(v);
                }
                Some(tmp)
            }
            _ => None,
        });
    }

    fn walk<F>(vec: &[&'a Value], tmp: &mut Vec<&'a Value>, fun: &F) where F: Fn(&Value) -> Option<Vec<&Value>> {
        for v in vec {
            Self::_walk(v, tmp, fun);
        }
    }

    fn _walk<F>(v: &'a Value, tmp: &mut Vec<&'a Value>, fun: &F) where F: Fn(&Value) -> Option<Vec<&Value>> {
        if let Some(mut ret) = fun(v) {
            tmp.append(&mut ret);
        }

        match v {
            Value::Array(vec) => {
                for v in vec {
                    Self::_walk(v, tmp, fun);
                }
            }
            Value::Object(map) => {
                for (_, v) in map {
                    Self::_walk(v, tmp, fun);
                }
            }
            _ => {}
        }
    }

    pub fn walk_dedup(v: &'a Value,
                      tmp: &mut Vec<&'a Value>,
                      key: &str,
                      visited: &mut HashSet<*const Value>, ) {
        match v {
            Value::Object(map) => {
                if map.contains_key(key) {
                    let ptr = v as *const Value;
                    if !visited.contains(&ptr) {
                        visited.insert(ptr);
                        tmp.push(v)
                    }
                }
            }
            Value::Array(vec) => {
                for v in vec {
                    Self::walk_dedup(v, tmp, key, visited);
                }
            }
            _ => {}
        }
    }
}

