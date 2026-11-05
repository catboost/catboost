use serde_json::Number;

pub fn to_f64(n: &Number) -> f64 {
    if n.is_i64() {
        n.as_i64().unwrap() as f64
    } else if n.is_f64() {
        n.as_f64().unwrap()
    } else {
        n.as_u64().unwrap() as f64
    }
}

pub fn abs_index(n: isize, len: usize) -> usize {
    if n < 0_isize {
        (n + len as isize).max(0) as usize
    } else {
        n.min(len as isize) as usize
    }
}

pub struct PathKey<'a> {
    key: &'a str,
    special_key: Option<String>
}

impl<'a: 'b, 'b> PathKey<'a> {
    pub fn get_key(&'a self) -> &'b str {
        if let Some(skey) = self.special_key.as_ref() {
            skey
        } else {
            self.key
        }
    }

    pub fn get_origin_key(&self) -> &'a str {
        self.key
    }
}

pub fn to_path_str(key: &str) -> PathKey {
    let mut path_key = PathKey {
        key,
        special_key: None
    };

    if key.starts_with('\'') || key.starts_with('"') {
        let s = &key[1..key.len() - 1];
        path_key.key = s;
        if key.contains('\\') {
            path_key.special_key = Some(s.chars().filter(|ch| ch != &'\\').collect());
        }
    }
    path_key
}