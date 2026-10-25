use core::cmp::{max, min};
pub const kNumDistanceCacheEntries: usize = 4;

use super::super::dictionary::{
    kBrotliDictionary, kBrotliDictionaryOffsetsByLength, kBrotliDictionarySizeBitsByLength,
};
use super::static_dict_lut::{
    kDictHashMul32, kDictNumBits, kStaticDictionaryBuckets, kStaticDictionaryWords, DictWord,
};
#[allow(unused)]
static kUppercaseFirst: u8 = 10u8;

#[allow(unused)]
static kOmitLastNTransforms: [u8; 10] = [0, 12, 27, 23, 42, 63, 56, 48, 59, 64];

pub struct BrotliDictionary {
    pub size_bits_by_length: &'static [u8; 25],
    pub offsets_by_length: &'static [u32; 25],
    pub data: &'static [u8; 122784],
}

pub static kBrotliEncDictionary: BrotliDictionary = BrotliDictionary {
    size_bits_by_length: &kBrotliDictionarySizeBitsByLength,
    offsets_by_length: &kBrotliDictionaryOffsetsByLength,
    data: &kBrotliDictionary,
};

#[inline(always)]
pub fn BrotliGetDictionary() -> &'static BrotliDictionary {
    &kBrotliEncDictionary
}
#[inline(always)]
pub fn BROTLI_UNALIGNED_LOAD32(sl: &[u8]) -> u32 {
    let mut p = [0u8; 4];
    p[..].clone_from_slice(sl.split_at(4).0);
    (p[0] as u32) | ((p[1] as u32) << 8) | ((p[2] as u32) << 16) | ((p[3] as u32) << 24)
}
#[inline(always)]
pub fn Hash(data: &[u8]) -> u32 {
    let h: u32 = BROTLI_UNALIGNED_LOAD32(data).wrapping_mul(kDictHashMul32);
    h >> (32i32 - kDictNumBits)
}
#[inline(always)]
pub fn BROTLI_UNALIGNED_LOAD64(sl: &[u8]) -> u64 {
    let mut p = [0u8; 8];
    p[..].clone_from_slice(sl.split_at(8).0);
    (p[0] as u64)
        | ((p[1] as u64) << 8)
        | ((p[2] as u64) << 16)
        | ((p[3] as u64) << 24)
        | ((p[4] as u64) << 32)
        | ((p[5] as u64) << 40)
        | ((p[6] as u64) << 48)
        | ((p[7] as u64) << 56)
}
#[inline(always)]
pub fn BROTLI_UNALIGNED_STORE64(outp: &mut [u8], v: u64) {
    let p = [
        (v & 0xff) as u8,
        ((v >> 8) & 0xff) as u8,
        ((v >> 16) & 0xff) as u8,
        ((v >> 24) & 0xff) as u8,
        ((v >> 32) & 0xff) as u8,
        ((v >> 40) & 0xff) as u8,
        ((v >> 48) & 0xff) as u8,
        ((v >> 56) & 0xff) as u8,
    ];
    outp.split_at_mut(8).0.clone_from_slice(&p[..]);
}

macro_rules! sub_match {
    ($s1 : expr, $s2 : expr, $limit : expr, $matched : expr, $split_pair1 : expr, $split_pair2 : expr, $s1_lo : expr, $s2_lo : expr, $s1_as_64 : expr, $s2_as_64 : expr, $vec_len: expr) => {
        $split_pair1 = $s1.split_at($vec_len);
        $s1_lo[..$vec_len].clone_from_slice($split_pair1.0);
        $s1 = $split_pair1.1;
        $split_pair2 = $s2.split_at($vec_len);
        $s2_lo[..$vec_len].clone_from_slice($split_pair2.0);
        $s2 = $split_pair2.1;
        $limit -= $vec_len;
        for index in 0..($vec_len >> 3) {
            $s1_as_64 = BROTLI_UNALIGNED_LOAD64(&$s1_lo[(index << 3)..((index + 1) << 3)]);
            $s2_as_64 = BROTLI_UNALIGNED_LOAD64(&$s2_lo[(index << 3)..((index + 1) << 3)]);
            if $s2_as_64 == $s1_as_64 {
                $matched = $matched.wrapping_add(8) as u32 as usize;
            } else {
                $matched = $matched
                    .wrapping_add((($s2_as_64 ^ $s1_as_64).trailing_zeros() >> 3) as usize)
                    as u32 as usize;
                return $matched;
            }
        }
    };
}

macro_rules! sub_match8 {
    ($s1 : expr, $s2 : expr, $limit : expr, $matched : expr, $s1_as_64 : expr, $s2_as_64 : expr) => {
        $limit -= 8;
        $s1_as_64 = BROTLI_UNALIGNED_LOAD64($s1);
        $s1 = $s1.split_at(8).1;
        $s2_as_64 = BROTLI_UNALIGNED_LOAD64($s2);
        $s2 = $s2.split_at(8).1;
        if $s2_as_64 == $s1_as_64 {
            $matched = $matched.wrapping_add(8) as u32 as usize;
        } else {
            $matched = $matched
                .wrapping_add((($s2_as_64 ^ $s1_as_64).trailing_zeros() >> 3) as usize)
                as u32 as usize;
            return $matched;
        }
    };
}

// factor of 10 slower (example takes 158s, not 30, and for the 30 second run it took 15 of them)
#[allow(unused)]
pub fn SlowerFindMatchLengthWithLimit(s1: &[u8], s2: &[u8], limit: usize) -> usize {
    for index in 0..limit {
        if s1[index] != s2[index] {
            return index;
        }
    }
    limit
}
// factor of 5 slower (example takes 90 seconds)
#[allow(unused)]
pub fn FindMatchLengthWithLimit(s1: &[u8], s2: &[u8], limit: usize) -> usize {
    for (index, pair) in s1[..limit].iter().zip(s2[..limit].iter()).enumerate() {
        if *pair.0 != *pair.1 {
            return index;
        }
    }
    limit
}
#[allow(unused)]
pub fn FindMatchLengthWithLimitMin4(s1: &[u8], s2: &[u8], limit: usize) -> usize {
    let (s1_start, s1_rest) = s1.split_at(5);
    let (s2_start, s2_rest) = s2.split_at(5);
    let v0 = BROTLI_UNALIGNED_LOAD32(s1_start);
    let v1 = BROTLI_UNALIGNED_LOAD32(s2_start);
    let beyond_ok = s1_start[4] != s2_start[4];
    if v0 != v1 {
        return 0;
    }
    if limit <= 4 || beyond_ok {
        return min(limit, 4);
    }
    ComplexFindMatchLengthWithLimit(s1_rest, s2_rest, limit - 5) + 5
}
#[inline]
pub fn ComplexFindMatchLengthWithLimit(mut s1: &[u8], mut s2: &[u8], mut limit: usize) -> usize {
    let mut matched: usize = 0usize;
    let mut s1_as_64: u64;
    let mut s2_as_64: u64;
    if limit >= 8 {
        sub_match8!(s1, s2, limit, matched, s1_as_64, s2_as_64);
        if limit >= 16 {
            let mut split_pair1: (&[u8], &[u8]);
            let mut split_pair2: (&[u8], &[u8]);
            {
                let mut s1_lo = [0u8; 16];
                let mut s1_hi = [0u8; 16];
                sub_match!(
                    s1,
                    s2,
                    limit,
                    matched,
                    split_pair1,
                    split_pair2,
                    s1_lo,
                    s1_hi,
                    s1_as_64,
                    s2_as_64,
                    16
                );
            }
            if limit >= 32 {
                let mut s1_lo_a = [0u8; 128];
                let mut s1_hi_a = [0u8; 128];
                sub_match!(
                    s1,
                    s2,
                    limit,
                    matched,
                    split_pair1,
                    split_pair2,
                    s1_lo_a,
                    s1_hi_a,
                    s1_as_64,
                    s2_as_64,
                    32
                );
                if limit >= 64 {
                    sub_match!(
                        s1,
                        s2,
                        limit,
                        matched,
                        split_pair1,
                        split_pair2,
                        s1_lo_a,
                        s1_hi_a,
                        s1_as_64,
                        s2_as_64,
                        64
                    );
                    while limit >= 128 {
                        sub_match!(
                            s1,
                            s2,
                            limit,
                            matched,
                            split_pair1,
                            split_pair2,
                            s1_lo_a,
                            s1_hi_a,
                            s1_as_64,
                            s2_as_64,
                            128
                        );
                    }
                }
            }
        }
        while limit >= 8 {
            sub_match8!(s1, s2, limit, matched, s1_as_64, s2_as_64);
        }
    }
    assert!(s1.len() >= (limit & 7usize));
    assert!(s2.len() >= (limit & 7usize));
    for index in 0..(limit & 7usize) {
        if s1[index] != s2[index] {
            return matched + index;
        }
    }
    matched + (limit & 7usize) // made it through the loop
}

#[allow(unused)]
pub fn slowFindMatchLengthWithLimit(s1: &[u8], s2: &[u8], limit: usize) -> usize {
    for (index, it) in s1[..limit].iter().zip(s2[..limit].iter()).enumerate() {
        if it.0 != it.1 {
            return index;
        }
    }
    limit
}

pub fn IsMatch(dictionary: &BrotliDictionary, w: DictWord, data: &[u8], max_length: usize) -> i32 {
    if w.l as usize > max_length {
        0
    } else {
        let offset: usize = (dictionary.offsets_by_length[w.l as usize] as usize)
            .wrapping_add((w.len() as usize).wrapping_mul(w.idx() as usize));
        let dict = &dictionary.data.split_at(offset).1;
        if w.transform() as i32 == 0i32 {
            if FindMatchLengthWithLimit(dict, data, w.l as usize) == w.l as usize {
                1
            } else {
                0
            }
        } else if w.transform() as i32 == 10i32 {
            if dict[0] as i32 >= b'a' as i32
                && (dict[0] as i32 <= b'z' as i32)
                && (dict[0] as i32 ^ 32i32 == data[0] as i32)
                && (FindMatchLengthWithLimit(
                    dict.split_at(1).1,
                    data.split_at(1).1,
                    (w.len() as u32).wrapping_sub(1) as usize,
                ) == (w.len() as u32).wrapping_sub(1) as usize)
            {
                1
            } else {
                0
            }
        } else {
            for i in 0usize..w.len() as usize {
                if dict[i] as i32 >= b'a' as i32 && (dict[i] as i32 <= b'z' as i32) {
                    if dict[i] as i32 ^ 32i32 != data[i] as i32 {
                        return 0;
                    }
                } else if dict[i] as i32 != data[i] as i32 {
                    return 0;
                }
            }
            1
        }
    }
}

#[allow(unused)]
fn AddMatch(distance: usize, len: usize, len_code: usize, mut matches: &mut [u32]) {
    let match_: u32 = (distance << 5).wrapping_add(len_code) as u32;
    matches[len] = min(matches[len], match_);
}

#[allow(unused)]
fn DictMatchLength(
    dictionary: &BrotliDictionary,
    data: &[u8],
    id: usize,
    len: usize,
    maxlen: usize,
) -> usize {
    let offset: usize =
        (dictionary.offsets_by_length[len] as usize).wrapping_add(len.wrapping_mul(id));
    FindMatchLengthWithLimit(dictionary.data.split_at(offset).1, data, min(len, maxlen))
}

#[allow(unused)]
pub fn BrotliFindAllStaticDictionaryMatches(
    dictionary: &BrotliDictionary,
    data: &[u8],
    min_length: usize,
    max_length: usize,
    mut matches: &mut [u32],
) -> i32 {
    let mut has_found_match: i32 = 0i32;
    {
        let mut offset: usize = kStaticDictionaryBuckets[Hash(data) as usize] as usize;
        let mut end: i32 = (offset == 0) as i32;
        while end == 0 {
            let mut w: DictWord = kStaticDictionaryWords[offset];
            offset = offset.wrapping_add(1);
            let l: usize = (w.len() as i32 & 0x1fi32) as usize;
            let n: usize = 1usize << dictionary.size_bits_by_length[l] as i32;
            let id: usize = w.idx() as usize;
            end = !(w.len() as i32 & 0x80i32 == 0) as i32;
            w.l = l as u8;
            if w.transform() as i32 == 0i32 {
                let matchlen: usize = DictMatchLength(dictionary, data, id, l, max_length);

                let mut minlen: usize;

                let mut len: usize;
                if matchlen == l {
                    //eprint!("Adding match {} {} {}\n", w.len(), w.transform(), w.idx());
                    AddMatch(id, l, l, matches);
                    has_found_match = 1i32;
                }
                if matchlen >= l.wrapping_sub(1) {
                    //eprint!("Bdding match {} {} {}\n", w.len(), w.transform(), w.idx());
                    AddMatch(
                        id.wrapping_add((12usize).wrapping_mul(n)),
                        l.wrapping_sub(1),
                        l,
                        matches,
                    );
                    if l.wrapping_add(2) < max_length
                        && (data[(l.wrapping_sub(1) as usize)] as i32 == b'i' as i32)
                        && (data[(l as usize)] as i32 == b'n' as i32)
                        && (data[(l.wrapping_add(1) as usize)] as i32 == b'g' as i32)
                        && (data[(l.wrapping_add(2) as usize)] as i32 == b' ' as i32)
                    {
                        //eprint!("Cdding match {} {} {}\n", w.len(), w.transform(), w.idx());
                        AddMatch(
                            id.wrapping_add((49usize).wrapping_mul(n)),
                            l.wrapping_add(3),
                            l,
                            matches,
                        );
                    }
                    has_found_match = 1i32;
                }
                minlen = min_length;
                if l > 9usize {
                    minlen = max(minlen, l.wrapping_sub(9));
                }
                let maxlen: usize = min(matchlen, l.wrapping_sub(2));
                for len in minlen..=maxlen {
                    //eprint!("Ddding match {} {} {} {}\n", w.len(), w.transform(), w.idx(), len);
                    AddMatch(
                        id.wrapping_add(
                            (kOmitLastNTransforms[l.wrapping_sub(len)] as usize).wrapping_mul(n),
                        ),
                        len,
                        l,
                        matches,
                    );
                    has_found_match = 1i32;
                }

                if matchlen < l || l.wrapping_add(6) >= max_length {
                    continue;
                }
                let s: &[u8] = data.split_at(l as usize).1;
                if s[0] as i32 == b' ' as i32 {
                    //eprint!("Edding match {} {} {} {}\n", w.len(), w.transform(), w.idx(), len);
                    AddMatch(id.wrapping_add(n), l.wrapping_add(1), l, matches);
                    if s[1] as i32 == b'a' as i32 {
                        if s[2] as i32 == b' ' as i32 {
                            //eprint!("Fdding match {} {} {} {}\n", w.len(), w.transform(), w.idx(), len);
                            AddMatch(
                                id.wrapping_add((28usize).wrapping_mul(n)),
                                l.wrapping_add(3),
                                l,
                                matches,
                            );
                        } else if s[2] as i32 == b's' as i32 {
                            if s[3] as i32 == b' ' as i32 {
                                //eprint!("Gdding match {} {} {} {}\n", w.len(), w.transform(), w.idx(), len);
                                AddMatch(
                                    id.wrapping_add((46usize).wrapping_mul(n)),
                                    l.wrapping_add(4),
                                    l,
                                    matches,
                                );
                            }
                        } else if s[2] as i32 == b't' as i32 {
                            if s[3] as i32 == b' ' as i32 {
                                //eprint!("Hdding match {} {} {} {}\n", w.len(), w.transform(), w.idx(), len);
                                AddMatch(
                                    id.wrapping_add((60usize).wrapping_mul(n)),
                                    l.wrapping_add(4),
                                    l,
                                    matches,
                                );
                            }
                        } else if s[2] as i32 == b'n' as i32
                            && s[3] as i32 == b'd' as i32
                            && (s[4] as i32 == b' ' as i32)
                        {
                            //eprint!("Idding match {} {} {} {}\n", w.len(), w.transform(), w.idx(), len);
                            AddMatch(
                                id.wrapping_add((10usize).wrapping_mul(n)),
                                l.wrapping_add(5),
                                l,
                                matches,
                            );
                        }
                    } else if s[1] as i32 == b'b' as i32 {
                        if s[2] as i32 == b'y' as i32 && (s[3] as i32 == b' ' as i32) {
                            //eprint!("Jdding match {} {} {} {}\n", w.len(), w.transform(), w.idx(), len);
                            AddMatch(
                                id.wrapping_add((38usize).wrapping_mul(n)),
                                l.wrapping_add(4),
                                l,
                                matches,
                            );
                        }
                    } else if s[1] as i32 == b'i' as i32 {
                        if s[2] as i32 == b'n' as i32 {
                            if s[3] as i32 == b' ' as i32 {
                                //eprint!("Kdding match {} {} {} {}\n", w.len(), w.transform(), w.idx(), len);
                                AddMatch(
                                    id.wrapping_add((16usize).wrapping_mul(n)),
                                    l.wrapping_add(4),
                                    l,
                                    matches,
                                );
                            }
                        } else if s[2] as i32 == b's' as i32 && s[3] as i32 == b' ' as i32 {
                            //eprint!("Ldding match {} {} {} {}\n", w.len(), w.transform(), w.idx(), len);
                            AddMatch(
                                id.wrapping_add((47usize).wrapping_mul(n)),
                                l.wrapping_add(4),
                                l,
                                matches,
                            );
                        }
                    } else if s[1] as i32 == b'f' as i32 {
                        if s[2] as i32 == b'o' as i32 {
                            if s[3] as i32 == b'r' as i32 && (s[4] as i32 == b' ' as i32) {
                                //eprint!("Mdding match {} {} {} {}\n", w.len(), w.transform(), w.idx(), len);
                                AddMatch(
                                    id.wrapping_add((25usize).wrapping_mul(n)),
                                    l.wrapping_add(5),
                                    l,
                                    matches,
                                );
                            }
                        } else if s[2] as i32 == b'r' as i32
                            && s[3] as i32 == b'o' as i32
                            && (s[4] as i32 == b'm' as i32)
                            && (s[5] as i32 == b' ' as i32)
                        {
                            //eprint!("Ndding match {} {} {} {}\n", w.len(), w.transform(), w.idx(), len);
                            AddMatch(
                                id.wrapping_add((37usize).wrapping_mul(n)),
                                l.wrapping_add(6),
                                l,
                                matches,
                            );
                        }
                    } else if s[1] as i32 == b'o' as i32 {
                        if s[2] as i32 == b'f' as i32 {
                            if s[3] as i32 == b' ' as i32 {
                                //eprint!("Odding match {} {} {} {}\n", w.len(), w.transform(), w.idx(), len);
                                AddMatch(
                                    id.wrapping_add((8usize).wrapping_mul(n)),
                                    l.wrapping_add(4),
                                    l,
                                    matches,
                                );
                            }
                        } else if s[2] as i32 == b'n' as i32 && s[3] as i32 == b' ' as i32 {
                            //eprint!("Pdding match {} {} {} {}\n", w.len(), w.transform(), w.idx(), len);
                            AddMatch(
                                id.wrapping_add((45usize).wrapping_mul(n)),
                                l.wrapping_add(4),
                                l,
                                matches,
                            );
                        }
                    } else if s[1] as i32 == b'n' as i32 {
                        if s[2] as i32 == b'o' as i32
                            && (s[3] as i32 == b't' as i32)
                            && (s[4] as i32 == b' ' as i32)
                        {
                            //eprint!("Qdding match {} {} {} {}\n", w.len(), w.transform(), w.idx(), len);
                            AddMatch(
                                id.wrapping_add((80usize).wrapping_mul(n)),
                                l.wrapping_add(5),
                                l,
                                matches,
                            );
                        }
                    } else if s[1] as i32 == b't' as i32 {
                        if s[2] as i32 == b'h' as i32 {
                            if s[3] as i32 == b'e' as i32 {
                                if s[4] as i32 == b' ' as i32 {
                                    //eprint!("Rdding match {} {} {} {}\n", w.len(), w.transform(), w.idx(), len);
                                    AddMatch(
                                        id.wrapping_add((5usize).wrapping_mul(n)),
                                        l.wrapping_add(5),
                                        l,
                                        matches,
                                    );
                                }
                            } else if s[3] as i32 == b'a' as i32
                                && s[4] as i32 == b't' as i32
                                && (s[5] as i32 == b' ' as i32)
                            {
                                //eprint!("Sdding match {} {} {} {}\n", w.len(), w.transform(), w.idx(), len);
                                AddMatch(
                                    id.wrapping_add((29usize).wrapping_mul(n)),
                                    l.wrapping_add(6),
                                    l,
                                    matches,
                                );
                            }
                        } else if s[2] as i32 == b'o' as i32 && s[3] as i32 == b' ' as i32 {
                            //eprint!("Tdding match {} {} {} {}\n", w.len(), w.transform(), w.idx(), len);
                            AddMatch(
                                id.wrapping_add((17usize).wrapping_mul(n)),
                                l.wrapping_add(4),
                                l,
                                matches,
                            );
                        }
                    } else if s[1] as i32 == b'w' as i32
                        && s[2] as i32 == b'i' as i32
                        && (s[3] as i32 == b't' as i32)
                        && (s[4] as i32 == b'h' as i32)
                        && (s[5] as i32 == b' ' as i32)
                    {
                        //eprint!("Udding match {} {} {} {}\n", w.len(), w.transform(), w.idx(), len);
                        AddMatch(
                            id.wrapping_add((35usize).wrapping_mul(n)),
                            l.wrapping_add(6),
                            l,
                            matches,
                        );
                    }
                } else if s[0] as i32 == b'\"' as i32 {
                    //eprint!("Vdding match {} {} {} {}\n", w.len(), w.transform(), w.idx(), len);
                    AddMatch(
                        id.wrapping_add((19usize).wrapping_mul(n)),
                        l.wrapping_add(1),
                        l,
                        matches,
                    );
                    if s[1] as i32 == b'>' as i32 {
                        //eprint!("Wdding match {} {} {} {}\n", w.len(), w.transform(), w.idx(), len);
                        AddMatch(
                            id.wrapping_add((21usize).wrapping_mul(n)),
                            l.wrapping_add(2),
                            l,
                            matches,
                        );
                    }
                } else if s[0] as i32 == b'.' as i32 {
                    //eprint!("Xdding match {} {} {} {}\n", w.len(), w.transform(), w.idx(), len);
                    AddMatch(
                        id.wrapping_add((20usize).wrapping_mul(n)),
                        l.wrapping_add(1),
                        l,
                        matches,
                    );
                    if s[1] as i32 == b' ' as i32 {
                        //eprint!("Ydding match {} {} {} {}\n", w.len(), w.transform(), w.idx(), len);
                        AddMatch(
                            id.wrapping_add((31usize).wrapping_mul(n)),
                            l.wrapping_add(2),
                            l,
                            matches,
                        );
                        if s[2] as i32 == b'T' as i32 && (s[3] as i32 == b'h' as i32) {
                            if s[4] as i32 == b'e' as i32 {
                                if s[5] as i32 == b' ' as i32 {
                                    //eprint!("Zdding match {} {} {} {}\n", w.len(), w.transform(), w.idx(), len);
                                    AddMatch(
                                        id.wrapping_add((43usize).wrapping_mul(n)),
                                        l.wrapping_add(6),
                                        l,
                                        matches,
                                    );
                                }
                            } else if s[4] as i32 == b'i' as i32
                                && s[5] as i32 == b's' as i32
                                && (s[6] as i32 == b' ' as i32)
                            {
                                //eprint!("AAdding match {} {} {} {}\n", w.len(), w.transform(), w.idx(), len);
                                AddMatch(
                                    id.wrapping_add((75usize).wrapping_mul(n)),
                                    l.wrapping_add(7),
                                    l,
                                    matches,
                                );
                            }
                        }
                    }
                } else if s[0] as i32 == b',' as i32 {
                    //eprint!("ABdding match {} {} {} {}\n", w.len(), w.transform(), w.idx(), len);
                    AddMatch(
                        id.wrapping_add((76usize).wrapping_mul(n)),
                        l.wrapping_add(1),
                        l,
                        matches,
                    );
                    if s[1] as i32 == b' ' as i32 {
                        //eprint!("ACdding match {} {} {} {}\n", w.len(), w.transform(), w.idx(), len);
                        AddMatch(
                            id.wrapping_add((14usize).wrapping_mul(n)),
                            l.wrapping_add(2),
                            l,
                            matches,
                        );
                    }
                } else if s[0] as i32 == b'\n' as i32 {
                    //eprint!("ADdding match {} {} {} {}\n", w.len(), w.transform(), w.idx(), len);
                    AddMatch(
                        id.wrapping_add((22usize).wrapping_mul(n)),
                        l.wrapping_add(1),
                        l,
                        matches,
                    );
                    if s[1] as i32 == b'\t' as i32 {
                        //eprint!("AEdding match {} {} {} {}\n", w.len(), w.transform(), w.idx(), len);
                        AddMatch(
                            id.wrapping_add((50usize).wrapping_mul(n)),
                            l.wrapping_add(2),
                            l,
                            matches,
                        );
                    }
                } else if s[0] as i32 == b']' as i32 {
                    //eprint!("AFdding match {} {} {} {}\n", w.len(), w.transform(), w.idx(), len);
                    AddMatch(
                        id.wrapping_add((24usize).wrapping_mul(n)),
                        l.wrapping_add(1),
                        l,
                        matches,
                    );
                } else if s[0] as i32 == b'\'' as i32 {
                    //eprint!("AGdding match {} {} {} {}\n", w.len(), w.transform(), w.idx(), len);
                    AddMatch(
                        id.wrapping_add((36usize).wrapping_mul(n)),
                        l.wrapping_add(1),
                        l,
                        matches,
                    );
                } else if s[0] as i32 == b':' as i32 {
                    //eprint!("AHdding match {} {} {} {}\n", w.len(), w.transform(), w.idx(), len);
                    AddMatch(
                        id.wrapping_add((51usize).wrapping_mul(n)),
                        l.wrapping_add(1),
                        l,
                        matches,
                    );
                } else if s[0] as i32 == b'(' as i32 {
                    //eprint!("AIdding match {} {} {} {}\n", w.len(), w.transform(), w.idx(), len);
                    AddMatch(
                        id.wrapping_add((57usize).wrapping_mul(n)),
                        l.wrapping_add(1),
                        l,
                        matches,
                    );
                } else if s[0] as i32 == b'=' as i32 {
                    if s[1] as i32 == b'\"' as i32 {
                        //eprint!("AJdding match {} {} {} {}\n", w.len(), w.transform(), w.idx(), len);
                        AddMatch(
                            id.wrapping_add((70usize).wrapping_mul(n)),
                            l.wrapping_add(2),
                            l,
                            matches,
                        );
                    } else if s[1] as i32 == b'\'' as i32 {
                        //eprint!("AKdding match {} {} {} {}\n", w.len(), w.transform(), w.idx(), len);
                        AddMatch(
                            id.wrapping_add((86usize).wrapping_mul(n)),
                            l.wrapping_add(2),
                            l,
                            matches,
                        );
                    }
                } else if s[0] as i32 == b'a' as i32 {
                    if s[1] as i32 == b'l' as i32 && (s[2] as i32 == b' ' as i32) {
                        //eprint!("ALdding match {} {} {} {}\n", w.len(), w.transform(), w.idx(), len);
                        AddMatch(
                            id.wrapping_add((84usize).wrapping_mul(n)),
                            l.wrapping_add(3),
                            l,
                            matches,
                        );
                    }
                } else if s[0] as i32 == b'e' as i32 {
                    if s[1] as i32 == b'd' as i32 {
                        if s[2] as i32 == b' ' as i32 {
                            //eprint!("AMdding match {} {} {} {}\n", w.len(), w.transform(), w.idx(), len);
                            AddMatch(
                                id.wrapping_add((53usize).wrapping_mul(n)),
                                l.wrapping_add(3),
                                l,
                                matches,
                            );
                        }
                    } else if s[1] as i32 == b'r' as i32 {
                        if s[2] as i32 == b' ' as i32 {
                            //eprint!("ANdding match {} {} {} {}\n", w.len(), w.transform(), w.idx(), len);
                            AddMatch(
                                id.wrapping_add((82usize).wrapping_mul(n)),
                                l.wrapping_add(3),
                                l,
                                matches,
                            );
                        }
                    } else if s[1] as i32 == b's' as i32
                        && s[2] as i32 == b't' as i32
                        && (s[3] as i32 == b' ' as i32)
                    {
                        //eprint!("AOdding match {} {} {} {}\n", w.len(), w.transform(), w.idx(), len);
                        AddMatch(
                            id.wrapping_add((95usize).wrapping_mul(n)),
                            l.wrapping_add(4),
                            l,
                            matches,
                        );
                    }
                } else if s[0] as i32 == b'f' as i32 {
                    if s[1] as i32 == b'u' as i32
                        && (s[2] as i32 == b'l' as i32)
                        && (s[3] as i32 == b' ' as i32)
                    {
                        //eprint!("APdding match {} {} {} {}\n", w.len(), w.transform(), w.idx(), len);
                        AddMatch(
                            id.wrapping_add((90usize).wrapping_mul(n)),
                            l.wrapping_add(4),
                            l,
                            matches,
                        );
                    }
                } else if s[0] as i32 == b'i' as i32 {
                    if s[1] as i32 == b'v' as i32 {
                        if s[2] as i32 == b'e' as i32 && (s[3] as i32 == b' ' as i32) {
                            //eprint!("AQdding match {} {} {} {}\n", w.len(), w.transform(), w.idx(), len);
                            AddMatch(
                                id.wrapping_add((92usize).wrapping_mul(n)),
                                l.wrapping_add(4),
                                l,
                                matches,
                            );
                        }
                    } else if s[1] as i32 == b'z' as i32
                        && s[2] as i32 == b'e' as i32
                        && (s[3] as i32 == b' ' as i32)
                    {
                        //eprint!("ARdding match {} {} {} {}\n", w.len(), w.transform(), w.idx(), len);
                        AddMatch(
                            id.wrapping_add((100usize).wrapping_mul(n)),
                            l.wrapping_add(4),
                            l,
                            matches,
                        );
                    }
                } else if s[0] as i32 == b'l' as i32 {
                    if s[1] as i32 == b'e' as i32 {
                        if s[2] as i32 == b's' as i32
                            && (s[3] as i32 == b's' as i32)
                            && (s[4] as i32 == b' ' as i32)
                        {
                            //eprint!("ASdding match {} {} {} {}\n", w.len(), w.transform(), w.idx(), len);
                            AddMatch(
                                id.wrapping_add((93usize).wrapping_mul(n)),
                                l.wrapping_add(5),
                                l,
                                matches,
                            );
                        }
                    } else if s[1] as i32 == b'y' as i32 && s[2] as i32 == b' ' as i32 {
                        //eprint!("ATdding match {} {} {} {}\n", w.len(), w.transform(), w.idx(), len);
                        AddMatch(
                            id.wrapping_add((61usize).wrapping_mul(n)),
                            l.wrapping_add(3),
                            l,
                            matches,
                        );
                    }
                } else if s[0] as i32 == b'o' as i32
                    && s[1] as i32 == b'u' as i32
                    && (s[2] as i32 == b's' as i32)
                    && (s[3] as i32 == b' ' as i32)
                {
                    //eprint!("AUdding match {} {} {} {}\n", w.len(), w.transform(), w.idx(), len);
                    AddMatch(
                        id.wrapping_add((106usize).wrapping_mul(n)),
                        l.wrapping_add(4),
                        l,
                        matches,
                    );
                }
            } else {
                let is_all_caps = w.transform() != kUppercaseFirst;

                if IsMatch(dictionary, w, data, max_length) == 0 {
                    continue;
                }
                //eprint!("AVdding match {} {} {} {}\n", w.len(), w.transform(), w.idx(), 666);
                AddMatch(
                    id.wrapping_add((if is_all_caps { 44usize } else { 9usize }).wrapping_mul(n)),
                    l,
                    l,
                    matches,
                );
                has_found_match = 1i32;
                if l.wrapping_add(1) >= max_length {
                    continue;
                }
                let s: &[u8] = data.split_at(l as usize).1;
                if s[0] as i32 == b' ' as i32 {
                    //eprint!("AWdding match {} {} {} {}\n", w.len(), w.transform(), w.idx(), 666);
                    AddMatch(
                        id.wrapping_add(
                            (if is_all_caps { 68usize } else { 4usize }).wrapping_mul(n),
                        ),
                        l.wrapping_add(1),
                        l,
                        matches,
                    );
                } else if s[0] as i32 == b'\"' as i32 {
                    //eprint!("AXdding match {} {} {} {}\n", w.len(), w.transform(), w.idx(), 666);
                    AddMatch(
                        id.wrapping_add(
                            (if is_all_caps { 87usize } else { 66usize }).wrapping_mul(n),
                        ),
                        l.wrapping_add(1),
                        l,
                        matches,
                    );
                    if s[1] as i32 == b'>' as i32 {
                        //eprint!("AYdding match {} {} {} {}\n", w.len(), w.transform(), w.idx(), 666);
                        AddMatch(
                            id.wrapping_add(
                                (if is_all_caps { 97usize } else { 69usize }).wrapping_mul(n),
                            ),
                            l.wrapping_add(2),
                            l,
                            matches,
                        );
                    }
                } else if s[0] as i32 == b'.' as i32 {
                    //eprint!("AZdding match {} {} {} {}\n", w.len(), w.transform(), w.idx(), 666);
                    AddMatch(
                        id.wrapping_add(
                            (if is_all_caps { 101usize } else { 79usize }).wrapping_mul(n),
                        ),
                        l.wrapping_add(1),
                        l,
                        matches,
                    );
                    if s[1] as i32 == b' ' as i32 {
                        //eprint!("BAdding match {} {} {} {}\n", w.len(), w.transform(), w.idx(), 666);
                        AddMatch(
                            id.wrapping_add(
                                (if is_all_caps { 114usize } else { 88usize }).wrapping_mul(n),
                            ),
                            l.wrapping_add(2),
                            l,
                            matches,
                        );
                    }
                } else if s[0] as i32 == b',' as i32 {
                    //eprint!("BBdding match {} {} {} {}\n", w.len(), w.transform(), w.idx(), 666);
                    AddMatch(
                        id.wrapping_add(
                            (if is_all_caps { 112usize } else { 99usize }).wrapping_mul(n),
                        ),
                        l.wrapping_add(1),
                        l,
                        matches,
                    );
                    if s[1] as i32 == b' ' as i32 {
                        //eprint!("BCdding match {} {} {} {}\n", w.len(), w.transform(), w.idx(), 666);
                        AddMatch(
                            id.wrapping_add(
                                (if is_all_caps { 107usize } else { 58usize }).wrapping_mul(n),
                            ),
                            l.wrapping_add(2),
                            l,
                            matches,
                        );
                    }
                } else if s[0] as i32 == b'\'' as i32 {
                    //eprint!("BDdding match {} {} {} {}\n", w.len(), w.transform(), w.idx(), 666);
                    AddMatch(
                        id.wrapping_add(
                            (if is_all_caps { 94usize } else { 74usize }).wrapping_mul(n),
                        ),
                        l.wrapping_add(1),
                        l,
                        matches,
                    );
                } else if s[0] as i32 == b'(' as i32 {
                    //eprint!("BEdding match {} {} {} {}\n", w.len(), w.transform(), w.idx(), 666);
                    AddMatch(
                        id.wrapping_add(
                            (if is_all_caps { 113usize } else { 78usize }).wrapping_mul(n),
                        ),
                        l.wrapping_add(1),
                        l,
                        matches,
                    );
                } else if s[0] as i32 == b'=' as i32 {
                    if s[1] as i32 == b'\"' as i32 {
                        //eprint!("BFdding match {} {} {} {}\n", w.len(), w.transform(), w.idx(), 666);
                        AddMatch(
                            id.wrapping_add(
                                (if is_all_caps { 105usize } else { 104usize }).wrapping_mul(n),
                            ),
                            l.wrapping_add(2),
                            l,
                            matches,
                        );
                    } else if s[1] as i32 == b'\'' as i32 {
                        //eprint!("BGdding match {} {} {} {}\n", w.len(), w.transform(), w.idx(), 666);
                        AddMatch(
                            id.wrapping_add(
                                (if is_all_caps { 116usize } else { 108usize }).wrapping_mul(n),
                            ),
                            l.wrapping_add(2),
                            l,
                            matches,
                        );
                    }
                }
            }
        }
    }
    if max_length >= 5usize && (data[0] as i32 == b' ' as i32 || data[0] as i32 == b'.' as i32) {
        let is_space = data[0] == b' ';
        let mut offset: usize =
            kStaticDictionaryBuckets[Hash(data.split_at(1).1) as usize] as usize;
        let mut end: i32 = (offset == 0) as i32;
        while end == 0 {
            let mut w: DictWord = kStaticDictionaryWords[offset];
            offset = offset.wrapping_add(1);
            let l: usize = (w.len() as i32 & 0x1fi32) as usize;
            let n: usize = 1usize << dictionary.size_bits_by_length[l] as i32;
            let id: usize = w.idx() as usize;
            end = !(w.len() as i32 & 0x80i32 == 0) as i32;
            w.l = l as u8;
            if w.transform() as i32 == 0i32 {
                if IsMatch(
                    dictionary,
                    w,
                    data.split_at(1).1,
                    max_length.wrapping_sub(1),
                ) == 0
                {
                    continue;
                }
                //eprint!("BHdding match {} {} {} {}\n", w.len(), w.transform(), w.idx(), 666);
                AddMatch(
                    id.wrapping_add((if is_space { 6usize } else { 32usize }).wrapping_mul(n)),
                    l.wrapping_add(1),
                    l,
                    matches,
                );
                has_found_match = 1i32;
                if l.wrapping_add(2) >= max_length {
                    continue;
                }
                let s: &[u8] = data.split_at(l.wrapping_add(1) as usize).1;
                if s[0] as i32 == b' ' as i32 {
                    //eprint!("BIdding match {} {} {} {}\n", w.len(), w.transform(), w.idx(), 666);
                    AddMatch(
                        id.wrapping_add((if is_space { 2usize } else { 77usize }).wrapping_mul(n)),
                        l.wrapping_add(2),
                        l,
                        matches,
                    );
                } else if s[0] as i32 == b'(' as i32 {
                    //eprint!("BJdding match {} {} {} {}\n", w.len(), w.transform(), w.idx(), 666);
                    AddMatch(
                        id.wrapping_add((if is_space { 89usize } else { 67usize }).wrapping_mul(n)),
                        l.wrapping_add(2),
                        l,
                        matches,
                    );
                } else if is_space {
                    if s[0] as i32 == b',' as i32 {
                        //eprint!("BKdding match {} {} {} {}\n", w.len(), w.transform(), w.idx(), 666);
                        AddMatch(
                            id.wrapping_add((103usize).wrapping_mul(n)),
                            l.wrapping_add(2),
                            l,
                            matches,
                        );
                        if s[1] as i32 == b' ' as i32 {
                            //eprint!("BLdding match {} {} {} {}\n", w.len(), w.transform(), w.idx(), 666);
                            AddMatch(
                                id.wrapping_add((33usize).wrapping_mul(n)),
                                l.wrapping_add(3),
                                l,
                                matches,
                            );
                        }
                    } else if s[0] as i32 == b'.' as i32 {
                        //eprint!("BMdding match {} {} {} {}\n", w.len(), w.transform(), w.idx(), 666);
                        AddMatch(
                            id.wrapping_add((71usize).wrapping_mul(n)),
                            l.wrapping_add(2),
                            l,
                            matches,
                        );
                        if s[1] as i32 == b' ' as i32 {
                            //eprint!("BNdding match {} {} {} {}\n", w.len(), w.transform(), w.idx(), 666);
                            AddMatch(
                                id.wrapping_add((52usize).wrapping_mul(n)),
                                l.wrapping_add(3),
                                l,
                                matches,
                            );
                        }
                    } else if s[0] as i32 == b'=' as i32 {
                        if s[1] as i32 == b'\"' as i32 {
                            //eprint!("BOdding match {} {} {} {}\n", w.len(), w.transform(), w.idx(), 666);
                            AddMatch(
                                id.wrapping_add((81usize).wrapping_mul(n)),
                                l.wrapping_add(3),
                                l,
                                matches,
                            );
                        } else if s[1] as i32 == b'\'' as i32 {
                            //eprint!("BPdding match {} {} {} {}\n", w.len(), w.transform(), w.idx(), 666);
                            AddMatch(
                                id.wrapping_add((98usize).wrapping_mul(n)),
                                l.wrapping_add(3),
                                l,
                                matches,
                            );
                        }
                    }
                }
            } else if is_space {
                let is_all_caps = w.transform() != kUppercaseFirst;

                if IsMatch(
                    dictionary,
                    w,
                    data.split_at(1).1,
                    max_length.wrapping_sub(1),
                ) == 0
                {
                    continue;
                }
                //eprint!("CAdding match {} {} {} {}\n", w.len(), w.transform(), w.idx(), 666);
                AddMatch(
                    id.wrapping_add((if is_all_caps { 85usize } else { 30usize }).wrapping_mul(n)),
                    l.wrapping_add(1),
                    l,
                    matches,
                );
                has_found_match = 1i32;
                if l.wrapping_add(2) >= max_length {
                    continue;
                }
                let s: &[u8] = data.split_at(l.wrapping_add(1)).1;
                if s[0] as i32 == b' ' as i32 {
                    //eprint!("CBdding match {} {} {} {}\n", w.len(), w.transform(), w.idx(), 666);
                    AddMatch(
                        id.wrapping_add(
                            (if is_all_caps { 83usize } else { 15usize }).wrapping_mul(n),
                        ),
                        l.wrapping_add(2),
                        l,
                        matches,
                    );
                } else if s[0] as i32 == b',' as i32 {
                    if !is_all_caps {
                        //eprint!("CCdding match {} {} {} {}\n", w.len(), w.transform(), w.idx(), 666);
                        AddMatch(
                            id.wrapping_add((109usize).wrapping_mul(n)),
                            l.wrapping_add(2),
                            l,
                            matches,
                        );
                    }
                    if s[1] as i32 == b' ' as i32 {
                        //eprint!("CDdding match {} {} {} {}\n", w.len(), w.transform(), w.idx(), 666);
                        AddMatch(
                            id.wrapping_add(
                                (if is_all_caps { 111usize } else { 65usize }).wrapping_mul(n),
                            ),
                            l.wrapping_add(3),
                            l,
                            matches,
                        );
                    }
                } else if s[0] as i32 == b'.' as i32 {
                    //eprint!("CEdding match {} {} {} {}\n", w.len(), w.transform(), w.idx(), 666);
                    AddMatch(
                        id.wrapping_add(
                            (if is_all_caps { 115usize } else { 96usize }).wrapping_mul(n),
                        ),
                        l.wrapping_add(2),
                        l,
                        matches,
                    );
                    if s[1] as i32 == b' ' as i32 {
                        //eprint!("CFdding match {} {} {} {}\n", w.len(), w.transform(), w.idx(), 666);
                        AddMatch(
                            id.wrapping_add(
                                (if is_all_caps { 117usize } else { 91usize }).wrapping_mul(n),
                            ),
                            l.wrapping_add(3),
                            l,
                            matches,
                        );
                    }
                } else if s[0] as i32 == b'=' as i32 {
                    if s[1] as i32 == b'\"' as i32 {
                        //eprint!("CGdding match {} {} {} {}\n", w.len(), w.transform(), w.idx(), 666);
                        AddMatch(
                            id.wrapping_add(
                                (if is_all_caps { 110usize } else { 118usize }).wrapping_mul(n),
                            ),
                            l.wrapping_add(3),
                            l,
                            matches,
                        );
                    } else if s[1] as i32 == b'\'' as i32 {
                        //eprint!("CHdding match {} {} {} {}\n", w.len(), w.transform(), w.idx(), 666);
                        AddMatch(
                            id.wrapping_add(
                                (if is_all_caps { 119usize } else { 120usize }).wrapping_mul(n),
                            ),
                            l.wrapping_add(3),
                            l,
                            matches,
                        );
                    }
                }
            }
        }
    }
    if max_length >= 6usize
        && (data[1] as i32 == b' ' as i32
            && (data[0] as i32 == b'e' as i32
                || data[0] as i32 == b's' as i32
                || data[0] as i32 == b',' as i32)
            || data[0] as i32 == 0xc2i32 && (data[1] as i32 == 0xa0i32))
    {
        let mut offset: usize =
            kStaticDictionaryBuckets[Hash(data.split_at(2).1) as usize] as usize;
        let mut end: i32 = (offset == 0) as i32;
        while end == 0 {
            let mut w: DictWord = kStaticDictionaryWords[offset];
            offset = offset.wrapping_add(1);
            let l: usize = (w.len() as i32 & 0x1fi32) as usize;
            let n: usize = 1usize << dictionary.size_bits_by_length[l] as i32;
            let id: usize = w.idx() as usize;
            end = !(w.len() as i32 & 0x80i32 == 0) as i32;
            w.l = l as u8;
            if w.transform() as i32 == 0i32
                && (IsMatch(
                    dictionary,
                    w,
                    data.split_at(2).1,
                    max_length.wrapping_sub(2),
                ) != 0)
            {
                if data[0] as i32 == 0xc2i32 {
                    //eprint!("CIdding match {} {} {} {}\n", w.len(), w.transform(), w.idx(), 666);
                    AddMatch(
                        id.wrapping_add((102usize).wrapping_mul(n)),
                        l.wrapping_add(2),
                        l,
                        matches,
                    );
                    has_found_match = 1i32;
                } else if l.wrapping_add(2) < max_length
                    && (data[(l.wrapping_add(2) as usize)] as i32 == b' ' as i32)
                {
                    let t: usize = (if data[0] as i32 == b'e' as i32 {
                        18i32
                    } else if data[0] as i32 == b's' as i32 {
                        7i32
                    } else {
                        13i32
                    }) as usize;
                    //eprint!("CJdding match {} {} {} {}\n", w.len(), w.transform(), w.idx(), 666);
                    AddMatch(
                        id.wrapping_add(t.wrapping_mul(n)),
                        l.wrapping_add(3),
                        l,
                        matches,
                    );
                    has_found_match = 1i32;
                }
            }
        }
    }
    if max_length >= 9usize
        && (data[0] as i32 == b' ' as i32
            && (data[1] as i32 == b't' as i32)
            && (data[2] as i32 == b'h' as i32)
            && (data[3] as i32 == b'e' as i32)
            && (data[4] as i32 == b' ' as i32)
            || data[0] as i32 == b'.' as i32
                && (data[1] as i32 == b'c' as i32)
                && (data[2] as i32 == b'o' as i32)
                && (data[3] as i32 == b'm' as i32)
                && (data[4] as i32 == b'/' as i32))
    {
        let mut offset: usize =
            kStaticDictionaryBuckets[Hash(data.split_at(5).1) as usize] as usize;
        let mut end: i32 = (offset == 0) as i32;
        while end == 0 {
            let mut w: DictWord = kStaticDictionaryWords[offset];
            offset = offset.wrapping_add(1);
            let l: usize = (w.len() as i32 & 0x1fi32) as usize;
            let n: usize = 1usize << dictionary.size_bits_by_length[l] as i32;
            let id: usize = w.idx() as usize;
            end = !(w.len() as i32 & 0x80i32 == 0) as i32;
            w.l = l as u8;
            if w.transform() as i32 == 0i32
                && (IsMatch(
                    dictionary,
                    w,
                    data.split_at(5).1,
                    max_length.wrapping_sub(5),
                ) != 0)
            {
                //eprint!("CKdding match {} {} {} {}\n", w.len(), w.transform(), w.idx(), 666);
                AddMatch(
                    id.wrapping_add(
                        (if data[0] as i32 == b' ' as i32 {
                            41i32
                        } else {
                            72i32
                        } as usize)
                            .wrapping_mul(n),
                    ),
                    l.wrapping_add(5),
                    l,
                    matches,
                );
                has_found_match = 1i32;
                if l.wrapping_add(5) < max_length {
                    let s: &[u8] = data.split_at(l.wrapping_add(5) as usize).1;
                    if data[0] as i32 == b' ' as i32
                        && l.wrapping_add(8) < max_length
                        && (s[0] as i32 == b' ' as i32)
                        && (s[1] as i32 == b'o' as i32)
                        && (s[2] as i32 == b'f' as i32)
                        && (s[3] as i32 == b' ' as i32)
                    {
                        //eprint!("CLdding match {} {} {} {}\n", w.len(), w.transform(), w.idx(), 666);
                        AddMatch(
                            id.wrapping_add((62usize).wrapping_mul(n)),
                            l.wrapping_add(9),
                            l,
                            matches,
                        );
                        if l.wrapping_add(12) < max_length
                            && (s[4] as i32 == b't' as i32)
                            && (s[5] as i32 == b'h' as i32)
                            && (s[6] as i32 == b'e' as i32)
                            && (s[7] as i32 == b' ' as i32)
                        {
                            //eprint!("BQdding match {} {} {} {}\n", w.len(), w.transform(), w.idx(), 666);
                            AddMatch(
                                id.wrapping_add((73usize).wrapping_mul(n)),
                                l.wrapping_add(13),
                                l,
                                matches,
                            );
                        }
                    }
                }
            }
        }
    }
    has_found_match
}

#[cfg(test)]
mod test {
    #[allow(unused)]
    fn construct_situation(seed: &[u8], mut output: &mut [u8], limit: usize, matchfor: usize) {
        output[..].clone_from_slice(seed);
        if matchfor >= limit {
            return;
        }
        output[matchfor] = output[matchfor].wrapping_add((matchfor as u8 % 253u8).wrapping_add(1));
    }
    #[test]
    fn test_find_match_length() {
        let mut a = [91u8; 600000];
        let mut b = [0u8; 600000];
        for i in 1..a.len() {
            a[i] = (a[i - 1] % 19u8).wrapping_add(17);
        }
        construct_situation(&a[..], &mut b[..], a.len(), 0);
        assert_eq!(super::FindMatchLengthWithLimit(&a[..], &b[..], a.len()), 0);
        construct_situation(&a[..], &mut b[..], a.len(), 1);
        assert_eq!(super::FindMatchLengthWithLimit(&a[..], &b[..], a.len()), 1);
        construct_situation(&a[..], &mut b[..], a.len(), 10);
        assert_eq!(super::FindMatchLengthWithLimit(&a[..], &b[..], a.len()), 10);
        construct_situation(&a[..], &mut b[..], a.len(), 9);
        assert_eq!(super::FindMatchLengthWithLimit(&a[..], &b[..], a.len()), 9);
        construct_situation(&a[..], &mut b[..], a.len(), 7);
        assert_eq!(super::FindMatchLengthWithLimit(&a[..], &b[..], a.len()), 7);
        construct_situation(&a[..], &mut b[..], a.len(), 8);
        assert_eq!(super::FindMatchLengthWithLimit(&a[..], &b[..], a.len()), 8);
        construct_situation(&a[..], &mut b[..], a.len(), 48);
        assert_eq!(super::FindMatchLengthWithLimit(&a[..], &b[..], a.len()), 48);
        construct_situation(&a[..], &mut b[..], a.len(), 49);
        assert_eq!(super::FindMatchLengthWithLimit(&a[..], &b[..], a.len()), 49);
        construct_situation(&a[..], &mut b[..], a.len(), 63);
        assert_eq!(super::FindMatchLengthWithLimit(&a[..], &b[..], a.len()), 63);
        construct_situation(&a[..], &mut b[..], a.len(), 222);
        assert_eq!(
            super::FindMatchLengthWithLimit(&a[..], &b[..], a.len()),
            222
        );
        construct_situation(&a[..], &mut b[..], a.len(), 1590);
        assert_eq!(
            super::FindMatchLengthWithLimit(&a[..], &b[..], a.len()),
            1590
        );
        construct_situation(&a[..], &mut b[..], a.len(), 12590);
        assert_eq!(
            super::FindMatchLengthWithLimit(&a[..], &b[..], a.len()),
            12590
        );
        construct_situation(&a[..], &mut b[..], a.len(), 52592);
        assert_eq!(
            super::FindMatchLengthWithLimit(&a[..], &b[..], a.len()),
            52592
        );
        construct_situation(&a[..], &mut b[..], a.len(), 152592);
        assert_eq!(
            super::FindMatchLengthWithLimit(&a[..], &b[..], a.len()),
            152592
        );
        construct_situation(&a[..], &mut b[..], a.len(), 252591);
        assert_eq!(
            super::FindMatchLengthWithLimit(&a[..], &b[..], a.len()),
            252591
        );
        construct_situation(&a[..], &mut b[..], a.len(), 131072);
        assert_eq!(
            super::FindMatchLengthWithLimit(&a[..], &b[..], a.len()),
            131072
        );
        construct_situation(&a[..], &mut b[..], a.len(), 131073);
        assert_eq!(
            super::FindMatchLengthWithLimit(&a[..], &b[..], a.len()),
            131073
        );
        construct_situation(&a[..], &mut b[..], a.len(), 131072 + 64 + 32 + 16 + 8);
        assert_eq!(
            super::FindMatchLengthWithLimit(&a[..], &b[..], a.len()),
            131072 + 64 + 32 + 16 + 8
        );
        construct_situation(&a[..], &mut b[..], a.len(), 272144 + 64 + 32 + 16 + 8 + 1);
        assert_eq!(
            super::FindMatchLengthWithLimit(&a[..], &b[..], a.len()),
            272144 + 64 + 32 + 16 + 8 + 1
        );
        construct_situation(&a[..], &mut b[..], a.len(), 2 * 272144 + 64 + 32 + 16 + 8);
        assert_eq!(
            super::FindMatchLengthWithLimit(&a[..], &b[..], a.len()),
            2 * 272144 + 64 + 32 + 16 + 8
        );
        construct_situation(&a[..], &mut b[..], a.len(), a.len());
        assert_eq!(
            super::FindMatchLengthWithLimit(&a[..], &b[..], a.len()),
            a.len()
        );
    }
}
