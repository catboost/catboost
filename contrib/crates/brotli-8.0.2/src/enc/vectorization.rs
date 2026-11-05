#![allow(unknown_lints)]
#![allow(unused_macros)]

use crate::enc::util::FastLog2;
use crate::enc::{s8, v8};
pub type Mem256f = v8;
pub type Mem256i = s8;
pub type v256 = v8;
pub type v256i = s8;
pub fn sum8(x: v256) -> f32 {
    x[0] + x[1] + x[2] + x[3] + x[4] + x[5] + x[6] + x[7]
}

pub fn sum8i(x: v256i) -> i32 {
    x[0].wrapping_add(x[1])
        .wrapping_add(x[2])
        .wrapping_add(x[3])
        .wrapping_add(x[4])
        .wrapping_add(x[5])
        .wrapping_add(x[6])
        .wrapping_add(x[7])
}

pub fn log2i(x: v256i) -> v256 {
    [
        FastLog2(x[0] as u64),
        FastLog2(x[1] as u64),
        FastLog2(x[2] as u64),
        FastLog2(x[3] as u64),
        FastLog2(x[4] as u64),
        FastLog2(x[5] as u64),
        FastLog2(x[6] as u64),
        FastLog2(x[7] as u64),
    ]
    .into()
}
pub fn cast_i32_to_f32(x: v256i) -> v256 {
    [
        x[0] as f32,
        x[1] as f32,
        x[2] as f32,
        x[3] as f32,
        x[4] as f32,
        x[5] as f32,
        x[6] as f32,
        x[7] as f32,
    ]
    .into()
}
pub fn cast_f32_to_i32(x: v256) -> v256i {
    [
        x[0] as i32,
        x[1] as i32,
        x[2] as i32,
        x[3] as i32,
        x[4] as i32,
        x[5] as i32,
        x[6] as i32,
        x[7] as i32,
    ]
    .into()
}
