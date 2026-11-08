#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg_attr(not(feature = "no-inline"), inline)]
#[cfg(target_arch = "x86_64")]
pub fn add_overflow(value1: u64, value2: u64, result: &mut u64) -> bool {
    unsafe { _addcarry_u64(0, value1, value2, result) != 0 }
}

//TODO: static?

#[cfg_attr(not(feature = "no-inline"), inline)]
#[cfg(target_arch = "x86_64")]
pub fn hamming(input_num: u64) -> u32 {
    unsafe { _popcnt64(input_num as i64) as u32 }
}

#[cfg_attr(not(feature = "no-inline"), inline)]
#[cfg(target_arch = "x86_64")]
pub fn hamming(input_num: u64) -> u32 {
    unsafe { __popcnt(input_num as u32) + __popcnt((input_num >> 32) as u32) as u32 }
}

#[cfg_attr(not(feature = "no-inline"), inline)]
#[cfg(target_arch = "x86_64")]
pub fn trailingzeroes(input_num: u64) -> u32 {
    unsafe { _tzcnt_u64(input_num) as u32 }
}
