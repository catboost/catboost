#![allow(clippy::excessive_precision)]

use crate::enc::log_table_16::logs_16;
use crate::enc::log_table_8::logs_8;

#[cfg(feature = "float64")]
pub type floatX = f64;

#[cfg(not(feature = "float64"))]
pub type floatX = f32;

#[inline(always)]
pub fn FastLog2u16(v: u16) -> floatX {
    logs_16[v as usize]
}

#[cfg(feature = "std")]
#[inline(always)]
pub fn FastLog2(v: u64) -> floatX {
    if v < 256 {
        logs_8[v as usize]
    } else {
        (v as f32).log2() as floatX
    }
}

#[cfg(not(feature = "std"))]
#[inline(always)]
pub fn FastLog2(v: u64) -> floatX {
    if v < 256 {
        logs_8[v as usize]
    } else {
        FastLog2u64(v)
    }
}

#[cfg(feature = "std")]
#[inline(always)]
pub fn FastLog2f64(v: u64) -> floatX {
    if v < 256 {
        logs_8[v as usize]
    } else {
        (v as floatX).log2()
    }
}

#[cfg(not(feature = "std"))]
#[inline(always)]
pub fn FastLog2f64(v: u64) -> floatX {
    FastLog2(v) as floatX
}

#[inline]
pub fn FastLog2u64(v: u64) -> floatX {
    let bsr_8 = 56i8 - v.leading_zeros() as i8;
    let offset = bsr_8 & -((bsr_8 >= 0) as i8);
    (offset as floatX) + logs_8[(v >> offset) as usize]
}

#[inline(always)]
pub fn FastLog2u32(v: i32) -> floatX {
    let bsr_8 = 24i8 - v.leading_zeros() as i8;
    let offset = bsr_8 & -((bsr_8 >= 0) as i8);
    (offset as floatX) + logs_8[(v >> offset) as usize]
}

#[inline(always)]
pub fn xFastLog2u16(v: u16) -> floatX {
    let bsr_8 = 8i8 - v.leading_zeros() as i8;
    let offset = (bsr_8 & -((bsr_8 >= 0) as i8));
    (offset as floatX) + logs_8[(v >> offset) as usize]
}

#[cfg(feature = "std")]
#[inline(always)]
pub fn FastPow2(v: floatX) -> floatX {
    (2 as floatX).powf(v)
}

#[cfg(not(feature = "std"))]
#[inline(always)]
pub fn FastPow2(v: floatX) -> floatX {
    assert!(v >= 0 as floatX);
    let round_down = v as i32;
    let remainder = v - round_down as floatX;
    let mut x = 1 as floatX;
    // (1 + (x/n) * ln2) ^ n
    // let n = 8
    x += remainder * (0.693147180559945309417232121458 / 256.0) as floatX;
    x *= x;
    x *= x;
    x *= x;
    x *= x;
    x *= x;
    x *= x;
    x *= x;
    x *= x;

    (1 << round_down) as floatX * x
}

#[inline(always)]
pub fn Log2FloorNonZero(v: u64) -> u32 {
    63u32 ^ v.leading_zeros()
}

#[cfg(test)]
mod test {
    fn baseline_log2_floor_non_zero(mut n: u64) -> u32 {
        let mut result: u32 = 0;
        while {
            n >>= 1i32;
            n
        } != 0
        {
            result = result.wrapping_add(1);
        }
        result
    }

    #[test]
    fn log2floor_non_zero_works() {
        let examples = [
            4u64,
            254,
            256,
            1428,
            25412509,
            21350891256,
            65536,
            1258912591,
            60968101,
            1,
            12589125190825,
            105912059215091,
            0,
        ];
        for example in examples.iter() {
            let fast_version = super::Log2FloorNonZero(*example);
            let baseline_version = baseline_log2_floor_non_zero(*example);
            if *example != 0 {
                // make sure we don't panic when computing...but don't care about result
                assert_eq!(fast_version, baseline_version);
            }
        }
    }
    pub fn approx_eq(a: f64, b: f64, tol: f64) {
        let mut t0 = a - b;
        let mut t1 = b - a;
        if t0 < 0.0 {
            t0 = -t0;
        }
        if t1 < 0.0 {
            t1 = -t1;
        }
        if (!(t1 < tol)) {
            assert_eq!(a, b);
        }
        if (!(t0 < tol)) {
            assert_eq!(a, b);
        }
    }
    #[test]
    fn fast_log2_works() {
        let examples = [
            4u64,
            254,
            256,
            1428,
            25412509,
            21350891256,
            65536,
            1258912591,
            60968101,
            1,
            12589125190825,
            105912059215091,
            0,
        ];
        let tol = [
            0.00001, 0.0001, 0.0001, 0.005, 0.007, 0.008, 0.01, 0.01, 0.01, 0.000001, 0.01, 0.01,
            0.0001,
        ];
        for (index, example) in examples.iter().enumerate() {
            let fast_version = super::FastLog2(*example);
            if *example != 0 {
                // make sure we don't panic when computing...but don't care about result
                let baseline_version = (*example as f64).log2();
                approx_eq(fast_version as f64, baseline_version, tol[index]);
            } else {
                //assert_eq!(fast_version as f64, 0.0 as f64);
            }
        }
    }
}
