#![cfg_attr(feature = "simd", allow(unused))]

use core::ops::{Add, AddAssign, BitAnd, Index, IndexMut, Mul, Shr, Sub};

#[derive(Default, Copy, Clone, Debug)]
pub struct Compat16x16([i16; 16]);
impl Compat16x16 {
    #[inline(always)]
    pub fn splat(a: i16) -> Compat16x16 {
        Compat16x16([a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a])
    }
    #[inline(always)]
    pub fn to_int(&self) -> Self {
        *self
    }
    #[inline(always)]
    pub fn simd_gt(&self, rhs: Compat16x16) -> Compat16x16 {
        Self([
            -((self[0] > rhs[0]) as i16),
            -((self[1] > rhs[1]) as i16),
            -((self[2] > rhs[2]) as i16),
            -((self[3] > rhs[3]) as i16),
            -((self[4] > rhs[4]) as i16),
            -((self[5] > rhs[5]) as i16),
            -((self[6] > rhs[6]) as i16),
            -((self[7] > rhs[7]) as i16),
            -((self[8] > rhs[8]) as i16),
            -((self[9] > rhs[9]) as i16),
            -((self[10] > rhs[10]) as i16),
            -((self[11] > rhs[11]) as i16),
            -((self[12] > rhs[12]) as i16),
            -((self[13] > rhs[13]) as i16),
            -((self[14] > rhs[14]) as i16),
            -((self[15] > rhs[15]) as i16),
        ])
    }
}

macro_rules! op16 {
    ($a: expr, $b: expr, $op: expr) => {
        Compat16x16([
            $op($a[0], $b[0]),
            $op($a[1], $b[1]),
            $op($a[2], $b[2]),
            $op($a[3], $b[3]),
            $op($a[4], $b[4]),
            $op($a[5], $b[5]),
            $op($a[6], $b[6]),
            $op($a[7], $b[7]),
            $op($a[8], $b[8]),
            $op($a[9], $b[9]),
            $op($a[10], $b[10]),
            $op($a[11], $b[11]),
            $op($a[12], $b[12]),
            $op($a[13], $b[13]),
            $op($a[14], $b[14]),
            $op($a[15], $b[15]),
        ])
    };
}
macro_rules! scalar_op16 {
    ($a: expr, $b: expr, $op: expr) => {
        Compat16x16([
            $op($a[0], $b),
            $op($a[1], $b),
            $op($a[2], $b),
            $op($a[3], $b),
            $op($a[4], $b),
            $op($a[5], $b),
            $op($a[6], $b),
            $op($a[7], $b),
            $op($a[8], $b),
            $op($a[9], $b),
            $op($a[10], $b),
            $op($a[11], $b),
            $op($a[12], $b),
            $op($a[13], $b),
            $op($a[14], $b),
            $op($a[15], $b),
        ])
    };
}
#[inline(always)]
fn wrapping_i16_add(a: i16, b: i16) -> i16 {
    a.wrapping_add(b)
}
#[inline(always)]
fn wrapping_i16_sub(a: i16, b: i16) -> i16 {
    a.wrapping_sub(b)
}
#[inline(always)]
fn i16_bitand(a: i16, b: i16) -> i16 {
    a & b
}
#[inline(always)]
fn shift16<Scalar>(a: i16, b: Scalar) -> i16
where
    i64: From<Scalar>,
{
    a >> i64::from(b)
}
impl Add for Compat16x16 {
    type Output = Compat16x16;
    #[inline(always)]
    fn add(self, other: Compat16x16) -> Compat16x16 {
        op16!(self.0, other.0, wrapping_i16_add)
    }
}
impl Sub for Compat16x16 {
    type Output = Compat16x16;
    #[inline(always)]
    fn sub(self, other: Compat16x16) -> Compat16x16 {
        op16!(self.0, other.0, wrapping_i16_sub)
    }
}
impl BitAnd for Compat16x16 {
    type Output = Compat16x16;
    #[inline(always)]
    fn bitand(self, other: Compat16x16) -> Compat16x16 {
        op16!(self.0, other.0, i16_bitand)
    }
}
impl From<[i16; 16]> for Compat16x16 {
    fn from(value: [i16; 16]) -> Self {
        Self(value)
    }
}
impl<I> Index<I> for Compat16x16
where
    I: core::slice::SliceIndex<[i16]>,
{
    type Output = I::Output;

    fn index(&self, index: I) -> &Self::Output {
        &self.0[index]
    }
}
impl<I> IndexMut<I> for Compat16x16
where
    I: core::slice::SliceIndex<[i16]>,
{
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        &mut self.0[index]
    }
}
impl<Scalar: Clone> Shr<Scalar> for Compat16x16
where
    i64: From<Scalar>,
{
    type Output = Compat16x16;
    #[inline(always)]
    fn shr(self, other: Scalar) -> Compat16x16 {
        scalar_op16!(self.0, other.clone(), shift16)
    }
}

#[derive(Default, Copy, Clone, Debug)]
pub struct Compat32x8([i32; 8]);
impl Compat32x8 {
    #[inline(always)]
    pub fn splat(a: i32) -> Compat32x8 {
        Compat32x8([a, a, a, a, a, a, a, a])
    }
    #[inline(always)]
    pub fn simd_gt(&self, rhs: Compat32x8) -> Compat32x8 {
        Self([
            -((self[0] > rhs[0]) as i32),
            -((self[1] > rhs[1]) as i32),
            -((self[2] > rhs[2]) as i32),
            -((self[3] > rhs[3]) as i32),
            -((self[4] > rhs[4]) as i32),
            -((self[5] > rhs[5]) as i32),
            -((self[6] > rhs[6]) as i32),
            -((self[7] > rhs[7]) as i32),
        ])
    }
    #[inline(always)]
    pub fn simd_ge(&self, rhs: Compat32x8) -> Compat32x8 {
        Self([
            -((self[0] >= rhs[0]) as i32),
            -((self[1] >= rhs[1]) as i32),
            -((self[2] >= rhs[2]) as i32),
            -((self[3] >= rhs[3]) as i32),
            -((self[4] >= rhs[4]) as i32),
            -((self[5] >= rhs[5]) as i32),
            -((self[6] >= rhs[6]) as i32),
            -((self[7] >= rhs[7]) as i32),
        ])
    }
    pub fn to_int(&self) -> Self {
        *self
    }
}

#[inline(always)]
fn fmin(a: f32, b: f32) -> f32 {
    if a < b {
        a
    } else {
        b
    }
}
#[derive(Default, Copy, Clone, Debug)]
pub struct CompatF8([f32; 8]);
impl CompatF8 {
    #[inline(always)]
    pub fn splat(a: f32) -> CompatF8 {
        CompatF8([a, a, a, a, a, a, a, a])
    }
    #[inline(always)]
    pub fn simd_ge(&self, rhs: CompatF8) -> Compat32x8 {
        Compat32x8([
            -((self[0] >= rhs[0]) as i32),
            -((self[1] >= rhs[1]) as i32),
            -((self[2] >= rhs[2]) as i32),
            -((self[3] >= rhs[3]) as i32),
            -((self[4] >= rhs[4]) as i32),
            -((self[5] >= rhs[5]) as i32),
            -((self[6] >= rhs[6]) as i32),
            -((self[7] >= rhs[7]) as i32),
        ])
    }
    #[inline(always)]
    pub fn simd_min(&self, rhs: CompatF8) -> CompatF8 {
        Self([
            fmin(self[0], rhs[0]),
            fmin(self[1], rhs[1]),
            fmin(self[2], rhs[2]),
            fmin(self[3], rhs[3]),
            fmin(self[4], rhs[4]),
            fmin(self[5], rhs[5]),
            fmin(self[6], rhs[6]),
            fmin(self[7], rhs[7]),
        ])
    }
}
impl Add for Compat32x8 {
    type Output = Compat32x8;
    #[inline(always)]
    fn add(self, other: Compat32x8) -> Compat32x8 {
        Compat32x8([
            self.0[0].wrapping_add(other.0[0]),
            self.0[1].wrapping_add(other.0[1]),
            self.0[2].wrapping_add(other.0[2]),
            self.0[3].wrapping_add(other.0[3]),
            self.0[4].wrapping_add(other.0[4]),
            self.0[5].wrapping_add(other.0[5]),
            self.0[6].wrapping_add(other.0[6]),
            self.0[7].wrapping_add(other.0[7]),
        ])
    }
}

impl BitAnd for Compat32x8 {
    type Output = Compat32x8;
    #[inline(always)]
    fn bitand(self, other: Compat32x8) -> Compat32x8 {
        Compat32x8([
            self.0[0] & other.0[0],
            self.0[1] & other.0[1],
            self.0[2] & other.0[2],
            self.0[3] & other.0[3],
            self.0[4] & other.0[4],
            self.0[5] & other.0[5],
            self.0[6] & other.0[6],
            self.0[7] & other.0[7],
        ])
    }
}
impl Mul for Compat32x8 {
    type Output = Compat32x8;
    #[inline(always)]
    fn mul(self, other: Compat32x8) -> Compat32x8 {
        Compat32x8([
            self.0[0].wrapping_mul(other.0[0]),
            self.0[1].wrapping_mul(other.0[1]),
            self.0[2].wrapping_mul(other.0[2]),
            self.0[3].wrapping_mul(other.0[3]),
            self.0[4].wrapping_mul(other.0[4]),
            self.0[5].wrapping_mul(other.0[5]),
            self.0[6].wrapping_mul(other.0[6]),
            self.0[7].wrapping_mul(other.0[7]),
        ])
    }
}
impl From<[i32; 8]> for Compat32x8 {
    fn from(value: [i32; 8]) -> Self {
        Self(value)
    }
}
impl<I> Index<I> for Compat32x8
where
    I: core::slice::SliceIndex<[i32]>,
{
    type Output = I::Output;

    fn index(&self, index: I) -> &Self::Output {
        &self.0[index]
    }
}
impl<I> IndexMut<I> for Compat32x8
where
    I: core::slice::SliceIndex<[i32]>,
{
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        &mut self.0[index]
    }
}
impl Add for CompatF8 {
    type Output = CompatF8;
    #[inline(always)]
    fn add(self, other: CompatF8) -> CompatF8 {
        CompatF8([
            self.0[0] + other.0[0],
            self.0[1] + other.0[1],
            self.0[2] + other.0[2],
            self.0[3] + other.0[3],
            self.0[4] + other.0[4],
            self.0[5] + other.0[5],
            self.0[6] + other.0[6],
            self.0[7] + other.0[7],
        ])
    }
}
impl Sub for CompatF8 {
    type Output = CompatF8;
    #[inline(always)]
    fn sub(self, other: CompatF8) -> CompatF8 {
        CompatF8([
            self.0[0] - other.0[0],
            self.0[1] - other.0[1],
            self.0[2] - other.0[2],
            self.0[3] - other.0[3],
            self.0[4] - other.0[4],
            self.0[5] - other.0[5],
            self.0[6] - other.0[6],
            self.0[7] - other.0[7],
        ])
    }
}
impl Mul for CompatF8 {
    type Output = CompatF8;
    #[inline(always)]
    fn mul(self, other: CompatF8) -> CompatF8 {
        CompatF8([
            self.0[0] * other.0[0],
            self.0[1] * other.0[1],
            self.0[2] * other.0[2],
            self.0[3] * other.0[3],
            self.0[4] * other.0[4],
            self.0[5] * other.0[5],
            self.0[6] * other.0[6],
            self.0[7] * other.0[7],
        ])
    }
}
impl AddAssign for CompatF8 {
    #[inline(always)]
    fn add_assign(&mut self, other: CompatF8) {
        self.0[0] += other.0[0];
        self.0[1] += other.0[1];
        self.0[2] += other.0[2];
        self.0[3] += other.0[3];
        self.0[4] += other.0[4];
        self.0[5] += other.0[5];
        self.0[6] += other.0[6];
        self.0[7] += other.0[7];
    }
}
impl From<[f32; 8]> for CompatF8 {
    fn from(value: [f32; 8]) -> Self {
        Self(value)
    }
}
impl<I> Index<I> for CompatF8
where
    I: core::slice::SliceIndex<[f32]>,
{
    type Output = I::Output;

    fn index(&self, index: I) -> &Self::Output {
        &self.0[index]
    }
}
impl<I> IndexMut<I> for CompatF8
where
    I: core::slice::SliceIndex<[f32]>,
{
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        &mut self.0[index]
    }
}
