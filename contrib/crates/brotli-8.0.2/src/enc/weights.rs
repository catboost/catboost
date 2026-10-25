use core::cmp::{max, min};
pub type Prob = u16;

pub const BLEND_FIXED_POINT_PRECISION: i8 = 15;
#[allow(dead_code)]
pub const LOG2_SCALE: i32 = 15;
#[derive(Debug, Copy, Clone)]
pub struct Weights {
    model_weights: [i32; 2],
    mixing_param: u8,
    normalized_weight: Prob,
}
impl Default for Weights {
    fn default() -> Self {
        Self::new()
    }
}
impl Weights {
    pub fn new() -> Self {
        Weights {
            model_weights: [1; 2],
            mixing_param: 1,
            normalized_weight: 1 << (BLEND_FIXED_POINT_PRECISION - 1),
        }
    }
    #[allow(unused)]
    #[inline(always)]
    pub fn update(&mut self, model_probs: [Prob; 2], weighted_prob: Prob) {
        debug_assert!(self.mixing_param != 0);
        normalize_weights(&mut self.model_weights);
        let w0new = compute_new_weight(
            model_probs,
            weighted_prob,
            self.model_weights,
            false,
            self.mixing_param - 1,
        );
        let w1new = compute_new_weight(
            model_probs,
            weighted_prob,
            self.model_weights,
            true,
            self.mixing_param - 1,
        );
        self.model_weights = [w0new, w1new];
        self.normalized_weight = compute_normalized_weight(self.model_weights);
    }
    #[allow(dead_code)]
    #[inline(always)]
    pub fn norm_weight(&self) -> Prob {
        self.normalized_weight
    }
}

#[allow(dead_code)]
#[inline(always)]
fn compute_normalized_weight(model_weights: [i32; 2]) -> Prob {
    let total = i64::from(model_weights[0]) + i64::from(model_weights[1]);
    let leading_zeros = total.leading_zeros();
    let shift = max(56 - (leading_zeros as i8), 0);
    let total_8bit = total >> shift;
    /*::probability::numeric::fast_divide_16bit_by_8bit(
    ((model_weights[0] >> shift) as u16)<< 8,
    ::probability::numeric::lookup_divisor8(total_8bit as u8)) << (BLEND_FIXED_POINT_PRECISION - 8)
    */
    ((((model_weights[0] >> shift) as u16) << 8) / total_8bit as u16/*fixme??*/)
        << (BLEND_FIXED_POINT_PRECISION - 8)
}

#[allow(dead_code)]
#[cold]
fn fix_weights(weights: &mut [i32; 2]) {
    let ilog = 32 - min(weights[0].leading_zeros(), weights[1].leading_zeros());
    let max_log = 24;
    if ilog >= max_log {
        weights[0] >>= ilog - max_log;
        weights[1] >>= ilog - max_log;
    }
}

#[allow(dead_code)]
#[inline(always)]
fn normalize_weights(weights: &mut [i32; 2]) {
    if ((weights[0] | weights[1]) & 0x7f00_0000) != 0 {
        fix_weights(weights);
    }
}

#[allow(dead_code)]
#[cfg(feature = "floating_point_context_mixing")]
fn compute_new_weight(
    probs: [Prob; 2],
    weighted_prob: Prob,
    weights: [i32; 2],
    index_equal_1: bool,
    _speed: u8,
) -> i32 {
    // speed ranges from 1 to 14 inclusive
    let index = index_equal_1 as usize;
    let n1i = probs[index] as f64 / ((1i64 << LOG2_SCALE) as f64);
    //let n0i = 1.0f64 - n1i;
    let ni = 1.0f64;
    let s1 = weighted_prob as f64 / ((1i64 << LOG2_SCALE) as f64);
    let s0 = 1.0f64 - s1;
    let s = 1.0f64;
    //let p0 = s0;
    let p1 = s1;
    let wi = weights[index] as f64 / ((1i64 << LOG2_SCALE) as f64);
    let mut wi_new = wi + (1.0 - p1) * (s * n1i - s1 * ni) / (s0 * s1);
    let eps = 0.00001f64;
    if !(wi_new > eps) {
        wi_new = eps;
    }
    (wi_new * ((1i64 << LOG2_SCALE) as f64)) as i32
}

#[allow(dead_code)]
#[cfg(not(feature = "floating_point_context_mixing"))]
#[inline(always)]
fn compute_new_weight(
    probs: [Prob; 2],
    weighted_prob: Prob,
    weights: [i32; 2],
    index_equal_1: bool,
    _speed: u8,
) -> i32 {
    // speed ranges from 1 to 14 inclusive
    let index = index_equal_1 as usize;
    let full_model_sum_p1 = i64::from(weighted_prob);
    let full_model_total = 1i64 << LOG2_SCALE;
    let full_model_sum_p0 = full_model_total.wrapping_sub(i64::from(weighted_prob));
    let n1i = i64::from(probs[index]);
    let ni = 1i64 << LOG2_SCALE;
    let error = full_model_total.wrapping_sub(full_model_sum_p1);
    let wi = i64::from(weights[index]);
    let efficacy = full_model_total.wrapping_mul(n1i) - full_model_sum_p1.wrapping_mul(ni);
    //let geometric_probabilities = full_model_sum_p1 * full_model_sum_p0;
    let log_geometric_probabilities =
        64 - (full_model_sum_p1.wrapping_mul(full_model_sum_p0)).leading_zeros();
    //let scaled_geometric_probabilities = geometric_probabilities * S;
    //let new_weight_adj = (error * efficacy) >> log_geometric_probabilities;// / geometric_probabilities;
    //let new_weight_adj = (error * efficacy)/(full_model_sum_p1 * full_model_sum_p0);
    let new_weight_adj = (error.wrapping_mul(efficacy)) >> log_geometric_probabilities;
    //    assert!(wi + new_weight_adj < (1i64 << 31));
    //print!("{} -> {} due to {:?} vs {}\n", wi as f64 / (weights[0] + weights[1]) as f64, (wi + new_weight_adj) as f64 /(weights[0] as i64 + new_weight_adj as i64 + weights[1] as i64) as f64, probs[index], weighted_prob);
    max(1, wi.wrapping_add(new_weight_adj) as i32)
}
