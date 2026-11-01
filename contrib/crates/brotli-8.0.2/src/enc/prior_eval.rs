use core;
use core::cmp::min;
#[cfg(feature = "simd")]
use core::simd::prelude::SimdPartialOrd;

use super::super::alloc;
use super::super::alloc::{Allocator, SliceWrapper, SliceWrapperMut};
use super::backward_references::BrotliEncoderParams;
use super::input_pair::{InputPair, InputReference, InputReferenceMut};
use super::ir_interpret::{push_base, IRInterpreter};
use super::util::{floatX, FastLog2u16};
use super::{find_stride, interface, s16, v8};
use crate::enc::combined_alloc::{alloc_default, alloc_if};

// the high nibble, followed by the low nibbles
pub const CONTEXT_MAP_PRIOR_SIZE: usize = 256 * 17;
pub const STRIDE_PRIOR_SIZE: usize = 256 * 256 * 2;
pub const ADV_PRIOR_SIZE: usize = 65536 + (20 << 16);
pub const DEFAULT_SPEED: (u16, u16) = (8, 8192);

pub enum WhichPrior {
    CM = 0,
    ADV = 1,
    SLOW_CM = 2,
    FAST_CM = 3,
    STRIDE1 = 4,
    STRIDE2 = 5,
    STRIDE3 = 6,
    STRIDE4 = 7,
    //    STRIDE8 = 8,
    NUM_PRIORS = 8,
    // future ideas
}

pub trait Prior {
    fn lookup_lin(
        stride_byte: u8,
        selected_context: u8,
        actual_context: usize,
        high_nibble: Option<u8>,
    ) -> usize;
    #[inline(always)]
    fn lookup_mut(
        data: &mut [s16],
        stride_byte: u8,
        selected_context: u8,
        actual_context: usize,
        high_nibble: Option<u8>,
    ) -> CDF {
        let index = Self::lookup_lin(stride_byte, selected_context, actual_context, high_nibble);
        CDF::from(&mut data[index])
    }
    #[inline(always)]
    fn lookup(
        data: &[s16],
        stride_byte: u8,
        selected_context: u8,
        actual_context: usize,
        high_nibble: Option<u8>,
    ) -> &s16 {
        let index = Self::lookup_lin(stride_byte, selected_context, actual_context, high_nibble);
        &data[index]
    }
    #[allow(unused_variables)]
    #[inline(always)]
    fn score_index(
        stride_byte: u8,
        selected_context: u8,
        actual_context: usize,
        high_nibble: Option<u8>,
    ) -> usize {
        let which = Self::which();
        assert!(which < WhichPrior::NUM_PRIORS as usize);
        assert!(actual_context < 256);
        if let Some(nibble) = high_nibble {
            WhichPrior::NUM_PRIORS as usize * (actual_context + 4096 + 256 * nibble as usize)
                + which
        } else {
            WhichPrior::NUM_PRIORS as usize * (actual_context + 256 * (stride_byte >> 4) as usize)
                + which
        }
    }
    fn which() -> usize;
}

#[inline(always)]
fn upper_score_index(stride_byte: u8, _selected_context: u8, actual_context: usize) -> usize {
    actual_context + 256 * (stride_byte >> 4) as usize
}
#[inline(always)]
fn lower_score_index(
    _stride_byte: u8,
    _selected_context: u8,
    actual_context: usize,
    high_nibble: u8,
) -> usize {
    debug_assert!(actual_context < 256);
    debug_assert!(high_nibble < 16);
    actual_context + 4096 + 256 * high_nibble as usize
}

#[allow(unused_variables)]
#[inline(always)]
fn stride_lookup_lin(
    stride_byte: u8,
    selected_context: u8,
    actual_context: usize,
    high_nibble: Option<u8>,
) -> usize {
    if let Some(nibble) = high_nibble {
        1 + 2 * (actual_context | ((stride_byte as usize & 0x0f) << 8) | ((nibble as usize) << 12))
    } else {
        2 * (actual_context | ((stride_byte as usize) << 8))
    }
}
pub struct Stride1Prior {}
impl Stride1Prior {
    #[inline(always)]
    pub fn offset() -> usize {
        0
    }
}

impl Prior for Stride1Prior {
    #[inline(always)]
    fn lookup_lin(
        stride_byte: u8,
        selected_context: u8,
        actual_context: usize,
        high_nibble: Option<u8>,
    ) -> usize {
        stride_lookup_lin(stride_byte, selected_context, actual_context, high_nibble)
    }
    #[inline(always)]
    fn which() -> usize {
        WhichPrior::STRIDE1 as usize
    }
}
/*impl StridePrior for Stride1Prior {
    const STRIDE_OFFSET:usize = 0;
}*/
pub struct Stride2Prior {}
impl Stride2Prior {
    #[inline(always)]
    pub fn offset() -> usize {
        1
    }
}

impl Prior for Stride2Prior {
    #[inline(always)]
    fn lookup_lin(
        stride_byte: u8,
        selected_context: u8,
        actual_context: usize,
        high_nibble: Option<u8>,
    ) -> usize {
        stride_lookup_lin(stride_byte, selected_context, actual_context, high_nibble)
    }
    #[inline]
    fn which() -> usize {
        WhichPrior::STRIDE2 as usize
    }
}
/*impl StridePrior for Stride2Prior {
    const STRIDE_OFFSET:usize = 1;
}*/
pub struct Stride3Prior {}
impl Stride3Prior {
    #[inline(always)]
    pub fn offset() -> usize {
        2
    }
}

impl Prior for Stride3Prior {
    #[inline(always)]
    fn lookup_lin(
        stride_byte: u8,
        selected_context: u8,
        actual_context: usize,
        high_nibble: Option<u8>,
    ) -> usize {
        stride_lookup_lin(stride_byte, selected_context, actual_context, high_nibble)
    }
    #[inline(always)]
    fn which() -> usize {
        WhichPrior::STRIDE3 as usize
    }
}

/*impl StridePrior for Stride3Prior {
    const STRIDE_OFFSET:usize = 2;
}*/
pub struct Stride4Prior {}
impl Stride4Prior {
    #[inline(always)]
    pub fn offset() -> usize {
        3
    }
}
impl Prior for Stride4Prior {
    #[inline(always)]
    fn lookup_lin(
        stride_byte: u8,
        selected_context: u8,
        actual_context: usize,
        high_nibble: Option<u8>,
    ) -> usize {
        stride_lookup_lin(stride_byte, selected_context, actual_context, high_nibble)
    }
    #[inline]
    fn which() -> usize {
        WhichPrior::STRIDE4 as usize
    }
}

/*impl StridePrior for Stride4Prior {
    const STRIDE_OFFSET:usize = 3;
}*/
/*pub struct Stride8Prior{
}
impl StridePrior for Stride8Prior {
    const STRIDE_OFFSET:usize = 7;
}
impl Stride8Prior {
    #[inline(always)]
    pub fn offset() -> usize{
        7
    }
}
impl Prior for Stride8Prior {
    fn lookup_lin(stride_byte:u8, selected_context:u8, actual_context:usize, high_nibble: Option<u8>) -> usize {
        stride_lookup_lin(stride_byte, selected_context, actual_context, high_nibble)
    }
    #[inline]
    fn which() -> usize {
      WhichPrior::STRIDE8 as usize
    }
}
*/
pub struct CMPrior {}
impl Prior for CMPrior {
    #[allow(unused_variables)]
    #[inline(always)]
    fn lookup_lin(
        stride_byte: u8,
        selected_context: u8,
        actual_context: usize,
        high_nibble: Option<u8>,
    ) -> usize {
        if let Some(nibble) = high_nibble {
            (nibble as usize + 1) + 17 * actual_context
        } else {
            17 * actual_context
        }
    }
    #[inline(always)]
    fn which() -> usize {
        WhichPrior::CM as usize
    }
}
pub struct FastCMPrior {}
impl Prior for FastCMPrior {
    #[allow(unused_variables)]
    #[inline(always)]
    fn lookup_lin(
        stride_byte: u8,
        selected_context: u8,
        actual_context: usize,
        high_nibble: Option<u8>,
    ) -> usize {
        if let Some(nibble) = high_nibble {
            2 * actual_context
        } else {
            2 * actual_context + 1
        }
    }
    #[inline(always)]
    fn which() -> usize {
        WhichPrior::FAST_CM as usize
    }
}

pub struct SlowCMPrior {}
impl Prior for SlowCMPrior {
    #[allow(unused_variables)]
    #[inline(always)]
    fn lookup_lin(
        stride_byte: u8,
        selected_context: u8,
        actual_context: usize,
        high_nibble: Option<u8>,
    ) -> usize {
        if let Some(nibble) = high_nibble {
            (nibble as usize + 1) + 17 * actual_context
        } else {
            17 * actual_context
        }
    }
    #[inline]
    fn which() -> usize {
        WhichPrior::SLOW_CM as usize
    }
}

pub struct AdvPrior {}
impl Prior for AdvPrior {
    #[allow(unused_variables)]
    #[inline(always)]
    fn lookup_lin(
        stride_byte: u8,
        selected_context: u8,
        actual_context: usize,
        high_nibble: Option<u8>,
    ) -> usize {
        if let Some(nibble) = high_nibble {
            65536
                + (actual_context | ((stride_byte as usize) << 8) | ((nibble as usize & 0xf) << 16))
        } else {
            actual_context | ((stride_byte as usize & 0xf0) << 8)
        }
    }
    #[inline(always)]
    fn which() -> usize {
        WhichPrior::ADV as usize
    }
}

pub struct CDF<'a> {
    cdf: &'a mut s16,
}

impl<'a> CDF<'a> {
    #[inline(always)]
    pub fn cost(&self, nibble_u8: u8) -> floatX {
        let nibble = nibble_u8 as usize & 0xf;
        let mut pdf = self.cdf[nibble];
        if nibble_u8 != 0 {
            pdf -= self.cdf[(nibble - 1)];
        }
        FastLog2u16(self.cdf[15] as u16) - FastLog2u16(pdf as u16)
    }
    #[inline(always)]
    pub fn update(&mut self, nibble_u8: u8, speed: (u16, u16)) {
        let mut cdf = *self.cdf;
        let increment_v = s16::splat(speed.0 as i16);
        let one_to_16 = s16::from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
        let mask_v: s16 = one_to_16.simd_gt(s16::splat(i16::from(nibble_u8))).to_int();
        cdf = cdf + (increment_v & mask_v);
        if cdf[15] >= speed.1 as i16 {
            let cdf_bias = one_to_16;
            cdf = cdf + cdf_bias - ((cdf + cdf_bias) >> 2);
        }
        *self.cdf = cdf;
    }
}

impl<'a> From<&'a mut s16> for CDF<'a> {
    #[inline(always)]
    fn from(cdf: &'a mut s16) -> CDF<'a> {
        CDF { cdf }
    }
}

pub fn init_cdfs(cdfs: &mut [s16]) {
    for item in cdfs.iter_mut() {
        *item = s16::from([4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64]);
    }
}

pub struct PriorEval<
    'a,
    Alloc: alloc::Allocator<s16> + alloc::Allocator<u32> + alloc::Allocator<v8>,
> {
    input: InputPair<'a>,
    context_map: interface::PredictionModeContextMap<InputReferenceMut<'a>>,
    block_type: u8,
    local_byte_offset: usize,
    _nop: <Alloc as Allocator<u32>>::AllocatedMemory,
    cm_priors: <Alloc as Allocator<s16>>::AllocatedMemory,
    slow_cm_priors: <Alloc as Allocator<s16>>::AllocatedMemory,
    fast_cm_priors: <Alloc as Allocator<s16>>::AllocatedMemory,
    stride_priors: [<Alloc as Allocator<s16>>::AllocatedMemory; 4],
    adv_priors: <Alloc as Allocator<s16>>::AllocatedMemory,
    _stride_pyramid_leaves: [u8; find_stride::NUM_LEAF_NODES],
    score: <Alloc as Allocator<v8>>::AllocatedMemory,
    cm_speed: [(u16, u16); 2],
    stride_speed: [(u16, u16); 2],
    cur_stride: u8,
}

impl<'a, Alloc: alloc::Allocator<s16> + alloc::Allocator<u32> + alloc::Allocator<v8>>
    PriorEval<'a, Alloc>
{
    pub fn new(
        alloc: &mut Alloc,
        input: InputPair<'a>,
        stride: [u8; find_stride::NUM_LEAF_NODES],
        prediction_mode: interface::PredictionModeContextMap<InputReferenceMut<'a>>,
        params: &BrotliEncoderParams,
    ) -> Self {
        let do_alloc = params.prior_bitmask_detection != 0;
        let mut cm_speed = prediction_mode.context_map_speed();
        let mut stride_speed = prediction_mode.stride_context_speed();
        if cm_speed[0] == (0, 0) {
            cm_speed[0] = params.literal_adaptation[2]
        }
        if cm_speed[0] == (0, 0) {
            cm_speed[0] = DEFAULT_SPEED;
        }
        if cm_speed[1] == (0, 0) {
            cm_speed[1] = params.literal_adaptation[3]
        }
        if cm_speed[1] == (0, 0) {
            cm_speed[1] = cm_speed[0];
        }
        if stride_speed[0] == (0, 0) {
            stride_speed[0] = params.literal_adaptation[0]
        }
        if stride_speed[0] == (0, 0) {
            stride_speed[0] = DEFAULT_SPEED;
        }
        if stride_speed[1] == (0, 0) {
            stride_speed[1] = params.literal_adaptation[1]
        }
        if stride_speed[1] == (0, 0) {
            stride_speed[1] = stride_speed[0];
        }
        let mut ret = PriorEval::<Alloc> {
            input,
            context_map: prediction_mode,
            block_type: 0,
            cur_stride: 1,
            local_byte_offset: 0,
            _nop: alloc_default::<u32, Alloc>(),
            cm_priors: alloc_if::<s16, _>(do_alloc, alloc, CONTEXT_MAP_PRIOR_SIZE),
            slow_cm_priors: alloc_if::<s16, _>(do_alloc, alloc, CONTEXT_MAP_PRIOR_SIZE),
            fast_cm_priors: alloc_if::<s16, _>(do_alloc, alloc, CONTEXT_MAP_PRIOR_SIZE),
            stride_priors: [
                alloc_if::<s16, _>(do_alloc, alloc, STRIDE_PRIOR_SIZE),
                alloc_if::<s16, _>(do_alloc, alloc, STRIDE_PRIOR_SIZE),
                alloc_if::<s16, _>(do_alloc, alloc, STRIDE_PRIOR_SIZE),
                alloc_if::<s16, _>(do_alloc, alloc, STRIDE_PRIOR_SIZE),
                /*if do_alloc {m16x16.alloc_cell(STRIDE_PRIOR_SIZE)} else {
                Alloc16x16::AllocatedMemory::default()},*/
            ],
            adv_priors: alloc_if::<s16, _>(do_alloc, alloc, ADV_PRIOR_SIZE),
            _stride_pyramid_leaves: stride,
            score: alloc_if::<v8, _>(do_alloc, alloc, 8192),
            cm_speed,
            stride_speed,
        };
        init_cdfs(ret.cm_priors.slice_mut());
        init_cdfs(ret.slow_cm_priors.slice_mut());
        init_cdfs(ret.fast_cm_priors.slice_mut());
        init_cdfs(ret.stride_priors[0].slice_mut());
        init_cdfs(ret.stride_priors[1].slice_mut());
        init_cdfs(ret.stride_priors[2].slice_mut());
        init_cdfs(ret.stride_priors[3].slice_mut());
        //init_cdfs(ret.stride_priors[4].slice_mut());
        init_cdfs(ret.adv_priors.slice_mut());
        ret
    }
    pub fn choose_bitmask(&mut self) {
        let epsilon = 6.0;
        let mut max_popularity = 0u32;
        let mut max_popularity_index = 0u8;
        assert_eq!(WhichPrior::NUM_PRIORS as usize, 8);
        let mut popularity = [0u32; 8];
        let mut bitmask = [0u8; super::interface::NUM_MIXING_VALUES];
        for (i, score) in self.score.slice().iter().enumerate() {
            let cm_score = score[WhichPrior::CM as usize];
            let slow_cm_score = score[WhichPrior::SLOW_CM as usize];
            let fast_cm_score = score[WhichPrior::FAST_CM as usize] + 16.0;
            let stride1_score = score[WhichPrior::STRIDE1 as usize];
            let stride2_score = score[WhichPrior::STRIDE2 as usize];
            let stride3_score = score[WhichPrior::STRIDE3 as usize] + 16.0;
            let stride4_score = score[WhichPrior::STRIDE4 as usize];
            //let stride8_score = score[WhichPrior::STRIDE8] * 1.125 + 16.0;
            let stride8_score = stride4_score + 1.0; // FIXME: never lowest -- ignore stride 8
            let stride_score = min(
                stride1_score as u64,
                min(
                    stride2_score as u64,
                    min(
                        stride3_score as u64,
                        min(stride4_score as u64, stride8_score as u64),
                    ),
                ),
            );

            let adv_score = score[WhichPrior::ADV as usize];
            if adv_score + epsilon < (stride_score as floatX)
                && adv_score + epsilon < cm_score
                && adv_score + epsilon < slow_cm_score
                && adv_score + epsilon < fast_cm_score
            {
                bitmask[i] = 1;
            } else if slow_cm_score + epsilon < (stride_score as floatX)
                && slow_cm_score + epsilon < cm_score
                && slow_cm_score + epsilon < fast_cm_score
            {
                bitmask[i] = 2;
            } else if fast_cm_score + epsilon < (stride_score as floatX)
                && fast_cm_score + epsilon < cm_score
            {
                bitmask[i] = 3;
            } else if epsilon + (stride_score as floatX) < cm_score {
                bitmask[i] = WhichPrior::STRIDE1 as u8;
                if stride_score == stride8_score as u64 {
                    //bitmask[i] = WhichPrior::STRIDE8 as u8;
                }
                if stride_score == stride4_score as u64 {
                    bitmask[i] = WhichPrior::STRIDE4 as u8;
                }
                if stride_score == stride3_score as u64 {
                    bitmask[i] = WhichPrior::STRIDE3 as u8;
                }
                if stride_score == stride2_score as u64 {
                    bitmask[i] = WhichPrior::STRIDE2 as u8;
                }
                if stride_score == stride1_score as u64 {
                    bitmask[i] = WhichPrior::STRIDE1 as u8;
                }
            } else {
                bitmask[i] = 0;
            }
            if stride_score == 0 {
                bitmask[i] = max_popularity_index;
                //eprintln!("Miss {}[{}] ~ {}", bitmask[i], i, max_popularity_index);
            } else {
                popularity[bitmask[i] as usize] += 1;
                if popularity[bitmask[i] as usize] > max_popularity {
                    max_popularity = popularity[bitmask[i] as usize];
                    max_popularity_index = bitmask[i];
                }
                //eprintln!("Score {} {} {} {} {}: {}[{}] max={},{}", cm_score, adv_score, slow_cm_score, fast_cm_score, stride_score, bitmask[i], i, max_popularity, max_popularity_index);
            }
        }
        self.context_map.set_mixing_values(&bitmask);
    }
    pub fn free(&mut self, alloc: &mut Alloc) {
        <Alloc as Allocator<v8>>::free_cell(alloc, core::mem::take(&mut self.score));
        <Alloc as Allocator<s16>>::free_cell(alloc, core::mem::take(&mut self.cm_priors));
        <Alloc as Allocator<s16>>::free_cell(alloc, core::mem::take(&mut self.slow_cm_priors));
        <Alloc as Allocator<s16>>::free_cell(alloc, core::mem::take(&mut self.fast_cm_priors));
        <Alloc as Allocator<s16>>::free_cell(alloc, core::mem::take(&mut self.stride_priors[0]));
        <Alloc as Allocator<s16>>::free_cell(alloc, core::mem::take(&mut self.stride_priors[1]));
        <Alloc as Allocator<s16>>::free_cell(alloc, core::mem::take(&mut self.stride_priors[2]));
        <Alloc as Allocator<s16>>::free_cell(alloc, core::mem::take(&mut self.stride_priors[3]));
        //<Alloc as Allocator<s16>>::free_cell(alloc, core::mem::replace(&mut self.stride_priors[4], alloc_default::<s16, Alloc>()));
        <Alloc as Allocator<s16>>::free_cell(alloc, core::mem::take(&mut self.adv_priors));
    }

    pub fn take_prediction_mode(
        &mut self,
    ) -> interface::PredictionModeContextMap<InputReferenceMut<'a>> {
        core::mem::replace(
            &mut self.context_map,
            interface::PredictionModeContextMap::<InputReferenceMut<'a>> {
                literal_context_map: InputReferenceMut::default(),
                predmode_speed_and_distance_context_map: InputReferenceMut::default(),
            },
        )
    }
    fn update_cost_base(
        &mut self,
        stride_prior: [u8; 8],
        stride_prior_offset: usize,
        selected_bits: u8,
        cm_prior: usize,
        literal: u8,
    ) {
        let mut l_score = v8::splat(0.0);
        let mut h_score = v8::splat(0.0);
        let base_stride_prior =
            stride_prior[stride_prior_offset.wrapping_sub(self.cur_stride as usize) & 7];
        let hscore_index = upper_score_index(base_stride_prior, selected_bits, cm_prior);
        let lscore_index =
            lower_score_index(base_stride_prior, selected_bits, cm_prior, literal >> 4);
        {
            type CurPrior = CMPrior;
            let mut cdf = CurPrior::lookup_mut(
                self.cm_priors.slice_mut(),
                base_stride_prior,
                selected_bits,
                cm_prior,
                None,
            );
            h_score[CurPrior::which()] = cdf.cost(literal >> 4);
            cdf.update(literal >> 4, self.cm_speed[1]);
        }
        {
            type CurPrior = CMPrior;
            let mut cdf = CurPrior::lookup_mut(
                self.cm_priors.slice_mut(),
                base_stride_prior,
                selected_bits,
                cm_prior,
                Some(literal >> 4),
            );
            l_score[CurPrior::which()] = cdf.cost(literal & 0xf);
            cdf.update(literal & 0xf, self.cm_speed[0]);
        }
        {
            type CurPrior = SlowCMPrior;
            let mut cdf = CurPrior::lookup_mut(
                self.slow_cm_priors.slice_mut(),
                base_stride_prior,
                selected_bits,
                cm_prior,
                None,
            );
            h_score[CurPrior::which()] = cdf.cost(literal >> 4);
            cdf.update(literal >> 4, (0, 1024));
        }
        {
            type CurPrior = SlowCMPrior;
            let mut cdf = CurPrior::lookup_mut(
                self.slow_cm_priors.slice_mut(),
                base_stride_prior,
                selected_bits,
                cm_prior,
                Some(literal >> 4),
            );
            l_score[CurPrior::which()] = cdf.cost(literal & 0xf);
            cdf.update(literal & 0xf, (0, 1024));
        }
        {
            type CurPrior = FastCMPrior;
            let mut cdf = CurPrior::lookup_mut(
                self.fast_cm_priors.slice_mut(),
                base_stride_prior,
                selected_bits,
                cm_prior,
                None,
            );
            h_score[CurPrior::which()] = cdf.cost(literal >> 4);
            cdf.update(literal >> 4, self.cm_speed[0]);
        }
        {
            type CurPrior = FastCMPrior;
            let mut cdf = CurPrior::lookup_mut(
                self.fast_cm_priors.slice_mut(),
                base_stride_prior,
                selected_bits,
                cm_prior,
                Some(literal >> 4),
            );
            l_score[CurPrior::which()] = cdf.cost(literal & 0xf);
            cdf.update(literal & 0xf, self.cm_speed[0]);
        }
        {
            type CurPrior = Stride1Prior;
            let mut cdf = CurPrior::lookup_mut(
                self.stride_priors[0].slice_mut(),
                stride_prior[stride_prior_offset.wrapping_sub(CurPrior::offset()) & 7],
                selected_bits,
                cm_prior,
                None,
            );
            h_score[CurPrior::which()] = cdf.cost(literal >> 4);
            cdf.update(literal >> 4, self.stride_speed[1]);
        }
        {
            type CurPrior = Stride1Prior;
            let mut cdf = CurPrior::lookup_mut(
                self.stride_priors[0].slice_mut(),
                stride_prior[stride_prior_offset.wrapping_sub(CurPrior::offset()) & 7],
                selected_bits,
                cm_prior,
                Some(literal >> 4),
            );
            l_score[CurPrior::which()] = cdf.cost(literal & 0xf);
            cdf.update(literal & 0xf, self.stride_speed[0]);
        }
        {
            type CurPrior = Stride2Prior;
            let mut cdf = CurPrior::lookup_mut(
                self.stride_priors[1].slice_mut(),
                stride_prior[stride_prior_offset.wrapping_sub(CurPrior::offset()) & 7],
                selected_bits,
                cm_prior,
                None,
            );
            h_score[CurPrior::which()] = cdf.cost(literal >> 4);
            cdf.update(literal >> 4, self.stride_speed[1]);
        }
        {
            type CurPrior = Stride2Prior;
            let mut cdf = CurPrior::lookup_mut(
                self.stride_priors[1].slice_mut(),
                stride_prior[stride_prior_offset.wrapping_sub(CurPrior::offset()) & 7],
                selected_bits,
                cm_prior,
                Some(literal >> 4),
            );
            l_score[CurPrior::which()] = cdf.cost(literal & 0xf);
            cdf.update(literal & 0xf, self.stride_speed[0]);
        }
        {
            type CurPrior = Stride3Prior;
            let mut cdf = CurPrior::lookup_mut(
                self.stride_priors[2].slice_mut(),
                stride_prior[stride_prior_offset.wrapping_sub(CurPrior::offset()) & 7],
                selected_bits,
                cm_prior,
                None,
            );
            h_score[CurPrior::which()] = cdf.cost(literal >> 4);
            cdf.update(literal >> 4, self.stride_speed[1]);
        }
        {
            type CurPrior = Stride3Prior;
            let mut cdf = CurPrior::lookup_mut(
                self.stride_priors[2].slice_mut(),
                stride_prior[stride_prior_offset.wrapping_sub(CurPrior::offset()) & 7],
                selected_bits,
                cm_prior,
                Some(literal >> 4),
            );
            l_score[CurPrior::which()] = cdf.cost(literal & 0xf);
            cdf.update(literal & 0xf, self.stride_speed[0]);
        }
        {
            type CurPrior = Stride4Prior;
            let mut cdf = CurPrior::lookup_mut(
                self.stride_priors[3].slice_mut(),
                stride_prior[stride_prior_offset.wrapping_sub(CurPrior::offset()) & 7],
                selected_bits,
                cm_prior,
                None,
            );
            h_score[CurPrior::which()] = cdf.cost(literal >> 4);
            cdf.update(literal >> 4, self.stride_speed[1]);
        }
        {
            type CurPrior = Stride4Prior;
            let mut cdf = CurPrior::lookup_mut(
                self.stride_priors[3].slice_mut(),
                stride_prior[stride_prior_offset.wrapping_sub(CurPrior::offset()) & 7],
                selected_bits,
                cm_prior,
                Some(literal >> 4),
            );
            l_score[CurPrior::which()] = cdf.cost(literal & 0xf);
            cdf.update(literal & 0xf, self.stride_speed[0]);
        }
        /*       {
                   type CurPrior = Stride8Prior;
                   let mut cdf = CurPrior::lookup_mut(self.stride_priors[4].slice_mut(),
                                                      stride_prior[stride_prior_offset.wrapping_sub(CurPrior::offset())&7], selected_bits, cm_prior, None);
                   h_score[CurPrior::which()] = cdf.cost(literal>>4);
                   cdf.update(literal >> 4, self.stride_speed[1]);
               }
               {
                   type CurPrior = Stride8Prior;
                   let mut cdf = CurPrior::lookup_mut(self.stride_priors[4].slice_mut(),
                                                      stride_prior[stride_prior_offset.wrapping_sub(CurPrior::offset()) & 7],
                                                      selected_bits,
                                                      cm_prior,
                                                      Some(literal >> 4));
                   l_score[CurPrior::which()] = cdf.cost(literal&0xf);
                   cdf.update(literal&0xf, self.stride_speed[0]);
               }
        */
        type CurPrior = AdvPrior;
        {
            let mut cdf = CurPrior::lookup_mut(
                self.adv_priors.slice_mut(),
                base_stride_prior,
                selected_bits,
                cm_prior,
                None,
            );
            h_score[CurPrior::which()] = cdf.cost(literal >> 4);
            cdf.update(literal >> 4, self.stride_speed[1]);
        }
        {
            let mut cdf = CurPrior::lookup_mut(
                self.adv_priors.slice_mut(),
                base_stride_prior,
                selected_bits,
                cm_prior,
                Some(literal >> 4),
            );
            l_score[CurPrior::which()] = cdf.cost(literal & 0xf);
            cdf.update(literal & 0xf, self.stride_speed[0]);
        }
        self.score.slice_mut()[lscore_index] += l_score;
        self.score.slice_mut()[hscore_index] += h_score;
    }
}
impl<'a, Alloc: alloc::Allocator<s16> + alloc::Allocator<u32> + alloc::Allocator<v8>> IRInterpreter
    for PriorEval<'a, Alloc>
{
    #[inline]
    fn inc_local_byte_offset(&mut self, inc: usize) {
        self.local_byte_offset += inc;
    }
    #[inline]
    fn local_byte_offset(&self) -> usize {
        self.local_byte_offset
    }
    #[inline]
    fn update_block_type(&mut self, new_type: u8, stride: u8) {
        self.block_type = new_type;
        self.cur_stride = stride;
    }
    #[inline]
    fn block_type(&self) -> u8 {
        self.block_type
    }
    #[inline]
    fn literal_data_at_offset(&self, index: usize) -> u8 {
        self.input[index]
    }
    #[inline]
    fn literal_context_map(&self) -> &[u8] {
        self.context_map.literal_context_map.slice()
    }
    #[inline]
    fn prediction_mode(&self) -> crate::interface::LiteralPredictionModeNibble {
        self.context_map.literal_prediction_mode()
    }
    #[inline]
    fn update_cost(
        &mut self,
        stride_prior: [u8; 8],
        stride_prior_offset: usize,
        selected_bits: u8,
        cm_prior: usize,
        literal: u8,
    ) {
        //let stride = self.cur_stride as usize;
        self.update_cost_base(
            stride_prior,
            stride_prior_offset,
            selected_bits,
            cm_prior,
            literal,
        )
    }
}

impl<'a, 'b, Alloc: alloc::Allocator<s16> + alloc::Allocator<u32> + alloc::Allocator<v8>>
    interface::CommandProcessor<'b> for PriorEval<'a, Alloc>
{
    #[inline]
    fn push(&mut self, val: interface::Command<InputReference<'b>>) {
        push_base(self, val)
    }
}
