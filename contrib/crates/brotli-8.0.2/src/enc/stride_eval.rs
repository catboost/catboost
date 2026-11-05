use core;

use super::super::alloc;
use super::super::alloc::{Allocator, SliceWrapper, SliceWrapperMut};
use super::backward_references::BrotliEncoderParams;
use super::input_pair::{InputPair, InputReference, InputReferenceMut};
use super::interface;
use super::ir_interpret::{push_base, IRInterpreter};
use super::prior_eval::DEFAULT_SPEED;
use super::util::{floatX, FastLog2u16};
use crate::enc::combined_alloc::{alloc_default, allocate};
const NIBBLE_PRIOR_SIZE: usize = 16;
pub const STRIDE_PRIOR_SIZE: usize = 256 * 256 * NIBBLE_PRIOR_SIZE * 2;

pub fn local_init_cdfs(cdfs: &mut [u16]) {
    for (index, item) in cdfs.iter_mut().enumerate() {
        *item = 4 + 4 * (index as u16 & 0x0f);
    }
}
#[allow(unused_variables)]
fn stride_lookup_lin(
    stride_byte: u8,
    selected_context: u8,
    actual_context: usize,
    high_nibble: Option<u8>,
) -> usize {
    if let Some(nibble) = high_nibble {
        1 + 2 * (actual_context | ((stride_byte as usize & 0xf) << 8) | ((nibble as usize) << 12))
    } else {
        2 * (actual_context | ((stride_byte as usize) << 8))
    }
}

struct CDF<'a> {
    cdf: &'a mut [u16],
}
struct Stride1Prior {}
impl Stride1Prior {
    fn lookup_lin(
        stride_byte: u8,
        selected_context: u8,
        actual_context: usize,
        high_nibble: Option<u8>,
    ) -> usize {
        stride_lookup_lin(stride_byte, selected_context, actual_context, high_nibble)
    }
    fn lookup_mut(
        data: &mut [u16],
        stride_byte: u8,
        selected_context: u8,
        actual_context: usize,
        high_nibble: Option<u8>,
    ) -> CDF {
        let index = Self::lookup_lin(stride_byte, selected_context, actual_context, high_nibble)
            * NIBBLE_PRIOR_SIZE;
        CDF::from(data.split_at_mut(index).1.split_at_mut(16).0)
    }
}

impl<'a> CDF<'a> {
    pub fn cost(&self, nibble_u8: u8) -> floatX {
        assert_eq!(self.cdf.len(), 16);
        let nibble = nibble_u8 as usize & 0xf;
        let mut pdf = self.cdf[nibble];
        if nibble_u8 != 0 {
            pdf -= self.cdf[nibble - 1];
        }
        FastLog2u16(self.cdf[15]) - FastLog2u16(pdf)
    }
    pub fn update(&mut self, nibble_u8: u8, speed: (u16, u16)) {
        assert_eq!(self.cdf.len(), 16);
        for nib_range in (nibble_u8 as usize & 0xf)..16 {
            self.cdf[nib_range] += speed.0;
        }
        if self.cdf[15] >= speed.1 {
            const CDF_BIAS: [u16; 16] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
            for nibble_index in 0..16 {
                let tmp = &mut self.cdf[nibble_index];
                *tmp = (tmp.wrapping_add(CDF_BIAS[nibble_index]))
                    .wrapping_sub(tmp.wrapping_add(CDF_BIAS[nibble_index]) >> 2);
            }
        }
    }
}

impl<'a> From<&'a mut [u16]> for CDF<'a> {
    fn from(cdf: &'a mut [u16]) -> CDF<'a> {
        assert_eq!(cdf.len(), 16);
        CDF { cdf }
    }
}

pub struct StrideEval<
    'a,
    Alloc: alloc::Allocator<u16> + alloc::Allocator<u32> + alloc::Allocator<floatX> + 'a,
> {
    input: InputPair<'a>,
    alloc: &'a mut Alloc,
    context_map: &'a interface::PredictionModeContextMap<InputReferenceMut<'a>>,
    block_type: u8,
    local_byte_offset: usize,
    stride_priors: [<Alloc as Allocator<u16>>::AllocatedMemory; 8],
    score: <Alloc as Allocator<floatX>>::AllocatedMemory,
    cur_score_epoch: usize,
    stride_speed: [(u16, u16); 2],
    cur_stride: u8,
}

impl<'a, Alloc: alloc::Allocator<u16> + alloc::Allocator<u32> + alloc::Allocator<floatX> + 'a>
    StrideEval<'a, Alloc>
{
    pub fn new(
        alloc: &'a mut Alloc,
        input: InputPair<'a>,
        prediction_mode: &'a interface::PredictionModeContextMap<InputReferenceMut<'a>>,
        params: &BrotliEncoderParams,
    ) -> Self {
        let do_alloc = true;
        let mut stride_speed = prediction_mode.stride_context_speed();
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
        let score = if do_alloc {
            allocate::<floatX, _>(alloc, 8 * 4) // FIXME make this bigger than just 4
        } else {
            alloc_default::<floatX, Alloc>()
        };
        let stride_priors = if do_alloc {
            [
                allocate::<u16, _>(alloc, STRIDE_PRIOR_SIZE),
                allocate::<u16, _>(alloc, STRIDE_PRIOR_SIZE),
                allocate::<u16, _>(alloc, STRIDE_PRIOR_SIZE),
                allocate::<u16, _>(alloc, STRIDE_PRIOR_SIZE),
                allocate::<u16, _>(alloc, STRIDE_PRIOR_SIZE),
                allocate::<u16, _>(alloc, STRIDE_PRIOR_SIZE),
                allocate::<u16, _>(alloc, STRIDE_PRIOR_SIZE),
                allocate::<u16, _>(alloc, STRIDE_PRIOR_SIZE),
            ]
        } else {
            [
                alloc_default::<u16, Alloc>(),
                alloc_default::<u16, Alloc>(),
                alloc_default::<u16, Alloc>(),
                alloc_default::<u16, Alloc>(),
                alloc_default::<u16, Alloc>(),
                alloc_default::<u16, Alloc>(),
                alloc_default::<u16, Alloc>(),
                alloc_default::<u16, Alloc>(),
            ]
        };
        let mut ret = StrideEval::<Alloc> {
            input,
            context_map: prediction_mode,
            block_type: 0,
            alloc,
            cur_stride: 1,
            cur_score_epoch: 0,
            local_byte_offset: 0,
            stride_priors,
            score,
            stride_speed,
        };
        for stride_prior in ret.stride_priors.iter_mut() {
            local_init_cdfs(stride_prior.slice_mut());
        }
        ret
    }
    pub fn alloc(&mut self) -> &mut Alloc {
        self.alloc
    }
    pub fn choose_stride(&self, stride_data: &mut [u8]) {
        assert_eq!(stride_data.len(), self.cur_score_epoch);
        assert!(self.score.slice().len() > stride_data.len());
        assert!(self.score.slice().len() > (stride_data.len() << 3) + 7 + 8);
        for (index, choice) in stride_data.iter_mut().enumerate() {
            let choices = self
                .score
                .slice()
                .split_at((1 + index) << 3)
                .1
                .split_at(8)
                .0;
            let mut best_choice: u8 = 0;
            let mut best_score = choices[0];
            for (cur_index, cur_score) in choices.iter().enumerate() {
                if *cur_score + 2.0 < best_score {
                    // needs to be 2 bits better to be worth the type switch
                    best_score = *cur_score;
                    best_choice = cur_index as u8;
                }
            }
            *choice = best_choice;
        }
    }
    pub fn num_types(&self) -> usize {
        self.cur_score_epoch
    }
    fn update_cost_base(
        &mut self,
        stride_prior: [u8; 8],
        selected_bits: u8,
        cm_prior: usize,
        literal: u8,
    ) {
        type CurPrior = Stride1Prior;
        {
            for i in 0..8 {
                let mut cdf = CurPrior::lookup_mut(
                    self.stride_priors[i].slice_mut(),
                    stride_prior[i],
                    selected_bits,
                    cm_prior,
                    None,
                );
                self.score.slice_mut()[self.cur_score_epoch * 8 + i] += cdf.cost(literal >> 4);
                cdf.update(literal >> 4, self.stride_speed[1]);
            }
        }
        {
            for i in 0..8 {
                let mut cdf = CurPrior::lookup_mut(
                    self.stride_priors[i].slice_mut(),
                    stride_prior[i],
                    selected_bits,
                    cm_prior,
                    Some(literal >> 4),
                );
                self.score.slice_mut()[self.cur_score_epoch * 8 + i] += cdf.cost(literal & 0xf);
                cdf.update(literal & 0xf, self.stride_speed[0]);
            }
        }
    }
}
impl<'a, Alloc: alloc::Allocator<u16> + alloc::Allocator<u32> + alloc::Allocator<floatX>> Drop
    for StrideEval<'a, Alloc>
{
    fn drop(&mut self) {
        <Alloc as Allocator<floatX>>::free_cell(self.alloc, core::mem::take(&mut self.score));
        for i in 0..8 {
            <Alloc as Allocator<u16>>::free_cell(
                self.alloc,
                core::mem::take(&mut self.stride_priors[i]),
            );
        }
    }
}

impl<'a, Alloc: alloc::Allocator<u16> + alloc::Allocator<u32> + alloc::Allocator<floatX>>
    IRInterpreter for StrideEval<'a, Alloc>
{
    fn inc_local_byte_offset(&mut self, inc: usize) {
        self.local_byte_offset += inc;
    }
    fn local_byte_offset(&self) -> usize {
        self.local_byte_offset
    }
    fn update_block_type(&mut self, new_type: u8, stride: u8) {
        self.block_type = new_type;
        self.cur_stride = stride;
        self.cur_score_epoch += 1;
        if self.cur_score_epoch * 8 + 7 >= self.score.slice().len() {
            let new_len = self.score.slice().len() * 2;
            let mut new_score = allocate::<floatX, _>(self.alloc, new_len);
            for (src, dst) in self.score.slice().iter().zip(
                new_score
                    .slice_mut()
                    .split_at_mut(self.score.slice().len())
                    .0
                    .iter_mut(),
            ) {
                *dst = *src;
            }
            <Alloc as Allocator<floatX>>::free_cell(
                self.alloc,
                core::mem::replace(&mut self.score, new_score),
            );
        }
    }
    fn block_type(&self) -> u8 {
        self.block_type
    }
    fn literal_data_at_offset(&self, index: usize) -> u8 {
        self.input[index]
    }
    fn literal_context_map(&self) -> &[u8] {
        self.context_map.literal_context_map.slice()
    }
    fn prediction_mode(&self) -> crate::interface::LiteralPredictionModeNibble {
        self.context_map.literal_prediction_mode()
    }
    fn update_cost(
        &mut self,
        stride_prior: [u8; 8],
        stride_prior_offset: usize,
        selected_bits: u8,
        cm_prior: usize,
        literal: u8,
    ) {
        let reversed_stride_priors = [
            stride_prior[stride_prior_offset & 7],
            stride_prior[stride_prior_offset.wrapping_sub(1) & 7],
            stride_prior[stride_prior_offset.wrapping_sub(2) & 7],
            stride_prior[stride_prior_offset.wrapping_sub(3) & 7],
            stride_prior[stride_prior_offset.wrapping_sub(4) & 7],
            stride_prior[stride_prior_offset.wrapping_sub(5) & 7],
            stride_prior[stride_prior_offset.wrapping_sub(6) & 7],
            stride_prior[stride_prior_offset.wrapping_sub(7) & 7],
        ];
        self.update_cost_base(reversed_stride_priors, selected_bits, cm_prior, literal)
    }
}

impl<'a, 'b, Alloc: alloc::Allocator<u16> + alloc::Allocator<u32> + alloc::Allocator<floatX>>
    interface::CommandProcessor<'b> for StrideEval<'a, Alloc>
{
    fn push(&mut self, val: interface::Command<InputReference<'b>>) {
        push_base(self, val)
    }
}
