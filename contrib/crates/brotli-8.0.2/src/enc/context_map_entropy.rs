use core;

use super::super::alloc;
use super::super::alloc::{Allocator, SliceWrapper, SliceWrapperMut};
use super::input_pair::{InputPair, InputReference, InputReferenceMut};
pub use super::ir_interpret::{push_base, Context, IRInterpreter};
use super::util::{floatX, FastLog2u16};
use super::weights::{Weights, BLEND_FIXED_POINT_PRECISION};
use super::{find_stride, interface};
use crate::enc::combined_alloc::alloc_if;

const DEFAULT_CM_SPEED_INDEX: usize = 8;
const NUM_SPEEDS_TO_TRY: usize = 16;
const SPEEDS_TO_SEARCH: [u16; NUM_SPEEDS_TO_TRY] = [
    0, 1, 1, 1, 2, 4, 8, 16, 16, 32, 64, 128, 128, 512, 1664, 1664,
];
const MAXES_TO_SEARCH: [u16; NUM_SPEEDS_TO_TRY] = [
    32, 32, 128, 16384, 1024, 1024, 8192, 48, 8192, 4096, 16384, 256, 16384, 16384, 16384, 16384,
];
const NIBBLE_PRIOR_SIZE: usize = 16 * NUM_SPEEDS_TO_TRY;
// the high nibble, followed by the low nibbles
const CONTEXT_MAP_PRIOR_SIZE: usize = 256 * NIBBLE_PRIOR_SIZE * 17;
const STRIDE_PRIOR_SIZE: usize = 256 * 256 * NIBBLE_PRIOR_SIZE * 2;
#[derive(Clone, Copy, Debug)]
pub struct SpeedAndMax(pub u16, pub u16);

pub fn speed_to_tuple(inp: [SpeedAndMax; 2]) -> [(u16, u16); 2] {
    [(inp[0].0, inp[0].1), (inp[1].0, inp[1].1)]
}

fn get_stride_cdf_low(
    data: &mut [u16],
    stride_prior: u8,
    cm_prior: usize,
    high_nibble: u8,
) -> &mut [u16] {
    let index: usize =
        1 + 2 * (cm_prior | ((stride_prior as usize & 0xf) << 8) | ((high_nibble as usize) << 12));
    data.split_at_mut((NUM_SPEEDS_TO_TRY * index) << 4)
        .1
        .split_at_mut(16 * NUM_SPEEDS_TO_TRY)
        .0
}

fn get_stride_cdf_high(data: &mut [u16], stride_prior: u8, cm_prior: usize) -> &mut [u16] {
    let index: usize = 2 * (cm_prior | ((stride_prior as usize) << 8));
    data.split_at_mut((NUM_SPEEDS_TO_TRY * index) << 4)
        .1
        .split_at_mut(16 * NUM_SPEEDS_TO_TRY)
        .0
}

fn get_cm_cdf_low(data: &mut [u16], cm_prior: usize, high_nibble: u8) -> &mut [u16] {
    let index: usize = (high_nibble as usize + 1) + 17 * cm_prior;
    data.split_at_mut((NUM_SPEEDS_TO_TRY * index) << 4)
        .1
        .split_at_mut(16 * NUM_SPEEDS_TO_TRY)
        .0
}

fn get_cm_cdf_high(data: &mut [u16], cm_prior: usize) -> &mut [u16] {
    let index: usize = 17 * cm_prior;
    data.split_at_mut((NUM_SPEEDS_TO_TRY * index) << 4)
        .1
        .split_at_mut(16 * NUM_SPEEDS_TO_TRY)
        .0
}
fn init_cdfs(cdfs: &mut [u16]) {
    assert_eq!(cdfs.len() % (16 * NUM_SPEEDS_TO_TRY), 0);
    let mut total_index = 0usize;
    let len = cdfs.len();
    loop {
        for cdf_index in 0..16 {
            let vec = cdfs
                .split_at_mut(total_index)
                .1
                .split_at_mut(NUM_SPEEDS_TO_TRY)
                .0;
            for item in vec {
                *item = 4 + 4 * cdf_index as u16;
            }
            total_index += NUM_SPEEDS_TO_TRY;
        }
        if total_index == len {
            break;
        }
    }
}
fn compute_combined_cost(
    singleton_cost: &mut [floatX; NUM_SPEEDS_TO_TRY],
    cdfs: &[u16],
    mixing_cdf: [u16; 16],
    nibble_u8: u8,
    _weights: &mut [Weights; NUM_SPEEDS_TO_TRY],
) {
    assert_eq!(cdfs.len(), 16 * NUM_SPEEDS_TO_TRY);
    let nibble = nibble_u8 as usize & 0xf;
    let mut stride_pdf = [0u16; NUM_SPEEDS_TO_TRY];
    stride_pdf.clone_from_slice(
        cdfs.split_at(NUM_SPEEDS_TO_TRY * nibble)
            .1
            .split_at(NUM_SPEEDS_TO_TRY)
            .0,
    );
    let mut cm_pdf: u16 = mixing_cdf[nibble];
    if nibble_u8 != 0 {
        let mut tmp = [0u16; NUM_SPEEDS_TO_TRY];
        tmp.clone_from_slice(
            cdfs.split_at(NUM_SPEEDS_TO_TRY * (nibble - 1))
                .1
                .split_at(NUM_SPEEDS_TO_TRY)
                .0,
        );
        for i in 0..NUM_SPEEDS_TO_TRY {
            stride_pdf[i] -= tmp[i];
        }
        cm_pdf -= mixing_cdf[nibble - 1]
    }
    let mut stride_max = [0u16; NUM_SPEEDS_TO_TRY];
    stride_max.clone_from_slice(cdfs.split_at(NUM_SPEEDS_TO_TRY * 15).1);
    let cm_max = mixing_cdf[15];
    for i in 0..NUM_SPEEDS_TO_TRY {
        if stride_pdf[i] == 0 {
            assert_ne!(stride_pdf[i], 0);
        }
        if stride_max[i] == 0 {
            assert_ne!(stride_max[i], 0);
        }

        let w = (1 << (BLEND_FIXED_POINT_PRECISION - 2)); // a quarter of weight to stride
        let combined_pdf = w * u32::from(stride_pdf[i])
            + ((1 << BLEND_FIXED_POINT_PRECISION) - w) * u32::from(cm_pdf);
        let combined_max = w * u32::from(stride_max[i])
            + ((1 << BLEND_FIXED_POINT_PRECISION) - w) * u32::from(cm_max);
        let del = FastLog2u16((combined_pdf >> BLEND_FIXED_POINT_PRECISION) as u16)
            - FastLog2u16((combined_max >> BLEND_FIXED_POINT_PRECISION) as u16);
        singleton_cost[i] -= del;
    }
}
fn compute_cost(singleton_cost: &mut [floatX; NUM_SPEEDS_TO_TRY], cdfs: &[u16], nibble_u8: u8) {
    assert_eq!(cdfs.len(), 16 * NUM_SPEEDS_TO_TRY);
    let nibble = nibble_u8 as usize & 0xf;
    let mut pdf = [0u16; NUM_SPEEDS_TO_TRY];
    pdf.clone_from_slice(
        cdfs.split_at(NUM_SPEEDS_TO_TRY * nibble)
            .1
            .split_at(NUM_SPEEDS_TO_TRY)
            .0,
    );
    if nibble_u8 != 0 {
        let mut tmp = [0u16; NUM_SPEEDS_TO_TRY];
        tmp.clone_from_slice(
            cdfs.split_at(NUM_SPEEDS_TO_TRY * (nibble - 1))
                .1
                .split_at(NUM_SPEEDS_TO_TRY)
                .0,
        );
        for i in 0..NUM_SPEEDS_TO_TRY {
            pdf[i] -= tmp[i];
        }
    }
    let mut max = [0u16; NUM_SPEEDS_TO_TRY];
    max.clone_from_slice(cdfs.split_at(NUM_SPEEDS_TO_TRY * 15).1);
    for i in 0..NUM_SPEEDS_TO_TRY {
        if pdf[i] == 0 {
            assert_ne!(pdf[i], 0);
        }
        if max[i] == 0 {
            assert_ne!(max[i], 0);
        }
        let del = FastLog2u16(pdf[i]) - FastLog2u16(max[i]);
        singleton_cost[i] -= del;
    }
}
fn update_cdf(cdfs: &mut [u16], nibble_u8: u8) {
    assert_eq!(cdfs.len(), 16 * NUM_SPEEDS_TO_TRY);
    let mut overall_index = nibble_u8 as usize * NUM_SPEEDS_TO_TRY;
    for _nibble in (nibble_u8 as usize & 0xf)..16 {
        for speed_index in 0..NUM_SPEEDS_TO_TRY {
            cdfs[overall_index + speed_index] += SPEEDS_TO_SEARCH[speed_index];
        }
        overall_index += NUM_SPEEDS_TO_TRY;
    }
    overall_index = 0;
    for nibble in 0..16 {
        for speed_index in 0..NUM_SPEEDS_TO_TRY {
            if nibble == 0 {
                assert_ne!(cdfs[overall_index + speed_index], 0);
            } else {
                assert_ne!(
                    cdfs[overall_index + speed_index]
                        - cdfs[overall_index + speed_index - NUM_SPEEDS_TO_TRY],
                    0
                );
            }
        }
        overall_index += NUM_SPEEDS_TO_TRY;
    }
    for max_index in 0..NUM_SPEEDS_TO_TRY {
        if cdfs[15 * NUM_SPEEDS_TO_TRY + max_index] >= MAXES_TO_SEARCH[max_index] {
            const CDF_BIAS: [u16; 16] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
            for nibble_index in 0..16 {
                let tmp = &mut cdfs[nibble_index * NUM_SPEEDS_TO_TRY + max_index];
                *tmp = (tmp.wrapping_add(CDF_BIAS[nibble_index]))
                    .wrapping_sub(tmp.wrapping_add(CDF_BIAS[nibble_index]) >> 2);
            }
        }
    }
    overall_index = 0;
    for nibble in 0..16 {
        for speed_index in 0..NUM_SPEEDS_TO_TRY {
            if nibble == 0 {
                assert_ne!(cdfs[overall_index + speed_index], 0);
            } else {
                assert_ne!(
                    cdfs[overall_index + speed_index]
                        - cdfs[overall_index + speed_index - NUM_SPEEDS_TO_TRY],
                    0
                );
            }
        }
        overall_index += NUM_SPEEDS_TO_TRY;
    }
}

fn extract_single_cdf(cdf_bundle: &[u16], index: usize) -> [u16; 16] {
    assert_eq!(cdf_bundle.len(), 16 * NUM_SPEEDS_TO_TRY);
    assert!(index < NUM_SPEEDS_TO_TRY);

    #[allow(clippy::identity_op)]
    [
        cdf_bundle[index + 0 * NUM_SPEEDS_TO_TRY],
        cdf_bundle[index + 1 * NUM_SPEEDS_TO_TRY],
        cdf_bundle[index + 2 * NUM_SPEEDS_TO_TRY],
        cdf_bundle[index + 3 * NUM_SPEEDS_TO_TRY],
        cdf_bundle[index + 4 * NUM_SPEEDS_TO_TRY],
        cdf_bundle[index + 5 * NUM_SPEEDS_TO_TRY],
        cdf_bundle[index + 6 * NUM_SPEEDS_TO_TRY],
        cdf_bundle[index + 7 * NUM_SPEEDS_TO_TRY],
        cdf_bundle[index + 8 * NUM_SPEEDS_TO_TRY],
        cdf_bundle[index + 9 * NUM_SPEEDS_TO_TRY],
        cdf_bundle[index + 10 * NUM_SPEEDS_TO_TRY],
        cdf_bundle[index + 11 * NUM_SPEEDS_TO_TRY],
        cdf_bundle[index + 12 * NUM_SPEEDS_TO_TRY],
        cdf_bundle[index + 13 * NUM_SPEEDS_TO_TRY],
        cdf_bundle[index + 14 * NUM_SPEEDS_TO_TRY],
        cdf_bundle[index + 15 * NUM_SPEEDS_TO_TRY],
    ]
}

fn min_cost_index_for_speed(cost: &[floatX]) -> usize {
    assert_eq!(cost.len(), NUM_SPEEDS_TO_TRY);
    let mut min_cost = cost[0];
    let mut best_choice = 0;
    for i in 1..NUM_SPEEDS_TO_TRY {
        if cost[i] < min_cost {
            best_choice = i;
            min_cost = cost[i];
        }
    }
    best_choice
}
fn min_cost_speed_max(cost: &[floatX]) -> SpeedAndMax {
    let best_choice = min_cost_index_for_speed(cost);
    SpeedAndMax(SPEEDS_TO_SEARCH[best_choice], MAXES_TO_SEARCH[best_choice])
}

fn min_cost_value(cost: &[floatX]) -> floatX {
    let best_choice = min_cost_index_for_speed(cost);
    cost[best_choice]
}

const SINGLETON_COMBINED_STRATEGY: usize = 2;
const SINGLETON_STRIDE_STRATEGY: usize = 1;
const SINGLETON_CM_STRATEGY: usize = 0;

pub struct ContextMapEntropy<
    'a,
    Alloc: alloc::Allocator<u16> + alloc::Allocator<u32> + alloc::Allocator<floatX>,
> {
    input: InputPair<'a>,
    context_map: interface::PredictionModeContextMap<InputReferenceMut<'a>>,
    block_type: u8,
    cur_stride: u8,
    local_byte_offset: usize,
    weight: [[Weights; NUM_SPEEDS_TO_TRY]; 2],

    cm_priors: <Alloc as Allocator<u16>>::AllocatedMemory,
    stride_priors: <Alloc as Allocator<u16>>::AllocatedMemory,
    _stride_pyramid_leaves: [u8; find_stride::NUM_LEAF_NODES],
    singleton_costs: [[[floatX; NUM_SPEEDS_TO_TRY]; 2]; 3],
}
impl<'a, Alloc: alloc::Allocator<u16> + alloc::Allocator<u32> + alloc::Allocator<floatX>>
    ContextMapEntropy<'a, Alloc>
{
    pub fn new(
        m16: &mut Alloc,
        input: InputPair<'a>,
        stride: [u8; find_stride::NUM_LEAF_NODES],
        prediction_mode: interface::PredictionModeContextMap<InputReferenceMut<'a>>,
        cdf_detection_quality: u8,
    ) -> Self {
        let cdf_detect = cdf_detection_quality != 0;
        let mut ret = ContextMapEntropy::<Alloc> {
            input,
            context_map: prediction_mode,
            block_type: 0,
            cur_stride: 1,
            local_byte_offset: 0,
            cm_priors: alloc_if::<u16, _>(cdf_detect, m16, CONTEXT_MAP_PRIOR_SIZE),
            stride_priors: alloc_if::<u16, _>(cdf_detect, m16, STRIDE_PRIOR_SIZE),
            _stride_pyramid_leaves: stride,
            weight: [
                [Weights::new(); NUM_SPEEDS_TO_TRY],
                [Weights::new(); NUM_SPEEDS_TO_TRY],
            ],
            singleton_costs: [[[0.0; NUM_SPEEDS_TO_TRY]; 2]; 3],
        };
        if cdf_detect {
            init_cdfs(ret.cm_priors.slice_mut());
            init_cdfs(ret.stride_priors.slice_mut());
        }
        ret
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
    pub fn prediction_mode_mut(
        &mut self,
    ) -> &mut interface::PredictionModeContextMap<InputReferenceMut<'a>> {
        &mut self.context_map
    }
    pub fn best_singleton_speeds(
        &self,
        cm: bool,
        combined: bool,
    ) -> ([SpeedAndMax; 2], [floatX; 2]) {
        let cost_type_index = if combined {
            2usize
        } else if cm {
            0usize
        } else {
            1
        };
        let mut ret_cost = [
            self.singleton_costs[cost_type_index][0][0],
            self.singleton_costs[cost_type_index][1][0],
        ];
        let mut best_indexes = [0, 0];
        for speed_index in 1..NUM_SPEEDS_TO_TRY {
            for highness in 0..2 {
                let cur_cost = self.singleton_costs[cost_type_index][highness][speed_index];
                if cur_cost < ret_cost[highness] {
                    best_indexes[highness] = speed_index;
                    ret_cost[highness] = cur_cost;
                }
            }
        }
        let ret_speed = [
            SpeedAndMax(
                SPEEDS_TO_SEARCH[best_indexes[0]],
                MAXES_TO_SEARCH[best_indexes[0]],
            ),
            SpeedAndMax(
                SPEEDS_TO_SEARCH[best_indexes[1]],
                MAXES_TO_SEARCH[best_indexes[1]],
            ),
        ];
        (ret_speed, ret_cost)
    }
    pub fn best_speeds(
        &mut self, // mut due to helpers
        cm: bool,
        combined: bool,
    ) -> [SpeedAndMax; 2] {
        let mut ret = [SpeedAndMax(SPEEDS_TO_SEARCH[0], MAXES_TO_SEARCH[0]); 2];
        let cost_type_index = if combined {
            2usize
        } else if cm {
            0usize
        } else {
            1
        };
        for high in 0..2 {
            /*eprintln!("TRIAL {} {}", cm, combined);
            for i in 0..NUM_SPEEDS_TO_TRY {
                eprintln!("{},{} costs {:?}", SPEEDS_TO_SEARCH[i], MAXES_TO_SEARCH[i], self.singleton_costs[cost_type_index][high][i]);
            }*/
            ret[high] = min_cost_speed_max(&self.singleton_costs[cost_type_index][high][..]);
        }
        ret
    }
    pub fn best_speeds_costs(
        &mut self, // mut due to helpers
        cm: bool,
        combined: bool,
    ) -> [floatX; 2] {
        let cost_type_index = if combined {
            2usize
        } else if cm {
            0usize
        } else {
            1
        };
        let mut ret = [0.0; 2];
        for high in 0..2 {
            ret[high] = min_cost_value(&self.singleton_costs[cost_type_index][high][..]);
        }
        ret
    }
    pub fn free(&mut self, alloc: &mut Alloc) {
        <Alloc as Allocator<u16>>::free_cell(alloc, core::mem::take(&mut self.cm_priors));
        <Alloc as Allocator<u16>>::free_cell(alloc, core::mem::take(&mut self.stride_priors));
    }
    fn update_cost_base(
        &mut self,
        stride_prior: u8,
        _selected_bits: u8,
        cm_prior: usize,
        literal: u8,
    ) {
        let upper_nibble = (literal >> 4);
        let lower_nibble = literal & 0xf;
        let provisional_cm_high_cdf: [u16; 16];
        let provisional_cm_low_cdf: [u16; 16];
        {
            let cm_cdf_high = get_cm_cdf_high(self.cm_priors.slice_mut(), cm_prior);
            compute_cost(
                &mut self.singleton_costs[SINGLETON_CM_STRATEGY][1],
                cm_cdf_high,
                upper_nibble,
            );
            // choose a fairly reasonable cm speed rather than a selected one
            let best_cm_index = DEFAULT_CM_SPEED_INDEX; // = min_cost_index_for_speed(&self.singleton_costs[SINGLETON_CM_STRATEGY][1]);
            provisional_cm_high_cdf = extract_single_cdf(cm_cdf_high, best_cm_index);
        }
        {
            let cm_cdf_low = get_cm_cdf_low(self.cm_priors.slice_mut(), cm_prior, upper_nibble);
            compute_cost(
                &mut self.singleton_costs[SINGLETON_CM_STRATEGY][0],
                cm_cdf_low,
                lower_nibble,
            );
            // choose a fairly reasonable cm speed rather than a selected one
            let best_cm_index = DEFAULT_CM_SPEED_INDEX; //min_cost_index_for_speed(&self.singleton_costs[SINGLETON_CM_STRATEGY][0]);
            provisional_cm_low_cdf = extract_single_cdf(cm_cdf_low, best_cm_index);
        }
        {
            let stride_cdf_high =
                get_stride_cdf_high(self.stride_priors.slice_mut(), stride_prior, cm_prior);
            compute_combined_cost(
                &mut self.singleton_costs[SINGLETON_COMBINED_STRATEGY][1],
                stride_cdf_high,
                provisional_cm_high_cdf,
                upper_nibble,
                &mut self.weight[1],
            );
            compute_cost(
                &mut self.singleton_costs[SINGLETON_STRIDE_STRATEGY][1],
                stride_cdf_high,
                upper_nibble,
            );
            update_cdf(stride_cdf_high, upper_nibble);
        }
        {
            let stride_cdf_low = get_stride_cdf_low(
                self.stride_priors.slice_mut(),
                stride_prior,
                cm_prior,
                upper_nibble,
            );
            compute_combined_cost(
                &mut self.singleton_costs[SINGLETON_COMBINED_STRATEGY][0],
                stride_cdf_low,
                provisional_cm_low_cdf,
                lower_nibble,
                &mut self.weight[0],
            );
            compute_cost(
                &mut self.singleton_costs[SINGLETON_STRIDE_STRATEGY][0],
                stride_cdf_low,
                lower_nibble,
            );
            update_cdf(stride_cdf_low, lower_nibble);
        }
        {
            let cm_cdf_high = get_cm_cdf_high(self.cm_priors.slice_mut(), cm_prior);
            update_cdf(cm_cdf_high, upper_nibble);
        }
        {
            let cm_cdf_low = get_cm_cdf_low(self.cm_priors.slice_mut(), cm_prior, upper_nibble);
            update_cdf(cm_cdf_low, lower_nibble);
        }
    }
}

impl<'a, 'b, Alloc: alloc::Allocator<u16> + alloc::Allocator<u32> + alloc::Allocator<floatX>>
    interface::CommandProcessor<'b> for ContextMapEntropy<'a, Alloc>
{
    fn push(&mut self, val: interface::Command<InputReference<'b>>) {
        push_base(self, val)
    }
}

impl<'a, Alloc: alloc::Allocator<u16> + alloc::Allocator<u32> + alloc::Allocator<floatX>>
    IRInterpreter for ContextMapEntropy<'a, Alloc>
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
        let stride = self.cur_stride as usize;
        self.update_cost_base(
            stride_prior[stride_prior_offset.wrapping_sub(stride) & 7],
            selected_bits,
            cm_prior,
            literal,
        )
    }
}
