use alloc::{Allocator, SliceWrapper, SliceWrapperMut};
use core;

use super::histogram;
pub use super::input_pair::{InputPair, InputReference, InputReferenceMut};

#[derive(Debug, Copy, Clone, Default)]
pub struct BlockSwitch(pub u8);

/// Commands that can instantiate as a no-op should implement this.
pub trait Nop<T> {
    fn nop() -> T;
}

impl BlockSwitch {
    #[inline(always)]
    pub fn new(block_type: u8) -> Self {
        BlockSwitch(block_type)
    }
    #[inline(always)]
    pub fn block_type(&self) -> u8 {
        self.0
    }
}

#[derive(Debug, Copy, Clone, Default)]
pub struct LiteralBlockSwitch(pub BlockSwitch, pub u8);

impl LiteralBlockSwitch {
    pub fn new(block_type: u8, stride: u8) -> Self {
        LiteralBlockSwitch(BlockSwitch::new(block_type), stride)
    }
    #[inline(always)]
    pub fn block_type(&self) -> u8 {
        self.0.block_type()
    }
    #[inline(always)]
    pub fn stride(&self) -> u8 {
        self.1
    }
    #[inline(always)]
    pub fn update_stride(&mut self, new_stride: u8) {
        self.1 = new_stride;
    }
}

pub const LITERAL_PREDICTION_MODE_SIGN: u8 = 3;
pub const LITERAL_PREDICTION_MODE_UTF8: u8 = 2;
pub const LITERAL_PREDICTION_MODE_MSB6: u8 = 1;
pub const LITERAL_PREDICTION_MODE_LSB6: u8 = 0;

#[derive(Default, Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct LiteralPredictionModeNibble(pub u8);

impl LiteralPredictionModeNibble {
    #[inline(always)]
    pub fn new(prediction_mode: u8) -> Result<Self, ()> {
        if prediction_mode < 16 {
            return Ok(LiteralPredictionModeNibble(prediction_mode));
        }
        Err(())
    }
    #[inline(always)]
    pub fn prediction_mode(&self) -> u8 {
        self.0
    }
    #[inline(always)]
    pub fn signed() -> Self {
        LiteralPredictionModeNibble(LITERAL_PREDICTION_MODE_SIGN)
    }
    #[inline(always)]
    pub fn utf8() -> Self {
        LiteralPredictionModeNibble(LITERAL_PREDICTION_MODE_UTF8)
    }
    #[inline(always)]
    pub fn msb6() -> Self {
        LiteralPredictionModeNibble(LITERAL_PREDICTION_MODE_MSB6)
    }
    #[inline(always)]
    pub fn lsb6() -> Self {
        LiteralPredictionModeNibble(LITERAL_PREDICTION_MODE_LSB6)
    }
    #[inline(always)]
    pub fn to_context_enum(&self) -> Result<histogram::ContextType, ()> {
        match self.0 {
            LITERAL_PREDICTION_MODE_LSB6 => Ok(histogram::ContextType::CONTEXT_LSB6),
            LITERAL_PREDICTION_MODE_MSB6 => Ok(histogram::ContextType::CONTEXT_MSB6),
            LITERAL_PREDICTION_MODE_UTF8 => Ok(histogram::ContextType::CONTEXT_UTF8),
            LITERAL_PREDICTION_MODE_SIGN => Ok(histogram::ContextType::CONTEXT_SIGNED),
            _ => Err(()),
        }
    }
}
pub const NUM_SPEED_VALUES: usize = 12;
pub const NUM_MIXING_VALUES: usize = 16 * 256 + 16 * 256;
pub const NUM_PREDMODE_SETUP_VALUES: usize = 4;
pub const RESERVED_OFFSET: usize = 3;
pub const ADV_CONTEXT_MAP_OFFSET: usize = 2;
pub const MIXING_MATH_OFFSET: usize = 1;
pub const PREDMODE_OFFSET: usize = 0;
pub const MIXING_OFFSET: usize = NUM_PREDMODE_SETUP_VALUES + PREDMODE_OFFSET;
pub const SPEED_OFFSET: usize = MIXING_OFFSET + NUM_MIXING_VALUES;
pub const DISTANCE_CONTEXT_MAP_OFFSET: usize = SPEED_OFFSET + NUM_SPEED_VALUES;
pub const MAX_PREDMODE_SPEED_AND_DISTANCE_CONTEXT_MAP_SIZE: usize =
    DISTANCE_CONTEXT_MAP_OFFSET + 256 * 4;
pub const MAX_LITERAL_CONTEXT_MAP_SIZE: usize = 256 * 64;
pub const MAX_ADV_LITERAL_CONTEXT_MAP_SIZE: usize = 256 * 64 * 2;
#[derive(Debug)]
pub struct PredictionModeContextMap<SliceType: SliceWrapper<u8>> {
    pub literal_context_map: SliceType,
    pub predmode_speed_and_distance_context_map: SliceType,
}
impl<SliceType: SliceWrapper<u8> + SliceWrapperMut<u8>> PredictionModeContextMap<SliceType> {
    #[inline]
    pub fn distance_context_map_mut(&mut self) -> &mut [u8] {
        self.predmode_speed_and_distance_context_map
            .slice_mut()
            .split_at_mut(DISTANCE_CONTEXT_MAP_OFFSET)
            .1
    }
    #[inline]
    pub fn set_stride_context_speed(&mut self, speed_max: [(u16, u16); 2]) {
        let cm_slice = self.predmode_speed_and_distance_context_map.slice_mut();
        for high in 0..2 {
            cm_slice[Self::stride_context_speed_offset() + high] =
                Self::u16_to_f8(speed_max[high].0);
            cm_slice[Self::stride_context_speed_max_offset() + high] =
                Self::u16_to_f8(speed_max[high].1);
        }
    }
    #[inline]
    pub fn set_context_map_speed(&mut self, speed_max: [(u16, u16); 2]) {
        let cm_slice = self.predmode_speed_and_distance_context_map.slice_mut();
        for high in 0..2 {
            cm_slice[Self::context_map_speed_offset() + high] = Self::u16_to_f8(speed_max[high].0);
            cm_slice[Self::context_map_speed_max_offset() + high] =
                Self::u16_to_f8(speed_max[high].1);
        }
    }
    pub fn set_mixing_math(&mut self, math_enum: u8) {
        let cm_slice = self.predmode_speed_and_distance_context_map.slice_mut();
        cm_slice[MIXING_MATH_OFFSET] = math_enum;
    }
    pub fn set_adv_context_map(&mut self, is_adv: u8) {
        let cm_slice = self.predmode_speed_and_distance_context_map.slice_mut();
        cm_slice[ADV_CONTEXT_MAP_OFFSET] = is_adv;
    }
    #[inline]
    pub fn set_mixing_values(&mut self, mixing_mask: &[u8; NUM_MIXING_VALUES]) {
        let cm_slice = self.predmode_speed_and_distance_context_map.slice_mut();
        cm_slice[MIXING_OFFSET..(MIXING_OFFSET + NUM_MIXING_VALUES)]
            .clone_from_slice(&mixing_mask[..]);
    }
    #[inline]
    pub fn get_mixing_values_mut(&mut self) -> &mut [u8] {
        let cm_slice = self.predmode_speed_and_distance_context_map.slice_mut();
        &mut cm_slice[MIXING_OFFSET..(MIXING_OFFSET + NUM_MIXING_VALUES)]
    }
    #[inline]
    pub fn set_combined_stride_context_speed(&mut self, speed_max: [(u16, u16); 2]) {
        let cm_slice = self.predmode_speed_and_distance_context_map.slice_mut();
        for high in 0..2 {
            cm_slice[Self::combined_stride_context_speed_offset() + high] =
                Self::u16_to_f8(speed_max[high].0);
            cm_slice[Self::combined_stride_context_speed_max_offset() + high] =
                Self::u16_to_f8(speed_max[high].1);
        }
    }
    pub fn set_literal_prediction_mode(&mut self, val: LiteralPredictionModeNibble) {
        let cm_slice = self.predmode_speed_and_distance_context_map.slice_mut();
        cm_slice[PREDMODE_OFFSET] = val.0;
    }
}
impl<SliceType: SliceWrapper<u8>> PredictionModeContextMap<SliceType> {
    #[inline]
    pub fn from_mut<Other: SliceWrapper<u8>>(
        other: PredictionModeContextMap<Other>,
    ) -> PredictionModeContextMap<SliceType>
    where
        SliceType: From<Other>,
    {
        PredictionModeContextMap::<SliceType> {
            literal_context_map: SliceType::from(other.literal_context_map),
            predmode_speed_and_distance_context_map: SliceType::from(
                other.predmode_speed_and_distance_context_map,
            ),
        }
    }
    #[inline]
    pub fn get_mixing_values(&self) -> &[u8] {
        let cm_slice = self.predmode_speed_and_distance_context_map.slice();
        &cm_slice[MIXING_OFFSET..(MIXING_OFFSET + NUM_MIXING_VALUES)]
    }
    #[inline]
    pub fn get_mixing_math(&self) -> u8 {
        let cm_slice = self.predmode_speed_and_distance_context_map.slice();
        if cm_slice.len() <= MIXING_MATH_OFFSET {
            return 1;
        }
        cm_slice[MIXING_MATH_OFFSET]
    }
    #[inline]
    pub fn get_is_adv_context_map(&self) -> u8 {
        let cm_slice = self.predmode_speed_and_distance_context_map.slice();
        if cm_slice.len() <= ADV_CONTEXT_MAP_OFFSET {
            return 0;
        }
        cm_slice[ADV_CONTEXT_MAP_OFFSET]
    }
    #[inline]
    pub fn has_context_speeds(&self) -> bool {
        self.predmode_speed_and_distance_context_map.slice().len() >= DISTANCE_CONTEXT_MAP_OFFSET
    }
    #[inline]
    pub fn size_of_combined_array(distance_context_map_size: usize) -> usize {
        distance_context_map_size + DISTANCE_CONTEXT_MAP_OFFSET
    }
    #[inline]
    pub fn context_speeds_standard_len(&self) -> usize {
        NUM_SPEED_VALUES
    }
    #[inline]
    pub fn context_speeds_f8(&self) -> &[u8] {
        &self.predmode_speed_and_distance_context_map.slice()
            [SPEED_OFFSET..DISTANCE_CONTEXT_MAP_OFFSET]
    }
    #[inline]
    pub fn distance_context_map(&self) -> &[u8] {
        self.predmode_speed_and_distance_context_map
            .slice()
            .split_at(DISTANCE_CONTEXT_MAP_OFFSET)
            .1
    }
    #[inline]
    pub fn f8_to_u16(data: u8) -> u16 {
        self::u8_to_speed(data)
    }
    #[inline]
    pub fn u16_to_f8(data: u16) -> u8 {
        self::speed_to_u8(data)
    }
    #[inline]
    pub fn stride_context_speed_offset() -> usize {
        SPEED_OFFSET
    }
    #[inline]
    pub fn stride_context_speed_max_offset() -> usize {
        SPEED_OFFSET + 2
    }
    #[inline]
    pub fn context_map_speed_offset() -> usize {
        SPEED_OFFSET + 4
    }
    #[inline]
    pub fn context_map_speed_max_offset() -> usize {
        SPEED_OFFSET + 6
    }
    #[inline]
    pub fn combined_stride_context_speed_offset() -> usize {
        SPEED_OFFSET + 8
    }
    #[inline]
    pub fn combined_stride_context_speed_max_offset() -> usize {
        SPEED_OFFSET + 10
    }
    #[inline]
    pub fn literal_prediction_mode(&self) -> LiteralPredictionModeNibble {
        let cm_slice = self.predmode_speed_and_distance_context_map.slice();
        if PREDMODE_OFFSET < cm_slice.len() {
            LiteralPredictionModeNibble(cm_slice[PREDMODE_OFFSET])
        } else {
            LiteralPredictionModeNibble::default()
        }
    }
    pub fn stride_context_speed(&self) -> [(u16, u16); 2] {
        let v = self.stride_context_speed_f8();
        [
            (self::u8_to_speed(v[0].0), self::u8_to_speed(v[0].1)),
            (self::u8_to_speed(v[1].0), self::u8_to_speed(v[1].1)),
        ]
    }
    pub fn context_map_speed(&self) -> [(u16, u16); 2] {
        let v = self.context_map_speed_f8();
        [
            (self::u8_to_speed(v[0].0), self::u8_to_speed(v[0].1)),
            (self::u8_to_speed(v[1].0), self::u8_to_speed(v[1].1)),
        ]
    }
    pub fn combined_stride_context_speed(&self) -> [(u16, u16); 2] {
        let v = self.combined_stride_context_speed_f8();
        [
            (self::u8_to_speed(v[0].0), self::u8_to_speed(v[0].1)),
            (self::u8_to_speed(v[1].0), self::u8_to_speed(v[1].1)),
        ]
    }
    #[inline]
    pub fn stride_context_speed_f8(&self) -> [(u8, u8); 2] {
        let cm_slice = self.predmode_speed_and_distance_context_map.slice();
        let low_speed = cm_slice[Self::stride_context_speed_offset()];
        let high_speed = cm_slice[Self::stride_context_speed_offset() + 1];
        let low_max = cm_slice[Self::stride_context_speed_max_offset()];
        let high_max = cm_slice[Self::stride_context_speed_max_offset() + 1];
        [(low_speed, low_max), (high_speed, high_max)]
    }
    #[inline]
    pub fn combined_stride_context_speed_f8(&self) -> [(u8, u8); 2] {
        let cm_slice = self.predmode_speed_and_distance_context_map.slice();
        let low_speed = cm_slice[Self::combined_stride_context_speed_offset()];
        let high_speed = cm_slice[Self::combined_stride_context_speed_offset() + 1];
        let low_max = cm_slice[Self::combined_stride_context_speed_max_offset()];
        let high_max = cm_slice[Self::combined_stride_context_speed_max_offset() + 1];
        [(low_speed, low_max), (high_speed, high_max)]
    }
    #[inline]
    pub fn context_map_speed_f8(&self) -> [(u8, u8); 2] {
        let cm_slice = self.predmode_speed_and_distance_context_map.slice();
        let low_speed = cm_slice[Self::context_map_speed_offset()];
        let high_speed = cm_slice[Self::context_map_speed_offset() + 1];
        let low_max = cm_slice[Self::context_map_speed_max_offset()];
        let high_max = cm_slice[Self::context_map_speed_max_offset() + 1];
        [(low_speed, low_max), (high_speed, high_max)]
    }
}

impl<SliceType: SliceWrapper<u8> + Clone> Clone for PredictionModeContextMap<SliceType> {
    #[inline(always)]
    fn clone(&self) -> Self {
        PredictionModeContextMap::<SliceType> {
            literal_context_map: self.literal_context_map.clone(),
            predmode_speed_and_distance_context_map: self
                .predmode_speed_and_distance_context_map
                .clone(),
        }
    }
}
impl<SliceType: SliceWrapper<u8> + Clone + Copy> Copy for PredictionModeContextMap<SliceType> {}

#[derive(Debug, Clone, Copy)]
pub struct CopyCommand {
    pub distance: u32,
    pub num_bytes: u32,
}

impl Nop<CopyCommand> for CopyCommand {
    #[inline(always)]
    fn nop() -> Self {
        CopyCommand {
            distance: 1,
            num_bytes: 0,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct DictCommand {
    pub word_size: u8,
    pub transform: u8,
    pub final_size: u8,
    pub empty: u8,
    pub word_id: u32,
}

impl Nop<DictCommand> for DictCommand {
    #[inline(always)]
    fn nop() -> Self {
        DictCommand {
            word_size: 0,
            transform: 0,
            final_size: 0,
            empty: 1,
            word_id: 0,
        }
    }
}

#[derive(Debug)]
#[cfg(not(feature = "external-literal-probability"))]
pub struct FeatureFlagSliceType<SliceType: SliceWrapper<u8>>(core::marker::PhantomData<SliceType>);

#[cfg(not(feature = "external-literal-probability"))]
impl<SliceType: SliceWrapper<u8>> SliceWrapper<u8> for FeatureFlagSliceType<SliceType> {
    fn slice(&self) -> &[u8] {
        &[]
    }
}

#[cfg(not(feature = "external-literal-probability"))]
impl<SliceType: SliceWrapper<u8> + Default> Default for FeatureFlagSliceType<SliceType> {
    fn default() -> Self {
        FeatureFlagSliceType::<SliceType>(core::marker::PhantomData::<SliceType>)
    }
}

#[derive(Debug)]
#[cfg(feature = "external-literal-probability")]
pub struct FeatureFlagSliceType<SliceType: SliceWrapper<u8>>(pub SliceType);

#[cfg(feature = "external-literal-probability")]
impl<SliceType: SliceWrapper<u8>> SliceWrapper<u8> for FeatureFlagSliceType<SliceType> {
    #[inline(always)]
    fn slice(&self) -> &[u8] {
        self.0.slice()
    }
}

#[cfg(feature = "external-literal-probability")]
impl<SliceType: SliceWrapper<u8> + Default> Default for FeatureFlagSliceType<SliceType> {
    #[inline(always)]
    fn default() -> Self {
        FeatureFlagSliceType::<SliceType>(SliceType::default())
    }
}

impl<SliceType: SliceWrapper<u8> + Clone> Clone for FeatureFlagSliceType<SliceType> {
    #[inline(always)]
    fn clone(&self) -> Self {
        FeatureFlagSliceType::<SliceType>(self.0)
    }
}
impl<SliceType: SliceWrapper<u8> + Clone + Copy> Copy for FeatureFlagSliceType<SliceType> {}

#[derive(Debug)]
pub struct LiteralCommand<SliceType: SliceWrapper<u8>> {
    pub data: SliceType,
    pub prob: FeatureFlagSliceType<SliceType>,
    pub high_entropy: bool, // this block of bytes is high entropy with a few patterns never seen again; adapt slower
}
impl<SliceType: SliceWrapper<u8>> SliceWrapper<u8> for LiteralCommand<SliceType> {
    #[inline(always)]
    fn slice(&self) -> &[u8] {
        self.data.slice()
    }
}
impl<SliceType: SliceWrapper<u8> + SliceWrapperMut<u8>> SliceWrapperMut<u8>
    for LiteralCommand<SliceType>
{
    #[inline(always)]
    fn slice_mut(&mut self) -> &mut [u8] {
        self.data.slice_mut()
    }
}

impl<SliceType: SliceWrapper<u8> + Default> Nop<LiteralCommand<SliceType>>
    for LiteralCommand<SliceType>
{
    #[inline(always)]
    fn nop() -> Self {
        LiteralCommand {
            data: SliceType::default(),
            prob: FeatureFlagSliceType::<SliceType>::default(),
            high_entropy: false,
        }
    }
}
impl<SliceType: SliceWrapper<u8> + Clone> Clone for LiteralCommand<SliceType> {
    #[inline(always)]
    fn clone(&self) -> LiteralCommand<SliceType> {
        LiteralCommand::<SliceType> {
            data: self.data.clone(),
            prob: self.prob.clone(),
            high_entropy: self.high_entropy,
        }
    }
}
impl<SliceType: SliceWrapper<u8> + Clone + Copy> Copy for LiteralCommand<SliceType> {}

#[derive(Debug)]
pub enum Command<SliceType: SliceWrapper<u8>> {
    Copy(CopyCommand),
    Dict(DictCommand),
    Literal(LiteralCommand<SliceType>),
    BlockSwitchCommand(BlockSwitch),
    BlockSwitchLiteral(LiteralBlockSwitch),
    BlockSwitchDistance(BlockSwitch),
    PredictionMode(PredictionModeContextMap<SliceType>),
}
impl<SliceType: SliceWrapper<u8> + Default> Command<SliceType> {
    #[inline]
    pub fn free_array<F>(&mut self, apply_func: &mut F)
    where
        F: FnMut(SliceType),
    {
        match self {
            Command::Literal(ref mut lit) => apply_func(core::mem::take(&mut lit.data)),
            Command::PredictionMode(ref mut pm) => {
                apply_func(core::mem::take(&mut pm.literal_context_map));
                apply_func(core::mem::take(
                    &mut pm.predmode_speed_and_distance_context_map,
                ));
            }
            _ => {}
        }
    }
}

impl<SliceType: SliceWrapper<u8>> Default for Command<SliceType> {
    #[inline(always)]
    fn default() -> Self {
        Command::<SliceType>::nop()
    }
}

impl<SliceType: SliceWrapper<u8>> Nop<Command<SliceType>> for Command<SliceType> {
    #[inline(always)]
    fn nop() -> Command<SliceType> {
        Command::Copy(CopyCommand::nop())
    }
}

impl<SliceType: SliceWrapper<u8> + Clone> Clone for Command<SliceType> {
    #[inline]
    fn clone(&self) -> Command<SliceType> {
        match self {
            Command::Copy(copy) => Command::Copy(*copy),
            Command::Dict(dict) => Command::Dict(*dict),
            Command::Literal(literal) => Command::Literal(literal.clone()),
            Command::BlockSwitchCommand(switch) => Command::BlockSwitchCommand(*switch),
            Command::BlockSwitchLiteral(switch) => Command::BlockSwitchLiteral(*switch),
            Command::BlockSwitchDistance(switch) => Command::BlockSwitchDistance(*switch),
            Command::PredictionMode(pm) => Command::PredictionMode(pm.clone()),
        }
    }
}

impl<SliceType: SliceWrapper<u8> + Clone + Copy> Copy for Command<SliceType> {}

#[inline(always)]
pub fn free_cmd_inline<SliceTypeAllocator: Allocator<u8>>(
    xself: &mut Command<SliceTypeAllocator::AllocatedMemory>,
    m8: &mut SliceTypeAllocator,
) {
    match *xself {
        Command::Literal(ref mut lit) => m8.free_cell(core::mem::take(&mut lit.data)),
        Command::PredictionMode(ref mut pm) => {
            m8.free_cell(core::mem::take(&mut pm.literal_context_map));
            m8.free_cell(core::mem::take(
                &mut pm.predmode_speed_and_distance_context_map,
            ));
        }
        Command::Dict(_)
        | Command::Copy(_)
        | Command::BlockSwitchCommand(_)
        | Command::BlockSwitchLiteral(_)
        | Command::BlockSwitchDistance(_) => {}
    }
}

#[inline(never)]
pub fn free_cmd<SliceTypeAllocator: Allocator<u8>>(
    xself: &mut Command<SliceTypeAllocator::AllocatedMemory>,
    m8: &mut SliceTypeAllocator,
) {
    free_cmd_inline(xself, m8)
}

#[derive(Clone, Copy, Default, Debug)]
pub struct SliceOffset(pub usize, pub u32);
impl SliceWrapper<u8> for SliceOffset {
    fn slice(&self) -> &[u8] {
        // not perfect--shouldn't be calling this without thawing the wrapper
        &[]
    }
}

pub trait Freezable {
    fn freeze(&self) -> SliceOffset;
}

pub trait Unfreezable {
    fn thaw<'a>(&self, data: &'a [u8]) -> InputReference<'a>;
    fn thaw_mut<'a>(&self, data: &'a mut [u8]) -> InputReferenceMut<'a>;
    fn thaw_pair<'a>(&self, pair: &InputPair<'a>) -> Result<InputReference<'a>, ()>;
}

impl<'a> From<InputReference<'a>> for SliceOffset {
    fn from(f: InputReference<'a>) -> Self {
        debug_assert!(f.data.len() <= 0xffff_ffff);
        SliceOffset(f.orig_offset, f.data.len() as u32)
    }
}
impl Unfreezable for SliceOffset {
    fn thaw<'a>(&self, data: &'a [u8]) -> InputReference<'a> {
        InputReference {
            data: data.split_at(self.0).1.split_at(self.1 as usize).0,
            orig_offset: self.0,
        }
    }
    fn thaw_mut<'a>(&self, data: &'a mut [u8]) -> InputReferenceMut<'a> {
        InputReferenceMut {
            data: data.split_at_mut(self.0).1.split_at_mut(self.1 as usize).0,
            orig_offset: self.0,
        }
    }
    fn thaw_pair<'a>(&self, pair: &InputPair<'a>) -> Result<InputReference<'a>, ()> {
        if self.0 >= pair.1.orig_offset {
            return Ok(InputReference {
                data: pair
                    .1
                    .data
                    .split_at(self.0 - pair.1.orig_offset)
                    .1
                    .split_at(self.1 as usize)
                    .0,
                orig_offset: self.0,
            });
        }
        let offset = self.0 - pair.0.orig_offset;
        if offset + self.1 as usize <= pair.0.data.len() {
            // overlap
            Ok(InputReference {
                data: pair.0.data.split_at(offset).1.split_at(self.1 as usize).0,
                orig_offset: self.0,
            })
        } else {
            Err(())
        }
    }
}
impl SliceOffset {
    pub fn offset(&self) -> usize {
        self.0
    }
    pub fn len(&self) -> usize {
        self.1 as usize
    }
    pub fn len32(&self) -> u32 {
        self.1
    }
}

pub type StaticCommand = Command<SliceOffset>;

pub trait CommandProcessor<'a> {
    fn push(&mut self, val: Command<InputReference<'a>>);
    fn push_literals(&mut self, data: &InputPair<'a>) {
        if data.0.len() != 0 {
            self.push(Command::Literal(LiteralCommand {
                data: data.0,
                prob: FeatureFlagSliceType::<InputReference>::default(),
                high_entropy: false,
            }));
        }
        if data.1.len() != 0 {
            self.push(Command::Literal(LiteralCommand {
                data: data.1,
                prob: FeatureFlagSliceType::<InputReference>::default(),
                high_entropy: false,
            }));
        }
    }
    fn push_rand_literals(&mut self, data: &InputPair<'a>) {
        if data.0.len() != 0 {
            self.push(Command::Literal(LiteralCommand {
                data: data.0,
                prob: FeatureFlagSliceType::<InputReference>::default(),
                high_entropy: true,
            }));
        }
        if data.1.len() != 0 {
            self.push(Command::Literal(LiteralCommand {
                data: data.1,
                prob: FeatureFlagSliceType::<InputReference>::default(),
                high_entropy: true,
            }));
        }
    }
    fn push_block_switch_literal(&mut self, block_type: u8) {
        self.push(Command::BlockSwitchLiteral(LiteralBlockSwitch::new(
            block_type, 0,
        )))
    }
}

impl<SliceType: Unfreezable + SliceWrapper<u8>> Command<SliceType> {
    pub fn thaw_pair<'a>(&self, data: &InputPair<'a>) -> Command<InputReference<'a>> {
        match self {
            Command::Literal(ref lit) => Command::Literal(LiteralCommand {
                data: lit.data.thaw_pair(data).unwrap(),
                prob: FeatureFlagSliceType::default(),
                high_entropy: lit.high_entropy,
            }),
            Command::PredictionMode(ref pm) => Command::PredictionMode(PredictionModeContextMap {
                literal_context_map: pm.literal_context_map.thaw_pair(data).unwrap(),
                predmode_speed_and_distance_context_map: pm
                    .predmode_speed_and_distance_context_map
                    .thaw_pair(data)
                    .unwrap(),
            }),
            Command::Dict(ref d) => Command::Dict(*d),
            Command::Copy(ref c) => Command::Copy(*c),
            Command::BlockSwitchCommand(ref c) => Command::BlockSwitchCommand(*c),
            Command::BlockSwitchLiteral(ref c) => Command::BlockSwitchLiteral(*c),
            Command::BlockSwitchDistance(ref c) => Command::BlockSwitchDistance(*c),
        }
    }

    pub fn thaw<'a>(&self, data: &'a [u8]) -> Command<InputReference<'a>> {
        match self {
            Command::Literal(ref lit) => Command::Literal(LiteralCommand {
                data: lit.data.thaw(data),
                prob: FeatureFlagSliceType::default(),
                high_entropy: lit.high_entropy,
            }),
            Command::PredictionMode(ref pm) => Command::PredictionMode(PredictionModeContextMap {
                literal_context_map: pm.literal_context_map.thaw(data),
                predmode_speed_and_distance_context_map: pm
                    .predmode_speed_and_distance_context_map
                    .thaw(data),
            }),
            Command::Dict(ref d) => Command::Dict(*d),
            Command::Copy(ref c) => Command::Copy(*c),
            Command::BlockSwitchCommand(ref c) => Command::BlockSwitchCommand(*c),
            Command::BlockSwitchLiteral(ref c) => Command::BlockSwitchLiteral(*c),
            Command::BlockSwitchDistance(ref c) => Command::BlockSwitchDistance(*c),
        }
    }
}

impl<SliceType: SliceWrapper<u8> + Freezable> Command<SliceType> {
    pub fn freeze(&self) -> Command<SliceOffset> {
        match self {
            Command::Literal(ref lit) => Command::Literal(LiteralCommand {
                data: lit.data.freeze(),
                prob: FeatureFlagSliceType::default(),
                high_entropy: lit.high_entropy,
            }),
            Command::PredictionMode(ref pm) => Command::PredictionMode(PredictionModeContextMap {
                literal_context_map: pm.literal_context_map.freeze(),
                predmode_speed_and_distance_context_map: pm
                    .predmode_speed_and_distance_context_map
                    .freeze(),
            }),
            Command::Dict(ref d) => Command::Dict(*d),
            Command::Copy(ref c) => Command::Copy(*c),
            Command::BlockSwitchCommand(ref c) => Command::BlockSwitchCommand(*c),
            Command::BlockSwitchLiteral(ref c) => Command::BlockSwitchLiteral(*c),
            Command::BlockSwitchDistance(ref c) => Command::BlockSwitchDistance(*c),
        }
    }
}

#[inline(always)]
pub fn speed_to_u8(data: u16) -> u8 {
    let length = 16 - data.leading_zeros() as u8;
    let mantissa = if data != 0 {
        let rem = data - (1 << (length - 1));
        (rem << 3) >> (length - 1)
    } else {
        0
    };
    (length << 3) | mantissa as u8
}

#[inline(always)]
pub fn u8_to_speed(data: u8) -> u16 {
    if data < 8 {
        0
    } else {
        let log_val = (data >> 3) - 1;
        let rem = (u16::from(data) & 0x07) << log_val;
        (1u16 << log_val) | (rem >> 3)
    }
}

#[cfg(test)]
mod test {
    use super::{speed_to_u8, u8_to_speed};
    fn tst_u8_to_speed(data: u16) {
        assert_eq!(u8_to_speed(speed_to_u8(data)), data);
    }
    #[test]
    fn test_u8_to_speed() {
        tst_u8_to_speed(0);
        tst_u8_to_speed(1);
        tst_u8_to_speed(2);
        tst_u8_to_speed(3);
        tst_u8_to_speed(4);
        tst_u8_to_speed(5);
        tst_u8_to_speed(6);
        tst_u8_to_speed(7);
        tst_u8_to_speed(8);
        tst_u8_to_speed(10);
        tst_u8_to_speed(12);
        tst_u8_to_speed(16);
        tst_u8_to_speed(24);
        tst_u8_to_speed(32);
        tst_u8_to_speed(48);
        tst_u8_to_speed(64);
        tst_u8_to_speed(96);
        tst_u8_to_speed(768);
        tst_u8_to_speed(1280);
        tst_u8_to_speed(1536);
        tst_u8_to_speed(1664);
    }
}
