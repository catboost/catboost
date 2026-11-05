#![forbid(unsafe_code)]

use crate::deflate::{State, HASH_SIZE, STD_MIN_MATCH};

#[derive(Debug, Clone, Copy)]
pub enum HashCalcVariant {
    Standard,
    Roll,
}

impl HashCalcVariant {
    /// Use rolling hash for deflate_slow algorithm with level 9. It allows us to
    /// properly lookup different hash chains to speed up longest_match search.
    pub fn for_max_chain_length(max_chain_length: u16) -> Self {
        if max_chain_length > 1024 {
            HashCalcVariant::Roll
        } else {
            HashCalcVariant::Standard
        }
    }
}

pub struct StandardHashCalc;

impl StandardHashCalc {
    const HASH_CALC_OFFSET: usize = 0;

    const HASH_CALC_MASK: u32 = (HASH_SIZE - 1) as u32;

    fn hash_calc(_: u32, val: u32) -> u32 {
        const HASH_SLIDE: u32 = 16;
        val.wrapping_mul(2654435761) >> HASH_SLIDE
    }

    pub fn update_hash(h: u32, val: u32) -> u32 {
        Self::hash_calc(h, val) & Self::HASH_CALC_MASK
    }

    #[inline]
    pub fn quick_insert_string(state: &mut State, string: usize) -> u16 {
        let slice = &state.window.filled()[string + Self::HASH_CALC_OFFSET..];
        let val = u32::from_le_bytes(slice[..4].try_into().unwrap());

        Self::quick_insert_value(state, string, val)
    }

    #[inline]
    pub fn quick_insert_value(state: &mut State, string: usize, val: u32) -> u16 {
        let hm = Self::update_hash(0, val) as usize;

        let head = state.head.as_slice()[hm];
        if head != string as u16 {
            let w_mask = state.w_mask();
            state.prev.as_mut_slice()[string & w_mask] = head;
            state.head.as_mut_slice()[hm] = string as u16;
        }

        head
    }

    pub fn insert_string(state: &mut State, string: usize, count: usize) {
        let slice = &state.window.filled()[string + Self::HASH_CALC_OFFSET..];

        // it can happen that insufficient bytes are initialized
        // .take(count) generates worse assembly
        let slice = &slice[..Ord::min(slice.len(), count + 3)];

        let w_mask = state.w_mask();
        for (i, w) in slice.windows(4).enumerate() {
            let idx = string as u16 + i as u16;

            let val = u32::from_le_bytes(w.try_into().unwrap());

            let hm = Self::update_hash(0, val) as usize;

            let head = state.head.as_slice()[hm];
            if head != idx {
                state.prev.as_mut_slice()[idx as usize & w_mask] = head;
                state.head.as_mut_slice()[hm] = idx;
            }
        }
    }
}

pub struct RollHashCalc;

impl RollHashCalc {
    const HASH_CALC_OFFSET: usize = STD_MIN_MATCH - 1;

    const HASH_CALC_MASK: u32 = (1 << 15) - 1;

    fn hash_calc(h: u32, val: u32) -> u32 {
        const HASH_SLIDE: u32 = 5;
        (h << HASH_SLIDE) ^ val
    }

    pub fn update_hash(h: u32, val: u32) -> u32 {
        Self::hash_calc(h, val) & Self::HASH_CALC_MASK
    }

    pub fn quick_insert_string(state: &mut State, string: usize) -> u16 {
        let val = state.window.filled()[string + Self::HASH_CALC_OFFSET] as u32;

        state.ins_h = Self::hash_calc(state.ins_h, val);
        state.ins_h &= Self::HASH_CALC_MASK;

        let hm = state.ins_h as usize;

        let head = state.head.as_slice()[hm];
        if head != string as u16 {
            let w_mask = state.w_mask();
            state.prev.as_mut_slice()[string & w_mask] = head;
            state.head.as_mut_slice()[hm] = string as u16;
        }

        head
    }

    pub fn insert_string(state: &mut State, string: usize, count: usize) {
        let slice = &state.window.filled()[string + Self::HASH_CALC_OFFSET..][..count];

        let w_mask = state.w_mask();
        for (i, val) in slice.iter().copied().enumerate() {
            let idx = string as u16 + i as u16;

            state.ins_h = Self::hash_calc(state.ins_h, val as u32);
            state.ins_h &= Self::HASH_CALC_MASK;
            let hm = state.ins_h as usize;

            let head = state.head.as_slice()[hm];
            if head != idx {
                state.prev.as_mut_slice()[idx as usize & w_mask] = head;
                state.head.as_mut_slice()[hm] = idx;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roll_hash_calc() {
        assert_eq!(RollHashCalc::hash_calc(2565, 93), 82173);
        assert_eq!(RollHashCalc::hash_calc(16637, 10), 532394);
        assert_eq!(RollHashCalc::hash_calc(8106, 100), 259364);
        assert_eq!(RollHashCalc::hash_calc(29988, 101), 959717);
        assert_eq!(RollHashCalc::hash_calc(9445, 98), 302274);
        assert_eq!(RollHashCalc::hash_calc(7362, 117), 235573);
        assert_eq!(RollHashCalc::hash_calc(6197, 103), 198343);
        assert_eq!(RollHashCalc::hash_calc(1735, 32), 55488);
        assert_eq!(RollHashCalc::hash_calc(22720, 61), 727101);
        assert_eq!(RollHashCalc::hash_calc(6205, 32), 198528);
        assert_eq!(RollHashCalc::hash_calc(3826, 117), 122421);
        assert_eq!(RollHashCalc::hash_calc(24117, 101), 771781);
    }

    #[test]
    fn standard_hash_calc() {
        assert_eq!(StandardHashCalc::hash_calc(0, 807411760), 65468);
        assert_eq!(StandardHashCalc::hash_calc(0, 540024864), 42837);
        assert_eq!(StandardHashCalc::hash_calc(0, 538980384), 33760);
        assert_eq!(StandardHashCalc::hash_calc(0, 775430176), 8925);
        assert_eq!(StandardHashCalc::hash_calc(0, 941629472), 42053);
    }
}
