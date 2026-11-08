#![forbid(unsafe_code)]

use crate::deflate::hash_calc::StandardHashCalc;
use crate::{
    deflate::{
        fill_window, flush_pending, BlockState, BlockType, DeflateStream, State, StaticTreeDesc,
        MIN_LOOKAHEAD, STD_MAX_MATCH, STD_MIN_MATCH, WANT_MIN_MATCH,
    },
    DeflateFlush,
};

pub fn deflate_quick(stream: &mut DeflateStream, flush: DeflateFlush) -> BlockState {
    let mut state = &mut stream.state;

    macro_rules! quick_end_block {
        ($last:expr) => {
            if state.block_open > 0 {
                state
                    .bit_writer
                    .emit_end_block_and_align(&StaticTreeDesc::L.static_tree, $last);
                state.block_open = 0;
                state.block_start = state.strstart as isize;
                flush_pending(stream);
                #[allow(unused_assignments)]
                {
                    state = &mut stream.state;
                }
                if stream.avail_out == 0 {
                    return match $last {
                        true => BlockState::FinishStarted,
                        false => BlockState::NeedMore,
                    };
                }
            }
        };
    }

    macro_rules! quick_start_block {
        ($last:expr) => {
            state.bit_writer.emit_tree(BlockType::StaticTrees, $last);
            state.block_open = 1 + $last as u8;
            state.block_start = state.strstart as isize;
        };
    }

    let last = matches!(flush, DeflateFlush::Finish);

    if last && state.block_open != 2 {
        /* Emit end of previous block */
        quick_end_block!(false);
        /* Emit start of last block */
        quick_start_block!(last);
    } else if state.block_open == 0 && state.lookahead > 0 {
        /* Start new block only when we have lookahead data, so that if no
        input data is given an empty block will not be written */
        quick_start_block!(last);
    }

    loop {
        if state.bit_writer.pending.pending + State::BIT_BUF_SIZE.div_ceil(8) as usize
            >= state.pending_buf_size()
        {
            flush_pending(stream);
            state = &mut stream.state;
            if stream.avail_out == 0 {
                return if last
                    && stream.avail_in == 0
                    && state.bit_writer.bits_valid == 0
                    && state.block_open == 0
                {
                    BlockState::FinishStarted
                } else {
                    BlockState::NeedMore
                };
            }
        }

        if state.lookahead < MIN_LOOKAHEAD {
            fill_window(stream);
            state = &mut stream.state;

            if state.lookahead < MIN_LOOKAHEAD && matches!(flush, DeflateFlush::NoFlush) {
                return BlockState::NeedMore;
            }
            if state.lookahead == 0 {
                break;
            }

            if state.block_open == 0 {
                // Start new block when we have lookahead data,
                // so that if no input data is given an empty block will not be written
                quick_start_block!(last);
            }
        }

        let lc = if state.lookahead >= WANT_MIN_MATCH {
            macro_rules! first_four_bytes {
                ($slice:expr, $offset:expr) => {
                    u32::from_le_bytes($slice[$offset..$offset + 4].try_into().unwrap())
                };
            }

            let str_val = {
                let str_start = &state.window.filled()[state.strstart..];
                first_four_bytes!(str_start, 0)
            };
            let hash_head = StandardHashCalc::quick_insert_value(state, state.strstart, str_val);
            let dist = state.strstart as isize - hash_head as isize;

            if dist <= state.max_dist() as isize && dist > 0 {
                let match_start = &state.window.filled()[hash_head as usize..];

                if str_val == first_four_bytes!(match_start, 0) {
                    let mut match_len = crate::deflate::compare256::compare256_slice(
                        &state.window.filled()[state.strstart + 2..],
                        &match_start[2..],
                    ) + 2;

                    if match_len >= WANT_MIN_MATCH {
                        match_len = Ord::min(match_len, state.lookahead);
                        match_len = Ord::min(match_len, STD_MAX_MATCH);

                        // TODO do this with a debug_assert?
                        // check_match(s, state.strstart, hash_head, match_len);

                        // The `dist` value is a distance within the window,
                        // and MAX_WBITS == 15 (32k), hence a u16 can always represent this value.
                        let dist = u16::try_from(dist).unwrap();

                        state
                            .bit_writer
                            .emit_dist_static((match_len - STD_MIN_MATCH) as u8, dist);
                        state.lookahead -= match_len;
                        state.strstart += match_len;
                        continue;
                    }
                }
            }
            str_val as u8
        } else {
            state.window.filled()[state.strstart]
        };
        state.bit_writer.emit_lit(StaticTreeDesc::L.static_tree, lc);
        state.strstart += 1;
        state.lookahead -= 1;
    }

    state.insert = Ord::min(state.strstart, STD_MIN_MATCH - 1);

    quick_end_block!(last);

    if last {
        BlockState::FinishDone
    } else {
        BlockState::BlockDone
    }
}
