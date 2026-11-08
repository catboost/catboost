#![forbid(unsafe_code)]

use super::flush_block;
use crate::deflate::hash_calc::StandardHashCalc;
use crate::{
    deflate::{
        fill_window, BlockState, DeflateStream, MIN_LOOKAHEAD, STD_MIN_MATCH, WANT_MIN_MATCH,
    },
    DeflateFlush,
};

pub fn deflate_fast(stream: &mut DeflateStream, flush: DeflateFlush) -> BlockState {
    loop {
        // Make sure that we always have enough lookahead, except
        // at the end of the input file. We need STD_MAX_MATCH bytes
        // for the next match, plus WANT_MIN_MATCH bytes to insert the
        // string following the next match.
        if stream.state.lookahead < MIN_LOOKAHEAD {
            fill_window(stream);
            if stream.state.lookahead < MIN_LOOKAHEAD && flush == DeflateFlush::NoFlush {
                return BlockState::NeedMore;
            }
            if stream.state.lookahead == 0 {
                break; /* flush the current block */
            }
        }

        let state = &mut stream.state;

        // Insert the string window[strstart .. strstart+2] in the
        // dictionary, and set hash_head to the head of the hash chain:

        // Literal character to output if there is no match.
        let lc = if state.lookahead >= WANT_MIN_MATCH {
            let val = u32::from_le_bytes(
                state.window.filled()[state.strstart..state.strstart + 4]
                    .try_into()
                    .unwrap(),
            );
            let hash_head = StandardHashCalc::quick_insert_value(state, state.strstart, val);
            let dist = state.strstart as isize - hash_head as isize;

            // Find the longest match for the string starting at offset state.strstart.
            if dist <= state.max_dist() as isize && dist > 0 && hash_head != 0 {
                // To simplify the code, we prevent matches with the string
                // of window index 0 (in particular we have to avoid a match
                // of the string with itself at the start of the input file).
                let mut match_len;
                (match_len, state.match_start) =
                    crate::deflate::longest_match::longest_match(state, hash_head);
                if match_len >= WANT_MIN_MATCH {
                    let bflush = state.tally_dist(
                        state.strstart - state.match_start as usize,
                        match_len - STD_MIN_MATCH,
                    );

                    state.lookahead -= match_len;

                    /* Insert new strings in the hash table only if the match length
                     * is not too large. This saves time but degrades compression.
                     */
                    if match_len <= state.max_insert_length() && state.lookahead >= WANT_MIN_MATCH {
                        match_len -= 1; /* string at strstart already in table */
                        state.strstart += 1;

                        state.insert_string(state.strstart, match_len);
                        state.strstart += match_len;
                    } else {
                        state.strstart += match_len;
                        StandardHashCalc::quick_insert_string(
                            state,
                            state.strstart + 2 - STD_MIN_MATCH,
                        );

                        /* If lookahead < STD_MIN_MATCH, ins_h is garbage, but it does not
                         * matter since it will be recomputed at next deflate call.
                         */
                    }
                    if bflush {
                        flush_block!(stream, false);
                    }
                    continue;
                }
            }
            val as u8
        } else {
            state.window.filled()[state.strstart]
        };
        /* No match, output a literal byte */
        let bflush = state.tally_lit(lc);
        state.lookahead -= 1;
        state.strstart += 1;
        if bflush {
            flush_block!(stream, false);
        }
    }

    stream.state.insert = if stream.state.strstart < (STD_MIN_MATCH - 1) {
        stream.state.strstart
    } else {
        STD_MIN_MATCH - 1
    };

    if flush == DeflateFlush::Finish {
        flush_block!(stream, true);
        return BlockState::FinishDone;
    }

    if !stream.state.sym_buf.is_empty() {
        flush_block!(stream, false);
    }

    BlockState::BlockDone
}
