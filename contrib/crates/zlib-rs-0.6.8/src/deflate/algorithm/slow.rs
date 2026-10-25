#![forbid(unsafe_code)]

use super::flush_block;
use crate::{
    deflate::{
        fill_window, flush_block_only, BlockState, DeflateStream, Strategy, MIN_LOOKAHEAD,
        STD_MIN_MATCH, WANT_MIN_MATCH,
    },
    DeflateFlush,
};

pub fn deflate_slow(stream: &mut DeflateStream, flush: DeflateFlush) -> BlockState {
    let mut hash_head; /* head of hash chain */
    let mut bflush; /* set if current block must be flushed */
    let mut dist;
    let mut match_len;

    let use_longest_match_slow = stream.state.max_chain_length > 1024;
    let valid_distance_range = 1..=stream.state.max_dist() as isize;

    let mut match_available = stream.state.match_available;

    /* Process the input block. */
    loop {
        /* Make sure that we always have enough lookahead, except
         * at the end of the input file. We need STD_MAX_MATCH bytes
         * for the next match, plus WANT_MIN_MATCH bytes to insert the
         * string following the next match.
         */
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

        /* Insert the string window[strstart .. strstart+2] in the
         * dictionary, and set hash_head to the head of the hash chain:
         */
        hash_head = if state.lookahead >= WANT_MIN_MATCH {
            state.quick_insert_string(state.strstart)
        } else {
            0
        };

        // Find the longest match, discarding those <= prev_length.
        state.prev_match = state.match_start;
        match_len = STD_MIN_MATCH - 1;
        dist = state.strstart as isize - hash_head as isize;

        if valid_distance_range.contains(&dist)
            && state.prev_length < state.max_lazy_match
            && hash_head != 0
        {
            // To simplify the code, we prevent matches with the string
            // of window index 0 (in particular we have to avoid a match
            // of the string with itself at the start of the input file).
            (match_len, state.match_start) = if use_longest_match_slow {
                crate::deflate::longest_match::longest_match_slow(state, hash_head)
            } else {
                crate::deflate::longest_match::longest_match(state, hash_head)
            };

            if match_len <= 5 && (state.strategy == Strategy::Filtered) {
                /* If prev_match is also WANT_MIN_MATCH, match_start is garbage
                 * but we will ignore the current match anyway.
                 */
                match_len = STD_MIN_MATCH - 1;
            }
        }

        // If there was a match at the previous step and the current
        // match is not better, output the previous match:
        if state.prev_length as usize >= STD_MIN_MATCH && match_len <= state.prev_length as usize {
            let max_insert = state.strstart + state.lookahead - STD_MIN_MATCH;
            /* Do not insert strings in hash table beyond this. */

            // check_match(s, state.strstart-1, state.prev_match, state.prev_length);

            bflush = state.tally_dist(
                state.strstart - 1 - state.prev_match as usize,
                state.prev_length as usize - STD_MIN_MATCH,
            );

            /* Insert in hash table all strings up to the end of the match.
             * strstart-1 and strstart are already inserted. If there is not
             * enough lookahead, the last two strings are not inserted in
             * the hash table.
             */
            state.prev_length -= 1;
            state.lookahead -= state.prev_length as usize;

            let mov_fwd = state.prev_length as usize - 1;
            if max_insert > state.strstart {
                let insert_cnt = Ord::min(mov_fwd, max_insert - state.strstart);
                state.insert_string(state.strstart + 1, insert_cnt);
            }
            state.prev_length = 0;
            state.match_available = false;
            match_available = false;
            state.strstart += mov_fwd + 1;

            if bflush {
                flush_block!(stream, false);
            }
        } else if match_available {
            // If there was no match at the previous position, output a
            // single literal. If there was a match but the current match
            // is longer, truncate the previous match to a single literal.
            let lc = state.window.filled()[state.strstart - 1];
            bflush = state.tally_lit(lc);
            if bflush {
                flush_block_only(stream, false);
            }

            stream.state.prev_length = match_len as u16;
            stream.state.strstart += 1;
            stream.state.lookahead -= 1;
            if stream.avail_out == 0 {
                return BlockState::NeedMore;
            }
        } else {
            // There is no previous match to compare with, wait for
            // the next step to decide.
            state.prev_length = match_len as u16;
            state.match_available = true;
            match_available = true;
            state.strstart += 1;
            state.lookahead -= 1;
        }
    }

    assert_ne!(flush, DeflateFlush::NoFlush, "no flush?");

    let state = &mut stream.state;

    if state.match_available {
        let lc = state.window.filled()[state.strstart - 1];
        let _ = state.tally_lit(lc);
        state.match_available = false;
    }

    state.insert = Ord::min(state.strstart, STD_MIN_MATCH - 1);

    if flush == DeflateFlush::Finish {
        flush_block!(stream, true);
        return BlockState::FinishDone;
    }

    if !stream.state.sym_buf.is_empty() {
        flush_block!(stream, false);
    }

    BlockState::BlockDone
}
