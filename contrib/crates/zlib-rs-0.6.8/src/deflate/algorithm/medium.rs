#![forbid(unsafe_code)]

use super::flush_block;
use crate::deflate::hash_calc::StandardHashCalc;
use crate::{
    deflate::{
        fill_window, BlockState, DeflateStream, State, MIN_LOOKAHEAD, STD_MIN_MATCH, WANT_MIN_MATCH,
    },
    DeflateFlush,
};

pub fn deflate_medium(stream: &mut DeflateStream, flush: DeflateFlush) -> BlockState {
    let mut state = &mut stream.state;

    // For levels below 5, don't check the next position for a better match
    let early_exit = state.level < 5;

    let mut current_match = Match {
        match_start: 0,
        match_length: 0,
        strstart: 0,
        orgstart: 0,
    };
    let mut next_match = Match {
        match_start: 0,
        match_length: 0,
        strstart: 0,
        orgstart: 0,
    };

    loop {
        let mut hash_head;

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

            next_match.match_length = 0;
        }

        state = &mut stream.state;

        // Insert the string window[strstart .. strstart+2] in the
        // dictionary, and set hash_head to the head of the hash chain:

        /* If we already have a future match from a previous round, just use that */
        if !early_exit && next_match.match_length > 0 {
            current_match = next_match;
            next_match.match_length = 0;
        } else {
            hash_head = 0;
            if state.lookahead >= WANT_MIN_MATCH {
                hash_head = StandardHashCalc::quick_insert_string(state, state.strstart);
            }

            current_match.strstart = state.strstart as u16;
            current_match.orgstart = current_match.strstart;

            /* Find the longest match, discarding those <= prev_length.
             * At this point we have always match_length < WANT_MIN_MATCH
             */

            let dist = state.strstart as i64 - hash_head as i64;
            if dist <= state.max_dist() as i64 && dist > 0 && hash_head != 0 {
                /* To simplify the code, we prevent matches with the string
                 * of window index 0 (in particular we have to avoid a match
                 * of the string with itself at the start of the input file).
                 */
                let (match_length, match_start) =
                    crate::deflate::longest_match::longest_match(state, hash_head);
                state.match_start = match_start;
                current_match.match_length = match_length as u16;
                current_match.match_start = match_start;
                if (current_match.match_length as usize) < WANT_MIN_MATCH {
                    current_match.match_length = 1;
                }
                if current_match.match_start >= current_match.strstart {
                    /* this can happen due to some restarts */
                    current_match.match_length = 1;
                }
            } else {
                /* Set up the match to be a 1 byte literal */
                current_match.match_start = 0;
                current_match.match_length = 1;
            }
        }

        insert_match(state, current_match);

        /* now, look ahead one */
        if !early_exit
            && state.lookahead > MIN_LOOKAHEAD
            && ((current_match.strstart + current_match.match_length) as usize)
                < (state.window_size - MIN_LOOKAHEAD)
        {
            state.strstart = (current_match.strstart + current_match.match_length) as usize;
            hash_head = StandardHashCalc::quick_insert_string(state, state.strstart);

            next_match.strstart = state.strstart as u16;
            next_match.orgstart = next_match.strstart;

            /* Find the longest match, discarding those <= prev_length.
             * At this point we have always match_length < WANT_MIN_MATCH
             */

            let dist = state.strstart as i64 - hash_head as i64;
            if dist <= state.max_dist() as i64 && dist > 0 && hash_head != 0 {
                /* To simplify the code, we prevent matches with the string
                 * of window index 0 (in particular we have to avoid a match
                 * of the string with itself at the start of the input file).
                 */
                let (match_length, match_start) =
                    crate::deflate::longest_match::longest_match(state, hash_head);
                state.match_start = match_start;
                next_match.match_length = match_length as u16;
                next_match.match_start = match_start;

                if next_match.match_start >= next_match.strstart {
                    /* this can happen due to some restarts */
                    next_match.match_length = 1;
                }
                if (next_match.match_length as usize) < WANT_MIN_MATCH {
                    next_match.match_length = 1;
                } else {
                    fizzle_matches(
                        state.window.filled(),
                        state.max_dist(),
                        &mut current_match,
                        &mut next_match,
                    );
                }
            } else {
                /* Set up the match to be a 1 byte literal */
                next_match.match_start = 0;
                next_match.match_length = 1;
            }

            state.strstart = current_match.strstart as usize;
        } else {
            next_match.match_length = 0;
        }

        /* now emit the current match */
        let bflush = emit_match(state, current_match);

        /* move the "cursor" forward */
        state.strstart += current_match.match_length as usize;

        if bflush {
            flush_block!(stream, false);
        }
    }

    stream.state.insert = Ord::min(stream.state.strstart, STD_MIN_MATCH - 1);

    if flush == DeflateFlush::Finish {
        flush_block!(stream, true);
        return BlockState::FinishDone;
    }

    if !stream.state.sym_buf.is_empty() {
        flush_block!(stream, false);
    }

    BlockState::BlockDone
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct Match {
    match_start: u16,
    match_length: u16,
    strstart: u16,
    orgstart: u16,
}

fn emit_match(state: &mut State, m: Match) -> bool {
    let mut bflush = false;

    /* matches that are not long enough we need to emit as literals */
    if (m.match_length as usize) < WANT_MIN_MATCH {
        for lc in &state.window.filled()[state.strstart..][..m.match_length as usize] {
            bflush |= State::tally_lit_help(&mut state.sym_buf, &mut state.l_desc, *lc);
        }
    } else {
        // check_match(s, m.strstart, m.match_start, m.match_length);

        bflush |= state.tally_dist(
            (m.strstart - m.match_start) as usize,
            m.match_length as usize - STD_MIN_MATCH,
        );
    }
    state.lookahead -= m.match_length as usize;

    bflush
}

#[inline(always)]
fn insert_match(state: &mut State, mut m: Match) {
    if state.lookahead <= (m.match_length as usize + WANT_MIN_MATCH) {
        return;
    }

    /* matches that are not long enough we need to emit as literals */
    if (m.match_length as usize) < WANT_MIN_MATCH {
        m.strstart += 1;
        m.match_length -= 1;
        if m.match_length > 0 && m.strstart >= m.orgstart {
            if m.strstart + m.match_length > m.orgstart {
                state.insert_string(m.strstart as usize, m.match_length as usize);
            } else {
                state.insert_string(m.strstart as usize, (m.orgstart - m.strstart + 1) as usize);
            }
        }
        return;
    }

    // Insert new strings in the hash table only if the match length
    // is not too large. This saves time but degrades compression.
    if usize::from(m.match_length) <= 16 * state.max_insert_length()
        && state.lookahead >= WANT_MIN_MATCH
    {
        m.match_length -= 1; /* string at strstart already in table */
        m.strstart += 1;

        if m.strstart >= m.orgstart {
            if m.strstart + m.match_length > m.orgstart {
                state.insert_string(m.strstart as usize, m.match_length as usize);
            } else {
                state.insert_string(m.strstart as usize, (m.orgstart - m.strstart + 1) as usize);
            }
        } else if m.orgstart < m.strstart + m.match_length {
            state.insert_string(
                m.orgstart as usize,
                (m.strstart + m.match_length - m.orgstart) as usize,
            );
        }
    } else {
        m.strstart += m.match_length;
        m.match_length = 0;

        if (m.strstart as usize) >= (STD_MIN_MATCH - 2) {
            StandardHashCalc::quick_insert_string(state, m.strstart as usize + 2 - STD_MIN_MATCH);
        }

        /* If lookahead < WANT_MIN_MATCH, ins_h is garbage, but it does not
         * matter since it will be recomputed at next deflate call.
         */
    }
}

fn fizzle_matches(window: &[u8], max_dist: usize, current: &mut Match, next: &mut Match) {
    /* step zero: sanity checks */

    if current.match_length <= 1 {
        return;
    }

    if current.match_length > 1 + next.match_start {
        return;
    }

    if current.match_length > 1 + next.strstart {
        return;
    }

    let m = &window[(-(current.match_length as isize) + 1 + next.match_start as isize) as usize..];
    let orig = &window[(-(current.match_length as isize) + 1 + next.strstart as isize) as usize..];

    /* quick exit check.. if this fails then don't bother with anything else */
    if m[0] != orig[0] {
        return;
    }

    /* step one: try to move the "next" match to the left as much as possible */
    let limit = next.strstart.saturating_sub(max_dist as u16);

    let mut c = *current;
    let mut n = *next;

    let m = &window[..n.match_start as usize];
    let orig = &window[..n.strstart as usize];

    let mut m = m.iter().rev();
    let mut orig = orig.iter().rev();

    let mut changed = 0;

    while m.next() == orig.next() {
        if c.match_length < 1 {
            break;
        }
        if n.strstart <= limit {
            break;
        }
        if n.match_length >= 256 {
            break;
        }
        if n.match_start <= 1 {
            break;
        }

        n.strstart -= 1;
        n.match_start -= 1;
        n.match_length += 1;
        c.match_length -= 1;
        changed += 1;
    }

    if changed == 0 {
        return;
    }

    if c.match_length <= 1 && n.match_length != 2 {
        n.orgstart += 1;
        *current = c;
        *next = n;
    }
}
