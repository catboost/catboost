use core::ffi::c_void;
use core::marker::PhantomData;

use crate::allocate::Allocator;
use crate::c_api::{in_func, internal_state, out_func};
use crate::inflate::bitreader::BitReader;
use crate::inflate::inftrees::{inflate_table, CodeType, InflateTable};
use crate::inflate::{
    Codes, Flags, InflateAllocOffsets, InflateConfig, InflateStream, Mode, State, Table, Window,
    INFLATE_FAST_MIN_HAVE, INFLATE_FAST_MIN_LEFT, INFLATE_STRICT, MAX_BITS, MAX_DIST_EXTRA_BITS,
};
use crate::traceln;
use crate::{c_api::z_stream, inflate::writer::Writer, ReturnCode};
use crate::{MAX_WBITS, MIN_WBITS};

/// Initialize the stream in an inflate state
pub fn back_init(stream: &mut z_stream, config: InflateConfig, window: Window) -> ReturnCode {
    assert_eq!(1 << config.window_bits, window.buffer_size());

    if !(MIN_WBITS..=MAX_WBITS).contains(&config.window_bits) {
        return ReturnCode::StreamError as _;
    }

    stream.msg = core::ptr::null_mut();

    // for safety we must really make sure that alloc and free are consistent
    // this is a (slight) deviation from stock zlib. In this crate we pick the rust
    // allocator as the default, but `libz-rs-sys` configures the C allocator
    #[cfg(feature = "rust-allocator")]
    if stream.zalloc.is_none() || stream.zfree.is_none() {
        stream.configure_default_rust_allocator()
    }

    #[cfg(feature = "c-allocator")]
    if stream.zalloc.is_none() || stream.zfree.is_none() {
        stream.configure_default_c_allocator()
    }

    if stream.zalloc.is_none() || stream.zfree.is_none() {
        return ReturnCode::StreamError;
    }

    let mut state = State::new(&[], Writer::new(&mut []));

    // TODO this can change depending on the used/supported SIMD instructions
    state.chunksize = 32;

    let alloc = Allocator {
        zalloc: stream.zalloc.unwrap(),
        zfree: stream.zfree.unwrap(),
        opaque: stream.opaque,
        _marker: PhantomData,
    };
    let allocs = InflateAllocOffsets::new();

    let Some(allocation_start) = alloc.allocate_slice_raw::<u8>(allocs.total_size) else {
        return ReturnCode::MemError;
    };

    let address = allocation_start.as_ptr() as usize;
    let align_offset = address.next_multiple_of(64) - address;
    let buf = unsafe { allocation_start.as_ptr().add(align_offset) };

    // NOTE: the window part of the allocation is ignored in this case.
    state.window = window;

    let state_allocation = unsafe { buf.add(allocs.state_pos).cast::<State>() };
    unsafe { state_allocation.write(state) };
    stream.state = state_allocation.cast::<internal_state>();

    // SAFETY: we've correctly initialized the stream to be an InflateStream
    let Some(stream) = (unsafe { InflateStream::from_stream_mut(stream) }) else {
        return ReturnCode::StreamError;
    };

    stream.state.allocation_start = allocation_start.as_ptr();
    stream.state.total_allocation_size = allocs.total_size;

    stream.state.wbits = config.window_bits as u8;
    stream.state.flags.update(Flags::SANE, true);

    ReturnCode::Ok
}

pub unsafe fn back(
    strm: &mut InflateStream,
    in_: in_func,
    in_desc: *mut c_void,
    out: out_func,
    out_desc: *mut c_void,
) -> ReturnCode {
    let mut ret;

    /* Reset the state */
    strm.msg = core::ptr::null_mut();
    strm.state.mode = Mode::Type;
    strm.state.flags.update(Flags::IS_LAST_BLOCK, false);
    strm.state.window.clear();

    let mut next = strm.next_in.cast_const();
    let mut have = if !next.is_null() { strm.avail_in } else { 0 };
    let mut hold = 0;
    let mut bits = 0u8;
    let mut put = strm.state.window.as_mut_ptr();
    let mut left = strm.state.window.buffer_size();

    let state = &mut strm.state;

    'inf_leave: loop {
        macro_rules! initbits {
            () => {
                hold = 0;
                bits = 0;
            };
        }

        macro_rules! bytebits {
            () => {
                hold >>= bits & 7;
                bits -= bits & 7;
            };
        }

        macro_rules! dropbits {
            ($n:expr) => {
                hold >>= $n;
                bits -= $n;
            };
        }

        macro_rules! bits {
            ($n:expr) => {
                hold & ((1 << $n) - 1)
            };
        }

        macro_rules! needbits {
            ($n:expr) => {
                while usize::from(bits) < $n {
                    pullbyte!();
                }
            };
        }

        macro_rules! pull {
            () => {
                if have == 0 {
                    have = unsafe { in_(in_desc, &mut next) };
                    if have == 0 {
                        #[allow(unused_assignments)]
                        {
                            next = core::ptr::null();
                        }
                        ret = ReturnCode::BufError;
                        break 'inf_leave;
                    }
                }
            };
        }

        macro_rules! pullbyte {
            () => {
                pull!();
                have -= 1;
                hold += (unsafe { *next as u64 }) << bits;
                next = unsafe { next.add(1) };
                bits += 8;
            };
        }

        macro_rules! room {
            () => {
                if left == 0 {
                    left = state.window.buffer_size();
                    put = state.window.as_mut_ptr();

                    unsafe { state.window.set_have(left) };

                    if unsafe { out(out_desc, put, left as u32) } != 0 {
                        ret = ReturnCode::BufError;
                        break 'inf_leave;
                    }
                }
            };
        }

        match state.mode {
            Mode::Type => {
                if state.flags.contains(Flags::IS_LAST_BLOCK) {
                    bytebits!();
                    state.mode = Mode::Done;
                    continue;
                }

                needbits!(3);

                let last = bits!(1) != 0;
                state.flags.update(Flags::IS_LAST_BLOCK, last);
                dropbits!(1);

                match bits!(2) {
                    0b00 => {
                        traceln!("inflate:     stored block (last = {last})");

                        dropbits!(2);
                        state.mode = Mode::Stored;
                        continue;
                    }
                    0b01 => {
                        traceln!("inflate:     fixed codes block (last = {last})");

                        state.len_table = Table {
                            codes: Codes::Fixed,
                            bits: 9,
                        };

                        state.dist_table = Table {
                            codes: Codes::Fixed,
                            bits: 5,
                        };

                        dropbits!(2);
                        state.mode = Mode::Len;
                        continue;
                    }
                    0b10 => {
                        traceln!("inflate:     dynamic codes block (last = {last})");

                        dropbits!(2);
                        state.mode = Mode::Table;
                        continue;
                    }
                    0b11 => {
                        traceln!("inflate:     invalid block type");

                        dropbits!(2);
                        state.mode = Mode::Bad;
                        state.bad("invalid block type\0");
                        continue;
                    }
                    _ => {
                        // LLVM will optimize this branch away
                        unreachable!("BitReader::bits(2) only yields a value of two bits, so this match is already exhaustive")
                    }
                }
            }
            Mode::Stored => {
                bytebits!();
                needbits!(32);

                if hold as u16 != !((hold >> 16) as u16) {
                    state.mode = Mode::Bad;
                    state.bad("invalid stored block lengths\0");
                    continue;
                }

                state.length = hold as usize & 0xFFFF;
                traceln!("inflate:     stored length {}", state.length);

                initbits!();

                /* copy stored block from input to output */
                while state.length != 0 {
                    let mut copy = state.length;

                    pull!();
                    room!();

                    copy = Ord::min(copy, have as usize);
                    copy = Ord::min(copy, left);

                    unsafe { core::ptr::copy(next, put, copy) };

                    have -= copy as u32;
                    next = unsafe { next.add(copy) };

                    left -= copy;
                    put = unsafe { put.add(copy) };

                    state.length -= copy;
                }

                state.mode = Mode::Type;
                continue;
            }
            Mode::Table => {
                needbits!(14);
                state.nlen = bits!(5) as usize + 257;
                dropbits!(5);
                state.ndist = bits!(5) as usize + 1;
                dropbits!(5);
                state.ncode = bits!(4) as usize + 4;
                dropbits!(4);

                // TODO pkzit_bug_workaround
                if state.nlen > 286 || state.ndist > 30 {
                    state.mode = Mode::Bad;
                    state.bad("too many length or distance symbols\0");
                    continue;
                }

                traceln!("inflate:       table sizes ok");
                state.have = 0;

                // permutation of code lengths ;
                const ORDER: [u8; 19] = [
                    16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
                ];

                while state.have < state.ncode {
                    needbits!(3);
                    state.lens[usize::from(ORDER[state.have])] = bits!(3) as u16;
                    state.have += 1;
                    dropbits!(3);
                }

                while state.have < 19 {
                    state.lens[usize::from(ORDER[state.have])] = 0;
                    state.have += 1;
                }

                let InflateTable::Success { root, used } = inflate_table(
                    CodeType::Codes,
                    &state.lens[..19],
                    &mut state.codes_codes,
                    7,
                    &mut state.work,
                ) else {
                    state.mode = Mode::Bad;
                    state.bad("invalid code lengths set\0");
                    continue;
                };

                state.next = used;
                state.len_table.codes = Codes::Codes;
                state.len_table.bits = root;

                traceln!("inflate:       table sizes ok");
                state.have = 0;

                while state.have < state.nlen + state.ndist {
                    let here = loop {
                        let here = state.len_table_get(bits!(state.len_table.bits) as usize);
                        if here.bits <= bits {
                            break here;
                        }

                        pullbyte!();
                    };

                    let here_bits = here.bits;

                    match here.val {
                        0..=15 => {
                            dropbits!(here_bits);
                            state.lens[state.have] = here.val;
                            state.have += 1;
                        }
                        16 => {
                            needbits!(usize::from(here_bits) + 2);
                            dropbits!(here_bits);
                            if state.have == 0 {
                                state.mode = Mode::Bad;
                                state.bad("invalid bit length repeat\0");
                                continue 'inf_leave;
                            }

                            let len = state.lens[state.have - 1];
                            let copy = 3 + bits!(2) as usize;
                            dropbits!(2);

                            if state.have + copy > state.nlen + state.ndist {
                                state.mode = Mode::Bad;
                                state.bad("invalid bit length repeat\0");
                                continue 'inf_leave;
                            }

                            state.lens[state.have..][..copy].fill(len);
                            state.have += copy;
                        }
                        17 => {
                            needbits!(usize::from(here_bits) + 3);
                            dropbits!(here_bits);
                            let copy = 3 + bits!(3) as usize;
                            dropbits!(3);

                            if state.have + copy > state.nlen + state.ndist {
                                state.mode = Mode::Bad;
                                state.bad("invalid bit length repeat\0");
                                continue 'inf_leave;
                            }

                            state.lens[state.have..][..copy].fill(0);
                            state.have += copy;
                        }
                        18.. => {
                            needbits!(usize::from(here_bits) + 7);
                            dropbits!(here_bits);
                            let copy = 11 + bits!(7) as usize;
                            dropbits!(7);

                            if state.have + copy > state.nlen + state.ndist {
                                state.mode = Mode::Bad;
                                state.bad("invalid bit length repeat\0");
                                continue 'inf_leave;
                            }

                            state.lens[state.have..][..copy].fill(0);
                            state.have += copy;
                        }
                    }
                }

                // check for end-of-block code (better have one)
                if state.lens[256] == 0 {
                    state.mode = Mode::Bad;
                    state.bad("invalid code -- missing end-of-block\0");
                    continue 'inf_leave;
                }

                // build code tables

                let InflateTable::Success { root, used } = inflate_table(
                    CodeType::Lens,
                    &state.lens[..state.nlen],
                    &mut state.len_codes,
                    10,
                    &mut state.work,
                ) else {
                    state.mode = Mode::Bad;
                    state.bad("invalid literal/lengths set\0");
                    continue 'inf_leave;
                };

                state.len_table.codes = Codes::Len;
                state.len_table.bits = root;
                state.next = used;

                let InflateTable::Success { root, used } = inflate_table(
                    CodeType::Dists,
                    &state.lens[state.nlen..][..state.ndist],
                    &mut state.dist_codes,
                    9,
                    &mut state.work,
                ) else {
                    state.mode = Mode::Bad;
                    state.bad("invalid distances set\0");
                    continue 'inf_leave;
                };

                state.dist_table.bits = root;
                state.dist_table.codes = Codes::Dist;
                state.next += used;

                state.mode = Mode::Len;
            }
            Mode::Len => {
                if (have as usize) >= INFLATE_FAST_MIN_HAVE && left >= INFLATE_FAST_MIN_LEFT {
                    let mut bit_reader = BitReader::new(&[]);
                    unsafe { bit_reader.update_slice(next, have as usize) };
                    bit_reader.prime(bits, hold);

                    state.bit_reader = bit_reader;
                    state.writer = unsafe {
                        Writer::new_uninit_raw(
                            put.wrapping_sub(state.window.buffer_size() - left),
                            state.window.buffer_size() - left,
                            state.window.buffer_size(),
                        )
                    };

                    if state.window.have() < state.window.buffer_size() {
                        unsafe { state.window.set_have(state.window.buffer_size() - left) };
                    }

                    unsafe { inflate_fast_back(state) };

                    hold = state.bit_reader.hold();
                    bits = state.bit_reader.bits_in_buffer();

                    next = state.bit_reader.as_ptr();
                    have = state.bit_reader.bytes_remaining() as u32;

                    put = state.writer.next_out().cast();
                    left = state.writer.remaining();

                    continue 'inf_leave;
                }

                let len_table = match state.len_table.codes {
                    Codes::Fixed => &crate::inflate::inffixed_tbl::LENFIX[..],
                    Codes::Codes => &state.codes_codes,
                    Codes::Len => &state.len_codes,
                    Codes::Dist => &state.dist_codes,
                };

                // get a literal, length, or end-of-block code
                let mut here;
                loop {
                    here = len_table[bits!(state.len_table.bits) as usize];

                    if here.bits <= bits {
                        break;
                    }

                    pullbyte!();
                }

                if here.op != 0 && here.op & 0xf0 == 0 {
                    let last = here;
                    loop {
                        let tmp = bits!((last.bits + last.op) as usize) as u16;
                        here = len_table[(last.val + (tmp >> last.bits)) as usize];
                        if last.bits + here.bits <= bits {
                            break;
                        }

                        pullbyte!();
                    }

                    dropbits!(last.bits);
                }

                dropbits!(here.bits);
                state.length = here.val as usize;

                if here.op == 0 {
                    if here.val >= 0x20 && here.val < 0x7f {
                        traceln!("inflate:         literal '{}'", here.val as u8 as char);
                    } else {
                        traceln!("inflate:         literal {:#04x}", here.val);
                    }
                    room!();

                    unsafe {
                        *put = state.length as u8;
                        put = put.add(1)
                    }
                    left -= 1;

                    state.mode = Mode::Len;
                    continue;
                } else if here.op & 32 != 0 {
                    // end of block

                    traceln!("inflate:         end of block");

                    state.mode = Mode::Type;
                    continue;
                } else if here.op & 64 != 0 {
                    state.mode = Mode::Bad;
                    state.bad("invalid literal/length code\0");
                    continue;
                } else {
                    // length code
                    state.extra = (here.op & MAX_BITS) as usize;
                }

                // get extra bits, if any
                if state.extra != 0 {
                    needbits!(state.extra);
                    state.length += bits!(state.extra) as usize;
                    dropbits!(state.extra as u8);
                }
                traceln!("inflate:         length {}", state.length);

                // get distance code
                let mut here;
                loop {
                    here = state.dist_table_get(bits!(state.dist_table.bits) as usize);
                    if here.bits <= bits {
                        break;
                    }

                    pullbyte!();
                }

                if here.op & 0xf0 == 0 {
                    let last = here;

                    loop {
                        here = state.dist_table_get(
                            last.val as usize
                                + ((bits!((last.bits + last.op) as usize) as usize) >> last.bits),
                        );

                        if last.bits + here.bits <= bits {
                            break;
                        }

                        pullbyte!();
                    }

                    dropbits!(last.bits);
                }

                dropbits!(here.bits);

                if here.op & 64 != 0 {
                    state.mode = Mode::Bad;
                    state.bad("invalid distance code\0");
                    continue 'inf_leave;
                }

                state.offset = here.val as usize;

                state.extra = (here.op & MAX_BITS) as usize;

                let extra = state.extra;

                if extra > 0 {
                    needbits!(extra);
                    state.offset += bits!(extra) as usize;
                    dropbits!(extra as u8);
                }

                if INFLATE_STRICT
                    && state.offset
                        > state.window.buffer_size()
                            - (if state.window.have() < state.window.buffer_size() {
                                left
                            } else {
                                0
                            })
                {
                    state.mode = Mode::Bad;
                    state.bad("invalid distance too far back\0");
                    continue 'inf_leave;
                }

                traceln!("inflate:         distance {}", state.offset);

                loop {
                    room!();
                    let mut copy = state.window.buffer_size() - state.offset;
                    let mut from;

                    if copy < left {
                        from = put.wrapping_add(copy);
                        copy = left - copy;
                    } else {
                        from = put.wrapping_sub(state.offset);
                        copy = left;
                    }

                    copy = Ord::min(copy, state.length);
                    state.length -= copy;
                    left -= copy;

                    for _ in 0..copy {
                        unsafe {
                            *put = *from;
                            put = put.add(1);
                            from = from.add(1);
                        }
                    }

                    if state.length == 0 {
                        break;
                    }
                }

                continue 'inf_leave;
            }
            Mode::Done => {
                ret = ReturnCode::StreamEnd;
                break 'inf_leave;
            }
            Mode::Bad => {
                ret = ReturnCode::DataError;
                break 'inf_leave;
            }

            Mode::Head
            | Mode::Flags
            | Mode::Time
            | Mode::Os
            | Mode::ExLen
            | Mode::Extra
            | Mode::Name
            | Mode::Comment
            | Mode::HCrc
            | Mode::Sync
            | Mode::Mem
            | Mode::Length
            | Mode::TypeDo
            | Mode::CopyBlock
            | Mode::Check
            | Mode::Len_
            | Mode::Lit
            | Mode::LenExt
            | Mode::Dist
            | Mode::DistExt
            | Mode::Match
            | Mode::LenLens
            | Mode::CodeLens
            | Mode::DictId
            | Mode::Dict => {
                // All other states should be unreachable, and return StreamError.
                ret = ReturnCode::StreamError;
                break 'inf_leave;
            }
        }
    }

    if left < state.window.buffer_size()
        && unsafe {
            out(
                out_desc,
                state.window.as_mut_ptr(),
                state.window.buffer_size() as u32 - left as u32,
            )
        } != 0
        && ret == ReturnCode::StreamEnd
    {
        ret = ReturnCode::BufError;
    }

    strm.next_in = next.cast_mut();
    strm.avail_in = have;

    ret
}

#[inline(always)]
unsafe fn inflate_fast_back(state: &mut State) {
    let mut bit_reader = BitReader::new(&[]);
    core::mem::swap(&mut bit_reader, &mut state.bit_reader);
    debug_assert!(bit_reader.bytes_remaining() >= 15);

    let mut writer = Writer::new(&mut []);
    core::mem::swap(&mut writer, &mut state.writer);

    let lcode = state.len_table_ref();
    let dcode = state.dist_table_ref();

    // IDEA: use const generics for the bits here?
    let lmask = (1u64 << state.len_table.bits) - 1;
    let dmask = (1u64 << state.dist_table.bits) - 1;

    // TODO verify if this is relevant for us
    let extra_safe = false;

    let window_size = state.window.buffer_size();

    let mut bad = None;

    if bit_reader.bits_in_buffer() < 10 {
        debug_assert!(bit_reader.bytes_remaining() >= 15);
        // Safety: Caller ensured that bit_reader has >= 15 bytes available; refill only needs 8.
        unsafe { bit_reader.refill() };
    }
    // We had at least 15 bytes in the slice, plus whatever was in the buffer. After filling the
    // buffer from the slice, we now have at least 8 bytes remaining in the slice, plus a full buffer.
    debug_assert!(
        bit_reader.bytes_remaining() >= 8 && bit_reader.bytes_remaining_including_buffer() >= 15
    );

    'outer: loop {
        // This condition is ensured above for the first iteration of the `outer` loop. For
        // subsequent iterations, the loop continuation condition is
        // `bit_reader.bytes_remaining_including_buffer() > 15`. And because the buffer
        // contributes at most 7 bytes to the result of bit_reader.bytes_remaining_including_buffer(),
        // that means that the slice contains at least 8 bytes.
        debug_assert!(
            bit_reader.bytes_remaining() >= 8
                && bit_reader.bytes_remaining_including_buffer() >= 15
        );

        let mut here = {
            let bits = bit_reader.bits_in_buffer();
            let hold = bit_reader.hold();

            // Safety: As described in the comments for the debug_assert at the start of
            // the `outer` loop, it is guaranteed that `bit_reader.bytes_remaining() >= 8` here,
            // which satisfies the safety precondition for `refill`. And, because the total
            // number of bytes in `bit_reader`'s buffer plus its slice is at least 15, and
            // `refill` moves at most 7 bytes from the slice to the buffer, the slice will still
            // contain at least 8 bytes after this `refill` call.
            unsafe { bit_reader.refill() };
            // After the refill, there will be at least 8 bytes left in the bit_reader's slice.
            debug_assert!(bit_reader.bytes_remaining() >= 8);

            // in most cases, the read can be interleaved with the logic
            // based on benchmarks this matters in practice. wild.
            if bits as usize >= state.len_table.bits {
                lcode[(hold & lmask) as usize]
            } else {
                lcode[(bit_reader.hold() & lmask) as usize]
            }
        };

        if here.op == 0 {
            writer.push(here.val as u8);
            bit_reader.drop_bits(here.bits);
            here = lcode[(bit_reader.hold() & lmask) as usize];

            if here.op == 0 {
                writer.push(here.val as u8);
                bit_reader.drop_bits(here.bits);
                here = lcode[(bit_reader.hold() & lmask) as usize];
            }
        }

        'dolen: loop {
            bit_reader.drop_bits(here.bits);
            let op = here.op;

            if op == 0 {
                writer.push(here.val as u8);
            } else if op & 16 != 0 {
                let op = op & MAX_BITS;
                let mut len = here.val + bit_reader.bits(op as usize) as u16;
                bit_reader.drop_bits(op);

                here = dcode[(bit_reader.hold() & dmask) as usize];

                // we have two fast-path loads: 10+10 + 15+5 = 40,
                // but we may need to refill here in the worst case
                if bit_reader.bits_in_buffer() < MAX_BITS + MAX_DIST_EXTRA_BITS {
                    debug_assert!(bit_reader.bytes_remaining() >= 8);
                    // Safety: On the first iteration of the `dolen` loop, we can rely on the
                    // invariant documented for the previous `refill` call above: after that
                    // operation, `bit_reader.bytes_remining >= 8`, which satisfies the safety
                    // precondition for this call. For subsequent iterations, this invariant
                    // remains true because nothing else within the `dolen` loop consumes data
                    // from the slice.
                    unsafe { bit_reader.refill() };
                }

                'dodist: loop {
                    bit_reader.drop_bits(here.bits);
                    let op = here.op;

                    if op & 16 != 0 {
                        let op = op & MAX_BITS;
                        let dist = here.val + bit_reader.bits(op as usize) as u16;

                        if INFLATE_STRICT && dist as usize > state.dmax {
                            bad = Some("invalid distance too far back\0");
                            state.mode = Mode::Bad;
                            break 'outer;
                        }

                        bit_reader.drop_bits(op);

                        // max distance in output
                        let written = writer.len();

                        if dist as usize > written {
                            // copy fropm the window
                            if (dist as usize - written) > state.window.have() {
                                if state.flags.contains(Flags::SANE) {
                                    bad = Some("invalid distance too far back\0");
                                    state.mode = Mode::Bad;
                                    break 'outer;
                                }

                                panic!("INFLATE_ALLOW_INVALID_DISTANCE_TOOFAR_ARRR")
                            }

                            let mut op = dist as usize - written;
                            let mut from;

                            let window_next = state.window.next();

                            if window_next == 0 {
                                // This case is hit when the window has just wrapped around
                                // by logic in `Window::extend`. It is special-cased because
                                // apparently this is quite common.
                                //
                                // the match is at the end of the window, even though the next
                                // position has now wrapped around.
                                from = window_size - op;
                            } else if window_next >= op {
                                // the standard case: a contiguous copy from the window, no wrapping
                                from = window_next - op;
                            } else {
                                // This case is hit when the window has recently wrapped around
                                // by logic in `Window::extend`.
                                //
                                // The match is (partially) at the end of the window
                                op -= window_next;
                                from = window_size - op;

                                if op < len as usize {
                                    // This case is hit when part of the match is at the end of the
                                    // window, and part of it has wrapped around to the start. Copy
                                    // the end section here, the start section will be copied below.
                                    len -= op as u16;
                                    writer.extend_from_window_back(&state.window, from..from + op);
                                    from = 0;
                                    op = window_next;
                                }
                            }

                            let copy = Ord::min(op, len as usize);
                            writer.extend_from_window_back(&state.window, from..from + copy);

                            if op < len as usize {
                                // here we need some bytes from the output itself
                                writer.copy_match_back(dist as usize, len as usize - op);
                            }
                        } else if extra_safe {
                            todo!()
                        } else {
                            writer.copy_match_back(dist as usize, len as usize)
                        }
                    } else if (op & 64) == 0 {
                        // 2nd level distance code
                        here = dcode[(here.val + bit_reader.bits(op as usize) as u16) as usize];
                        continue 'dodist;
                    } else {
                        bad = Some("invalid distance code\0");
                        state.mode = Mode::Bad;
                        break 'outer;
                    }

                    break 'dodist;
                }
            } else if (op & 64) == 0 {
                // 2nd level length code
                here = lcode[(here.val + bit_reader.bits(op as usize) as u16) as usize];
                continue 'dolen;
            } else if op & 32 != 0 {
                // end of block
                state.mode = Mode::Type;
                break 'outer;
            } else {
                bad = Some("invalid literal/length code\0");
                state.mode = Mode::Bad;
                break 'outer;
            }

            break 'dolen;
        }

        // For normal `inflate`, include the bits in the bit_reader buffer in the count of available bytes.
        let remaining = bit_reader.bytes_remaining();
        if remaining >= INFLATE_FAST_MIN_HAVE && writer.remaining() >= INFLATE_FAST_MIN_LEFT {
            continue;
        }

        break 'outer;
    }

    // return unused bytes (on entry, bits < 8, so in won't go too far back)
    bit_reader.return_unused_bytes();

    state.bit_reader = bit_reader;
    state.writer = writer;

    if let Some(error_message) = bad {
        debug_assert!(matches!(state.mode, Mode::Bad));
        state.bad(error_message);
    }
}

pub fn back_end<'a>(strm: &'a mut InflateStream<'a>) {
    // With infback the window is user-supplied, so we mustn't try to free it.
    let _ = core::mem::replace(&mut strm.state.window, Window::empty());
    crate::inflate::end(strm);
}
