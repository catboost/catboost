use crate::{
    deflate::{
        flush_pending, read_buf_window, zng_tr_stored_block, BlockState, DeflateStream, MAX_STORED,
    },
    DeflateFlush,
};

pub fn deflate_stored(stream: &mut DeflateStream, flush: DeflateFlush) -> BlockState {
    // Smallest worthy block size when not flushing or finishing. By default
    // this is 32K. This can be as small as 507 bytes for memLevel == 1. For
    // large input and output buffers, the stored block size will be larger.
    let min_block = Ord::min(
        stream.state.bit_writer.pending.capacity() - 5,
        stream.state.w_size,
    );

    // Copy as many min_block or larger stored blocks directly to next_out as
    // possible. If flushing, copy the remaining available input to next_out as
    // stored blocks, if there is enough space.

    // unsigned len, left, have, last = 0;
    let mut have;
    let mut last = false;
    let mut used = stream.avail_in;
    loop {
        // maximum deflate stored block length
        let mut len = MAX_STORED;

        // number of header bytes
        have = ((stream.state.bit_writer.bits_valid + 42) / 8) as usize;

        // we need room for at least the header
        if stream.avail_out < have as u32 {
            break;
        }

        let left = stream.state.strstart as isize - stream.state.block_start;
        let left = Ord::max(0, left) as usize;

        have = stream.avail_out as usize - have;

        if len > left + stream.avail_in as usize {
            // limit len to the input
            len = left + stream.avail_in as usize;
        }

        len = Ord::min(len, have);

        // If the stored block would be less than min_block in length, or if
        // unable to copy all of the available input when flushing, then try
        // copying to the window and the pending buffer instead. Also don't
        // write an empty block when flushing -- deflate() does that.
        if len < min_block
            && ((len == 0 && flush != DeflateFlush::Finish)
                || flush == DeflateFlush::NoFlush
                || len != left + stream.avail_in as usize)
        {
            break;
        }

        // Make a dummy stored block in pending to get the header bytes,
        // including any pending bits. This also updates the debugging counts.
        last = flush == DeflateFlush::Finish && len == left + stream.avail_in as usize;
        zng_tr_stored_block(stream.state, 0..0, last);

        /* Replace the lengths in the dummy stored block with len. */
        stream.state.bit_writer.pending.rewind(4);
        stream
            .state
            .bit_writer
            .pending
            .extend(&(len as u16).to_le_bytes());
        stream
            .state
            .bit_writer
            .pending
            .extend(&(!len as u16).to_le_bytes());

        // Write the stored block header bytes.
        flush_pending(stream);

        // Update debugging counts for the data about to be copied.
        stream.state.bit_writer.cmpr_bits_add(len << 3);
        stream.state.bit_writer.sent_bits_add(len << 3);

        if left > 0 {
            // SAFETY: `len` is effectively `min(stream.avail_in, stream.avail_out)`, so any reads
            // of `len` won't go out of bounds on `next_out`. `left` is calculated from indices of
            // the window, so `left` reads of the window won't go out of bounds.
            let left = Ord::min(left, len);
            let src = &stream.state.window.filled()[stream.state.block_start as usize..];
            unsafe { core::ptr::copy_nonoverlapping(src.as_ptr(), stream.next_out, left) };

            stream.next_out = stream.next_out.wrapping_add(left);
            stream.avail_out = stream.avail_out.wrapping_sub(left as _);
            stream.total_out = stream.total_out.wrapping_add(left as _);
            stream.state.block_start += left as isize;
            len -= left;
        }

        // Copy uncompressed bytes directly from next_in to next_out, updating the check value.
        if len > 0 {
            read_buf_direct_copy(stream, len);
        }

        if last {
            break;
        }
    }

    // Update the sliding window with the last s->w_size bytes of the copied
    // data, or append all of the copied data to the existing window if less
    // than s->w_size bytes were copied. Also update the number of bytes to
    // insert in the hash tables, in the event that deflateParams() switches to
    // a non-zero compression level.
    used -= stream.avail_in; /* number of input bytes directly copied */

    if used > 0 {
        let state = &mut stream.state;
        // If any input was used, then no unused input remains in the window, therefore s->block_start == s->strstart.
        if used as usize >= state.w_size {
            /* supplant the previous history */
            state.matches = 2; /* clear hash */

            // SAFETY: we've advanced the next_in pointer at minimum w_size bytes
            // read_buf_direct_copy(), so we are able to backtrack that number of bytes.
            let src = stream.next_in.wrapping_sub(state.w_size);
            unsafe { state.window.copy_and_initialize(0..state.w_size, src) };

            state.strstart = state.w_size;
            state.insert = state.strstart;
        } else {
            if state.window_size - state.strstart <= used as usize {
                /* Slide the window down. */
                state.strstart -= state.w_size;

                // make sure we don't copy uninitialized bytes. While we discard the first lower w_size
                // bytes, it is not guaranteed that the upper w_size bytes are all initialized
                let copy = Ord::min(state.strstart, state.window.filled().len() - state.w_size);

                state
                    .window
                    .filled_mut()
                    .copy_within(state.w_size..state.w_size + copy, 0);

                if state.matches < 2 {
                    state.matches += 1; /* add a pending slide_hash() */
                }
                state.insert = Ord::min(state.insert, state.strstart);
            }

            // SAFETY: we've advanced the next_in pointer at least `used` bytes
            // read_buf_direct_copy(), so we are able to backtrack that number of bytes.
            let src = stream.next_in.wrapping_sub(used as usize);
            let dst = state.strstart..state.strstart + used as usize;
            unsafe { state.window.copy_and_initialize(dst, src) };

            state.strstart += used as usize;
            state.insert += Ord::min(used as usize, state.w_size - state.insert);
        }
        state.block_start = state.strstart as isize;
    }

    // If the last block was written to next_out, then done.
    if last {
        stream.state.bit_writer.bits_used = 8;
        return BlockState::FinishDone;
    }

    // If flushing and all input has been consumed, then done.
    if flush != DeflateFlush::NoFlush
        && flush != DeflateFlush::Finish
        && stream.avail_in == 0
        && stream.state.strstart as isize == stream.state.block_start
    {
        return BlockState::BlockDone;
    }

    // Fill the window with any remaining input
    let mut have = stream.state.window_size - stream.state.strstart;
    if stream.avail_in as usize > have && stream.state.block_start >= stream.state.w_size as isize {
        // slide the window down
        let state = &mut stream.state;
        state.block_start -= state.w_size as isize;
        state.strstart -= state.w_size;

        // make sure we don't copy uninitialized bytes. While we discard the first lower w_size
        // bytes, it is not guaranteed that the upper w_size bytes are all initialized
        let copy = Ord::min(state.strstart, state.window.filled().len() - state.w_size);

        state
            .window
            .filled_mut()
            .copy_within(state.w_size..state.w_size + copy, 0);

        if state.matches < 2 {
            // add a pending slide_hash
            state.matches += 1;
        }

        have += state.w_size; // more space now
        state.insert = Ord::min(state.insert, state.strstart);
    }

    let have = Ord::min(have, stream.avail_in as usize);
    if have > 0 {
        read_buf_window(stream, stream.state.strstart, have);

        let state = &mut stream.state;
        state.strstart += have;
        state.insert += Ord::min(have, state.w_size - state.insert);
    }

    // There was not enough avail_out to write a complete worthy or flushed
    // stored block to next_out. Write a stored block to pending instead, if we
    // have enough input for a worthy block, or if flushing and there is enough
    // room for the remaining input as a stored block in the pending buffer.

    // number of header bytes
    let state = &mut stream.state;
    let have = ((state.bit_writer.bits_valid + 42) >> 3) as usize;

    // maximum stored block length that will fit in pending:
    let have = Ord::min(state.bit_writer.pending.capacity() - have, MAX_STORED);
    let min_block = Ord::min(have, state.w_size);
    let left = state.strstart as isize - state.block_start;

    if left >= min_block as isize
        || ((left > 0 || flush == DeflateFlush::Finish)
            && flush != DeflateFlush::NoFlush
            && stream.avail_in == 0
            && left <= have as isize)
    {
        let len = Ord::min(left as usize, have); // TODO wrapping?
        last = flush == DeflateFlush::Finish && stream.avail_in == 0 && len == (left as usize);

        let range = state.block_start as usize..state.block_start as usize + len;
        zng_tr_stored_block(state, range, last);

        state.block_start += len as isize;
        flush_pending(stream);
    }

    // We've done all we can with the available input and output.
    if last {
        stream.state.bit_writer.bits_used = 8;
        BlockState::FinishStarted
    } else {
        BlockState::NeedMore
    }
}

fn read_buf_direct_copy(stream: &mut DeflateStream, size: usize) -> usize {
    let len = Ord::min(stream.avail_in as usize, size);
    let output = stream.next_out;

    if len == 0 {
        return 0;
    }

    stream.avail_in -= len as u32;

    // SAFETY: len is effectively bounded by next_in and next_out (via size derived in the calling
    // function), so copies are in-bounds.
    if stream.state.wrap == 2 {
        // we likely cannot fuse the crc32 and the copy here because the input can be changed by
        // a concurrent thread. Therefore it cannot be converted into a slice!
        unsafe { core::ptr::copy_nonoverlapping(stream.next_in, output, len) }

        let data = unsafe { core::slice::from_raw_parts(output, len) };
        stream.state.crc_fold.fold(data, 0);
    } else if stream.state.wrap == 1 {
        // we cannot fuse the adler and the copy in our case, because adler32 takes a slice.
        // Another process is allowed to concurrently modify stream.next_in, so we cannot turn it
        // into a rust slice (violates its safety requirements)
        unsafe { core::ptr::copy_nonoverlapping(stream.next_in, output, len) }

        let data = unsafe { core::slice::from_raw_parts(output, len) };
        stream.adler = crate::adler32::adler32(stream.adler as u32, data) as _;
    } else {
        unsafe { core::ptr::copy_nonoverlapping(stream.next_in, output, len) }
    }

    stream.next_in = stream.next_in.wrapping_add(len);
    stream.total_in += len as crate::c_api::z_size;

    stream.next_out = stream.next_out.wrapping_add(len as _);
    stream.avail_out = stream.avail_out.wrapping_sub(len as _);
    stream.total_out = stream.total_out.wrapping_add(len as _);

    len
}
