use core::cmp::min;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BroCatliResult {
    Success = 0,
    NeedsMoreInput = 1,
    NeedsMoreOutput = 2,
    BrotliFileNotCraftedForAppend = 124,
    InvalidWindowSize = 125,
    WindowSizeLargerThanPreviousFile = 126,
    BrotliFileNotCraftedForConcatenation = 127,
}

const NUM_STREAM_HEADER_BYTES: usize = 5;

#[derive(Clone, Copy)]
struct NewStreamData {
    bytes_so_far: [u8; NUM_STREAM_HEADER_BYTES],
    num_bytes_read: u8,
    num_bytes_written: Option<u8>,
}
impl NewStreamData {
    pub fn new() -> NewStreamData {
        NewStreamData {
            bytes_so_far: [0, 0, 0, 0, 0],
            num_bytes_read: 0,
            num_bytes_written: None,
        }
    }
    fn sufficient(&self) -> bool {
        if self.num_bytes_read == 4 && (127 & self.bytes_so_far[0]) != 17 {
            return true;
        }
        self.num_bytes_read == 5
    }
}

fn parse_window_size(bytes_so_far: &[u8]) -> Result<(u8, usize), ()> {
    // returns window_size and offset in stream in bits
    if bytes_so_far[0] & 1 == 0 {
        return Ok((16, 1));
    }
    match bytes_so_far[0] & 15 {
        0x3 => return Ok((18, 4)),
        0x5 => return Ok((19, 4)),
        0x7 => return Ok((20, 4)),
        0x9 => return Ok((21, 4)),
        0xb => return Ok((22, 4)),
        0xd => return Ok((23, 4)),
        0xf => return Ok((24, 4)),
        _ => match bytes_so_far[0] & 127 {
            0x71 => return Ok((15, 7)),
            0x61 => return Ok((14, 7)),
            0x51 => return Ok((13, 7)),
            0x41 => return Ok((12, 7)),
            0x31 => return Ok((11, 7)),
            0x21 => return Ok((10, 7)),
            0x1 => return Ok((17, 7)),
            _ => {}
        },
    }
    if (bytes_so_far[0] & 0x80) != 0 {
        return Err(());
    }
    let ret = bytes_so_far[1] & 0x3f;
    if !(10..=30).contains(&ret) {
        return Err(());
    }
    Ok((ret, 14))
}

fn detect_varlen_offset(bytes_so_far: &[u8]) -> Result<(usize), ()> {
    // returns offfset in bits
    let (_, mut offset) = match parse_window_size(bytes_so_far) {
        Ok(x) => x,
        Err(_) => return Err(()),
    };
    let mut bytes = 0u64;
    for (index, item) in bytes_so_far.iter().enumerate() {
        bytes |= u64::from(*item) << (index * 8);
    }
    bytes >>= offset;
    offset += 1;
    if (bytes & 1) != 0 {
        // ISLAST
        bytes >>= 1;
        offset += 1;
        if (bytes & 1) != 0 {
            // ISLASTEMPTY
            return Ok(offset);
        }
    }
    bytes >>= 1;
    let mut mnibbles = bytes & 3;
    bytes >>= 2;
    offset += 2;
    if mnibbles == 3 {
        // metadata block
        if (bytes & 1) != 0 {
            return Err(()); // reserved, must be zero
        }
        bytes >>= 1;
        offset += 1;
        let mskipbytes = bytes & ((1 << 2) - 1);
        offset += 2;
        offset += (mskipbytes as usize) * 8; // next item is byte aligned
        return Ok(offset);
    }
    mnibbles += 4;
    offset += (mnibbles as usize) * 4;
    bytes >>= mnibbles * 4;
    offset += 1;
    if (bytes & 1) == 0 {
        // not UNCOMPRESSED
        Err(()) // not valid bitstream for concatenation
    } else {
        // UNCOMPRESSED: now things are aligend
        Ok(offset)
    }
}

// eat your vegetables
#[derive(Default)]
pub struct BroCatli {
    last_bytes: [u8; 2],
    last_bytes_len: u8,
    last_byte_sanitized: bool,
    any_bytes_emitted: bool,
    last_byte_bit_offset: u8,
    // need to make sure that window sizes stay similar or get smaller
    window_size: u8,
    new_stream_pending: Option<NewStreamData>,
}

impl BroCatli {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn deserialize_from_buffer(buffer: &[u8]) -> Result<BroCatli, ()> {
        if 16 + NUM_STREAM_HEADER_BYTES > buffer.len() {
            return Err(());
        }
        let mut possible_new_stream_pending = NewStreamData {
            num_bytes_read: buffer[12],
            num_bytes_written: if (buffer[9] & (1 << 7)) != 0 {
                Some(buffer[13])
            } else {
                None
            },
            bytes_so_far: [0; NUM_STREAM_HEADER_BYTES],
        };
        let xlen = possible_new_stream_pending.bytes_so_far.len();
        possible_new_stream_pending
            .bytes_so_far
            .clone_from_slice(&buffer[16..16 + xlen]);
        let new_stream_pending: Option<NewStreamData> = if (buffer[9] & (1 << 6)) != 0 {
            Some(possible_new_stream_pending)
        } else {
            None
        };
        let mut ret = BroCatli {
            last_bytes: [0, 0],
            last_bytes_len: buffer[8],
            last_byte_sanitized: (buffer[9] & 0x1) != 0,
            last_byte_bit_offset: buffer[10],
            any_bytes_emitted: (buffer[9] & (1 << 5)) != 0,
            window_size: buffer[11],
            new_stream_pending,
        };
        if ret.last_bytes.len() > 8 {
            return Err(());
        }
        let xlen = ret.last_bytes.len();
        ret.last_bytes.clone_from_slice(&buffer[..xlen]);
        Ok(ret)
    }
    #[inline(always)]
    pub fn serialize_to_buffer(&self, buffer: &mut [u8]) -> Result<(), ()> {
        if 16 + NUM_STREAM_HEADER_BYTES > buffer.len() {
            return Err(());
        }
        buffer[..self.last_bytes.len()].clone_from_slice(&self.last_bytes[..]);
        buffer[8] = self.last_bytes_len;
        buffer[9] = (self.last_byte_sanitized as u8)
            | ((self.new_stream_pending.is_some() as u8) << 6)
            | ((self.any_bytes_emitted as u8) << 5);
        buffer[10] = self.last_byte_bit_offset;
        buffer[11] = self.window_size;
        if let Some(new_stream_pending) = self.new_stream_pending {
            if new_stream_pending.num_bytes_written.is_some() {
                buffer[9] |= (1 << 7);
            }
            buffer[12] = new_stream_pending.num_bytes_read;
            buffer[13] = new_stream_pending.num_bytes_written.unwrap_or(0);
            // 14, 15 reserved
            buffer[16..16 + new_stream_pending.bytes_so_far.len()]
                .clone_from_slice(&new_stream_pending.bytes_so_far[..]);
        }
        Ok(())
    }
    pub fn new_with_window_size(log_window_size: u8) -> BroCatli {
        // in this case setup the last_bytes of the stream to perfectly mimic what would
        // appear in an empty stream with the selected window size...
        // this means the window size followed by 2 sequential 1 bits (LAST_METABLOCK, EMPTY)
        // the new_stream code should naturally find the sequential 1 bits and mask them
        // out and then prepend the window size... then the following window sizes should
        // be checked to be shorter
        let last_bytes_len;
        let last_bytes;

        if log_window_size > 24 {
            last_bytes = [17u8, log_window_size | 64 | 128];
            last_bytes_len = 2;
        } else if log_window_size == 16 {
            last_bytes = [1 | 2 | 4, 0];
            last_bytes_len = 1;
        } else if log_window_size > 17 {
            last_bytes = [(3 + (log_window_size - 18) * 2) | (16 | 32), 0];
            last_bytes_len = 1;
        } else {
            match log_window_size {
                15 => last_bytes = [0x71 | 0x80, 1],
                14 => last_bytes = [0x61 | 0x80, 1],
                13 => last_bytes = [0x51 | 0x80, 1],
                12 => last_bytes = [0x41 | 0x80, 1],
                11 => last_bytes = [0x31 | 0x80, 1],
                10 => last_bytes = [0x21 | 0x80, 1],
                _ => {
                    assert_eq!(log_window_size, 17);
                    last_bytes = [0x1 | 0x80, 1];
                } // 17
            }
            last_bytes_len = 2;
        }
        BroCatli {
            last_bytes,
            last_bytes_len,
            last_byte_bit_offset: 0,
            last_byte_sanitized: false,
            any_bytes_emitted: false,
            new_stream_pending: None,
            window_size: log_window_size,
        }
    }

    pub fn new_brotli_file(&mut self) {
        self.new_stream_pending = Some(NewStreamData::new());
    }
    fn flush_previous_stream(
        &mut self,
        out_bytes: &mut [u8],
        out_offset: &mut usize,
    ) -> BroCatliResult {
        if !self.last_byte_sanitized {
            // if the previous stream hasn't had the last metablock (bit 1,1) sanitized
            if self.last_bytes_len == 0 {
                // first stream or otherwise sanitized
                self.last_byte_sanitized = true;
                return BroCatliResult::Success;
            }
            // create a 16 bit integer with the last 2 bytes of data
            let mut last_bytes = self.last_bytes[0] as u16 + ((self.last_bytes[1] as u16) << 8);
            let max = self.last_bytes_len * 8;
            let mut index = max - 1;
            for i in 0..max {
                index = max - 1 - i;
                if ((1 << index) & last_bytes) != 0 {
                    break; // find the highest set bit
                }
            }
            if index == 0 {
                // if the bit is too low, return failure, since both bits could not possibly have been set
                return BroCatliResult::BrotliFileNotCraftedForAppend;
            }
            if (last_bytes >> (index - 1)) != 3 {
                // last two bits need to be set for the final metablock
                return BroCatliResult::BrotliFileNotCraftedForAppend;
            }
            index -= 1; // discard the final two bits
            last_bytes &= (1 << index) - 1; // mask them out
            self.last_bytes[0] = last_bytes as u8; // reset the last_bytes pair
            self.last_bytes[1] = (last_bytes >> 8) as u8;
            if index >= 8 {
                // if both bits and one useful bit were in the second block, then write that
                if out_bytes.len() > *out_offset {
                    out_bytes[*out_offset] = self.last_bytes[0];
                    self.last_bytes[0] = self.last_bytes[1];
                    *out_offset += 1;
                    self.any_bytes_emitted = true;
                    index -= 8;
                    self.last_bytes_len -= 1;
                } else {
                    return BroCatliResult::NeedsMoreOutput;
                }
            }
            self.last_byte_bit_offset = index;
            assert!(index < 8);
            self.last_byte_sanitized = true;
        }
        BroCatliResult::Success
    }

    fn shift_and_check_new_stream_header(
        &mut self,
        mut new_stream_pending: NewStreamData,
        out_bytes: &mut [u8],
        out_offset: &mut usize,
    ) -> BroCatliResult {
        if new_stream_pending.num_bytes_written.is_none() {
            let (window_size, window_offset) = if let Ok(results) = parse_window_size(
                &new_stream_pending.bytes_so_far[..usize::from(new_stream_pending.num_bytes_read)],
            ) {
                results
            } else {
                return BroCatliResult::InvalidWindowSize;
            };
            if self.window_size == 0 {
                // parse window size and just copy everything
                self.window_size = window_size;
                assert_eq!(self.last_byte_bit_offset, 0); // we are first stream
                out_bytes[*out_offset] = new_stream_pending.bytes_so_far[0];
                new_stream_pending.num_bytes_written = Some(1);
                self.any_bytes_emitted = true;
                *out_offset += 1;
            } else {
                if window_size > self.window_size {
                    return BroCatliResult::WindowSizeLargerThanPreviousFile;
                }
                let mut realigned_header: [u8; NUM_STREAM_HEADER_BYTES + 1] =
                    [self.last_bytes[0], 0, 0, 0, 0, 0];
                let varlen_offset = if let Ok(voffset) = detect_varlen_offset(
                    &new_stream_pending.bytes_so_far
                        [..usize::from(new_stream_pending.num_bytes_read)],
                ) {
                    voffset
                } else {
                    return BroCatliResult::BrotliFileNotCraftedForConcatenation;
                };
                let mut bytes_so_far = 0u64;
                for index in 0..usize::from(new_stream_pending.num_bytes_read) {
                    bytes_so_far |=
                        u64::from(new_stream_pending.bytes_so_far[index]) << (index * 8);
                }
                bytes_so_far >>= window_offset; // mask out the window size
                bytes_so_far &= (1u64 << (varlen_offset - window_offset)) - 1;
                let var_len_bytes = (((varlen_offset - window_offset) + 7) / 8);
                for byte_index in 0..var_len_bytes {
                    let cur_byte = (bytes_so_far >> (byte_index * 8));
                    realigned_header[byte_index] |=
                        ((cur_byte & ((1 << (8 - self.last_byte_bit_offset)) - 1))
                            << self.last_byte_bit_offset) as u8;
                    realigned_header[byte_index + 1] =
                        (cur_byte >> (8 - self.last_byte_bit_offset)) as u8;
                }
                let whole_byte_destination =
                    ((usize::from(self.last_byte_bit_offset) + varlen_offset - window_offset) + 7)
                        / 8;
                let whole_byte_source = (varlen_offset + 7) / 8;
                let num_whole_bytes_to_copy =
                    usize::from(new_stream_pending.num_bytes_read) - whole_byte_source;
                for aligned_index in 0..num_whole_bytes_to_copy {
                    realigned_header[whole_byte_destination + aligned_index] =
                        new_stream_pending.bytes_so_far[whole_byte_source + aligned_index];
                }
                out_bytes[*out_offset] = realigned_header[0];
                self.any_bytes_emitted = true;
                *out_offset += 1;
                // subtract one since that has just been written out and we're only copying realigned_header[1..]
                new_stream_pending.num_bytes_read =
                    (whole_byte_destination + num_whole_bytes_to_copy) as u8 - 1;
                new_stream_pending.num_bytes_written = Some(0);
                new_stream_pending
                    .bytes_so_far
                    .clone_from_slice(&realigned_header[1..]);
            }
        } else {
            assert_ne!(self.window_size, 0);
        }
        let to_copy = min(
            out_bytes.len() - *out_offset,
            usize::from(
                new_stream_pending.num_bytes_read - new_stream_pending.num_bytes_written.unwrap(),
            ),
        );
        out_bytes
            .split_at_mut(*out_offset)
            .1
            .split_at_mut(to_copy)
            .0
            .clone_from_slice(
                new_stream_pending
                    .bytes_so_far
                    .split_at(usize::from(new_stream_pending.num_bytes_written.unwrap()))
                    .1
                    .split_at(to_copy)
                    .0,
            );
        *out_offset += to_copy;
        if to_copy != 0 {
            self.any_bytes_emitted = true;
        }
        new_stream_pending.num_bytes_written =
            Some((new_stream_pending.num_bytes_written.unwrap() + to_copy as u8));
        if new_stream_pending.num_bytes_written.unwrap() != new_stream_pending.num_bytes_read {
            self.new_stream_pending = Some(new_stream_pending);
            return BroCatliResult::NeedsMoreOutput;
        }
        self.new_stream_pending = None;
        self.last_byte_sanitized = false;
        self.last_byte_bit_offset = 0;
        self.last_bytes_len = 0;
        self.last_bytes = [0, 0];
        //now unwrite from the stream, since the last byte may need to be adjusted to be EOF
        *out_offset -= 1;
        self.last_bytes[0] = out_bytes[*out_offset];
        self.last_bytes_len = 1;
        BroCatliResult::Success
    }
    pub fn stream(
        &mut self,
        in_bytes: &[u8],
        in_offset: &mut usize,
        out_bytes: &mut [u8],
        out_offset: &mut usize,
    ) -> BroCatliResult {
        if let Some(mut new_stream_pending) = self.new_stream_pending {
            let flush_result = self.flush_previous_stream(out_bytes, out_offset);
            if let BroCatliResult::Success = flush_result {
                if usize::from(new_stream_pending.num_bytes_read)
                    < new_stream_pending.bytes_so_far.len()
                {
                    {
                        let dst = &mut new_stream_pending.bytes_so_far
                            [usize::from(new_stream_pending.num_bytes_read)..];
                        let to_copy = min(dst.len(), in_bytes.len() - *in_offset);
                        dst[..to_copy]
                            .clone_from_slice(in_bytes.split_at(*in_offset).1.split_at(to_copy).0);
                        *in_offset += to_copy;
                        new_stream_pending.num_bytes_read += to_copy as u8;
                    }
                    self.new_stream_pending = Some(new_stream_pending); // write back changes
                }
                if !new_stream_pending.sufficient() {
                    return BroCatliResult::NeedsMoreInput;
                }
                if out_bytes.len() == *out_offset {
                    return BroCatliResult::NeedsMoreOutput;
                }
                let shift_result = self.shift_and_check_new_stream_header(
                    new_stream_pending,
                    out_bytes,
                    out_offset,
                );
                if let BroCatliResult::Success = shift_result {
                } else {
                    return shift_result;
                }
            } else {
                return flush_result;
            }
            if *out_offset == out_bytes.len() {
                return BroCatliResult::NeedsMoreOutput; // need to be able to write at least one byte of data to make progress
            }
        }
        assert!(self.new_stream_pending.is_none()); // this should have been handled above
        if self.last_bytes_len != 2 {
            if out_bytes.len() == *out_offset {
                return BroCatliResult::NeedsMoreOutput;
            }
            if in_bytes.len() == *in_offset {
                return BroCatliResult::NeedsMoreInput;
            }
            self.last_bytes[usize::from(self.last_bytes_len)] = in_bytes[*in_offset];
            *in_offset += 1;
            self.last_bytes_len += 1;
            if self.last_bytes_len != 2 {
                if out_bytes.len() == *out_offset {
                    return BroCatliResult::NeedsMoreOutput;
                }
                if in_bytes.len() == *in_offset {
                    return BroCatliResult::NeedsMoreInput;
                }
                self.last_bytes[usize::from(self.last_bytes_len)] = in_bytes[*in_offset];
                self.last_bytes_len += 1;
                *in_offset += 1;
            }
        }
        if out_bytes.len() == *out_offset {
            return BroCatliResult::NeedsMoreOutput;
        }
        if in_bytes.len() == *in_offset {
            return BroCatliResult::NeedsMoreInput;
        }
        let mut to_copy = min(out_bytes.len() - *out_offset, in_bytes.len() - *in_offset);
        assert_ne!(to_copy, 0);
        if to_copy == 1 {
            out_bytes[*out_offset] = self.last_bytes[0];
            self.last_bytes[0] = self.last_bytes[1];
            self.last_bytes[1] = in_bytes[*in_offset];
            *in_offset += 1;
            *out_offset += 1;
            if *out_offset == out_bytes.len() {
                return BroCatliResult::NeedsMoreOutput;
            }
            return BroCatliResult::NeedsMoreInput;
        }
        out_bytes
            .split_at_mut(*out_offset)
            .1
            .split_at_mut(2)
            .0
            .clone_from_slice(&self.last_bytes[..]);
        *out_offset += 2;
        let (new_in_offset, last_two) = in_bytes
            .split_at(*in_offset)
            .1
            .split_at(to_copy)
            .0
            .split_at(to_copy - 2);
        self.last_bytes.clone_from_slice(last_two);
        *in_offset += 2; // add this after the clone since we grab the last 2 bytes, not the first
        to_copy -= 2;
        out_bytes
            .split_at_mut(*out_offset)
            .1
            .split_at_mut(to_copy)
            .0
            .clone_from_slice(new_in_offset);
        *out_offset += to_copy;
        *in_offset += to_copy;
        if *out_offset == out_bytes.len() {
            return BroCatliResult::NeedsMoreOutput;
        }
        BroCatliResult::NeedsMoreInput
    }
    fn append_eof_metablock_to_last_bytes(&mut self) {
        assert!(self.last_byte_sanitized);
        let mut last_bytes = self.last_bytes[0] as u16 | ((self.last_bytes[1] as u16) << 8);
        let bit_end = (self.last_bytes_len - 1) * 8 + self.last_byte_bit_offset;
        last_bytes |= 3 << bit_end;
        self.last_bytes[0] = last_bytes as u8;
        self.last_bytes[1] = (last_bytes >> 8) as u8;
        self.last_byte_sanitized = false;
        self.last_byte_bit_offset += 2;
        if self.last_byte_bit_offset >= 8 {
            self.last_byte_bit_offset -= 8;
            self.last_bytes_len += 1;
        }
    }
    pub fn finish(&mut self, out_bytes: &mut [u8], out_offset: &mut usize) -> BroCatliResult {
        if self.last_byte_sanitized && self.last_bytes_len != 0 {
            self.append_eof_metablock_to_last_bytes();
        }
        while self.last_bytes_len != 0 {
            if *out_offset == out_bytes.len() {
                return BroCatliResult::NeedsMoreOutput;
            }
            out_bytes[*out_offset] = self.last_bytes[0];
            *out_offset += 1;
            self.last_bytes_len -= 1;
            self.last_bytes[0] = self.last_bytes[1];
            self.any_bytes_emitted = true;
        }
        if !self.any_bytes_emitted {
            if out_bytes.len() == *out_offset {
                return BroCatliResult::NeedsMoreOutput;
            }
            self.any_bytes_emitted = true;
            out_bytes[*out_offset] = b';';
            *out_offset += 1;
        }
        BroCatliResult::Success
    }
}

#[cfg(test)]
mod test {
    use super::BroCatli;

    #[test]
    fn test_deserialization() {
        let broccoli = BroCatli {
            new_stream_pending: Some(super::NewStreamData {
                bytes_so_far: [0x33; super::NUM_STREAM_HEADER_BYTES],
                num_bytes_read: 16,
                num_bytes_written: Some(3),
            }),
            last_bytes: [0x45, 0x46],
            last_bytes_len: 1,
            last_byte_sanitized: true,
            any_bytes_emitted: false,
            last_byte_bit_offset: 7,
            window_size: 22,
        };
        let mut buffer = [0u8; 248];
        broccoli.serialize_to_buffer(&mut buffer[..]).unwrap();
        let bc = BroCatli::deserialize_from_buffer(&buffer[..]).unwrap();
        assert_eq!(broccoli.last_bytes, bc.last_bytes);
        assert_eq!(broccoli.last_bytes_len, bc.last_bytes_len);
        assert_eq!(broccoli.last_byte_sanitized, bc.last_byte_sanitized);
        assert_eq!(broccoli.last_byte_bit_offset, bc.last_byte_bit_offset);
        assert_eq!(broccoli.window_size, bc.window_size);
        assert_eq!(
            broccoli.new_stream_pending.unwrap().bytes_so_far,
            bc.new_stream_pending.unwrap().bytes_so_far
        );
        assert_eq!(
            broccoli.new_stream_pending.unwrap().num_bytes_read,
            bc.new_stream_pending.unwrap().num_bytes_read
        );
        assert_eq!(
            broccoli.new_stream_pending.unwrap().num_bytes_written,
            bc.new_stream_pending.unwrap().num_bytes_written
        );
    }
    #[test]
    fn test_deserialization_any_written() {
        let broccoli = BroCatli {
            new_stream_pending: Some(super::NewStreamData {
                bytes_so_far: [0x33; super::NUM_STREAM_HEADER_BYTES],
                num_bytes_read: 16,
                num_bytes_written: Some(3),
            }),
            last_bytes: [0x45, 0x46],
            last_bytes_len: 1,
            last_byte_sanitized: true,
            any_bytes_emitted: true,
            last_byte_bit_offset: 7,
            window_size: 22,
        };
        let mut buffer = [0u8; 248];
        broccoli.serialize_to_buffer(&mut buffer[..]).unwrap();
        let bc = BroCatli::deserialize_from_buffer(&buffer[..]).unwrap();
        assert_eq!(broccoli.last_bytes, bc.last_bytes);
        assert_eq!(broccoli.last_bytes_len, bc.last_bytes_len);
        assert_eq!(broccoli.last_byte_sanitized, bc.last_byte_sanitized);
        assert_eq!(broccoli.last_byte_bit_offset, bc.last_byte_bit_offset);
        assert_eq!(broccoli.window_size, bc.window_size);
        assert_eq!(
            broccoli.new_stream_pending.unwrap().bytes_so_far,
            bc.new_stream_pending.unwrap().bytes_so_far
        );
        assert_eq!(
            broccoli.new_stream_pending.unwrap().num_bytes_read,
            bc.new_stream_pending.unwrap().num_bytes_read
        );
        assert_eq!(
            broccoli.new_stream_pending.unwrap().num_bytes_written,
            bc.new_stream_pending.unwrap().num_bytes_written
        );
    }
    #[test]
    fn test_serialization() {
        let mut buffer = [0u8; 248];
        let mut broccoli = BroCatli::deserialize_from_buffer(&buffer).unwrap();
        let mut buffer2 = [0u8; 248];
        broccoli.serialize_to_buffer(&mut buffer2[..]).unwrap();
        assert_eq!(&buffer[..], &buffer2[..]);
        for (index, item) in buffer.iter_mut().enumerate() {
            *item = index as u8;
        }
        broccoli = BroCatli::deserialize_from_buffer(&buffer).unwrap();
        broccoli.serialize_to_buffer(&mut buffer2[..]).unwrap();
        broccoli = BroCatli::deserialize_from_buffer(&buffer2).unwrap();
        for (_index, item) in buffer.iter_mut().enumerate() {
            *item = 0;
        }
        broccoli.serialize_to_buffer(&mut buffer[..]).unwrap();
        assert_eq!(&buffer[..], &buffer2[..]);
        for (index, item) in buffer.iter_mut().enumerate() {
            *item = 0xff ^ index as u8;
        }
        broccoli = BroCatli::deserialize_from_buffer(&buffer).unwrap();
        broccoli.serialize_to_buffer(&mut buffer2[..]).unwrap();
        broccoli = BroCatli::deserialize_from_buffer(&buffer2).unwrap();
        for (_index, item) in buffer.iter_mut().enumerate() {
            *item = 0;
        }
        broccoli.serialize_to_buffer(&mut buffer[..]).unwrap();
        assert_eq!(&buffer[..], &buffer2[..]);
    }
    #[test]
    fn test_cat_empty_stream() {
        let empty_catable = [b';'];
        let mut bcat = super::BroCatli::default();
        let mut in_offset = 0usize;
        let mut out_bytes = [0u8; 32];
        let mut out_offset = 0usize;
        bcat.new_brotli_file();
        let mut res = bcat.stream(
            &empty_catable[..],
            &mut in_offset,
            &mut out_bytes[..],
            &mut out_offset,
        );
        assert_eq!(res, super::BroCatliResult::NeedsMoreInput);
        bcat.new_brotli_file();
        in_offset = 0;
        res = bcat.stream(
            &empty_catable[..],
            &mut in_offset,
            &mut out_bytes[..],
            &mut out_offset,
        );
        assert_eq!(res, super::BroCatliResult::NeedsMoreInput);
        res = bcat.finish(&mut out_bytes[..], &mut out_offset);
        assert_eq!(res, super::BroCatliResult::Success);
        assert_ne!(out_offset, 0);
        assert_eq!(&out_bytes[..out_offset], &[b';']);
    }
}
