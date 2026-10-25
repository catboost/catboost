use crate::bit;
use crate::huffman;
use crate::huffman::Builder;
use crate::lz77;
#[cfg(feature = "no_std")]
use core::{cmp, iter, ops::Range};
#[cfg(feature = "no_std")]
use core2::io;
#[cfg(not(feature = "no_std"))]
use std::{cmp, io, iter, ops::Range};

const FIXED_LITERAL_OR_LENGTH_CODE_TABLE: [(u8, Range<u16>, u16); 4] = [
    (8, 000..144, 0b0_0011_0000),
    (9, 144..256, 0b1_1001_0000),
    (7, 256..280, 0b0_0000_0000),
    (8, 280..288, 0b0_1100_0000),
];

const BITWIDTH_CODE_ORDER: [usize; 19] = [
    16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
];

const END_OF_BLOCK: u16 = 256;

const LENGTH_TABLE: [(u16, u8); 29] = [
    (3, 0),
    (4, 0),
    (5, 0),
    (6, 0),
    (7, 0),
    (8, 0),
    (9, 0),
    (10, 0),
    (11, 1),
    (13, 1),
    (15, 1),
    (17, 1),
    (19, 2),
    (23, 2),
    (27, 2),
    (31, 2),
    (35, 3),
    (43, 3),
    (51, 3),
    (59, 3),
    (67, 4),
    (83, 4),
    (99, 4),
    (115, 4),
    (131, 5),
    (163, 5),
    (195, 5),
    (227, 5),
    (258, 0),
];

const MAX_DISTANCE_CODE_COUNT: usize = 30;

const DISTANCE_TABLE: [(u16, u8); 30] = [
    (1, 0),
    (2, 0),
    (3, 0),
    (4, 0),
    (5, 1),
    (7, 1),
    (9, 2),
    (13, 2),
    (17, 3),
    (25, 3),
    (33, 4),
    (49, 4),
    (65, 5),
    (97, 5),
    (129, 6),
    (193, 6),
    (257, 7),
    (385, 7),
    (513, 8),
    (769, 8),
    (1025, 9),
    (1537, 9),
    (2049, 10),
    (3073, 10),
    (4097, 11),
    (6145, 11),
    (8193, 12),
    (12_289, 12),
    (16_385, 13),
    (24_577, 13),
];

#[derive(Debug, PartialEq, Eq)]
pub enum Symbol {
    EndOfBlock,
    Code(lz77::Code),
}
impl Symbol {
    pub fn code(&self) -> u16 {
        match *self {
            Symbol::Code(lz77::Code::Literal(b)) => u16::from(b),
            Symbol::EndOfBlock => 256,
            Symbol::Code(lz77::Code::Pointer { length, .. }) => match length {
                3..=10 => 257 + length - 3,
                11..=18 => 265 + (length - 11) / 2,
                19..=34 => 269 + (length - 19) / 4,
                35..=66 => 273 + (length - 35) / 8,
                67..=130 => 277 + (length - 67) / 16,
                131..=257 => 281 + (length - 131) / 32,
                258 => 285,
                _ => unreachable!(),
            },
        }
    }
    pub fn extra_lengh(&self) -> Option<(u8, u16)> {
        if let Symbol::Code(lz77::Code::Pointer { length, .. }) = *self {
            match length {
                3..=10 | 258 => None,
                11..=18 => Some((1, (length - 11) % 2)),
                19..=34 => Some((2, (length - 19) % 4)),
                35..=66 => Some((3, (length - 35) % 8)),
                67..=130 => Some((4, (length - 67) % 16)),
                131..=257 => Some((5, (length - 131) % 32)),
                _ => unreachable!(),
            }
        } else {
            None
        }
    }
    pub fn distance(&self) -> Option<(u8, u8, u16)> {
        if let Symbol::Code(lz77::Code::Pointer {
            backward_distance: distance,
            ..
        }) = *self
        {
            if distance <= 4 {
                Some((distance as u8 - 1, 0, 0))
            } else {
                let mut extra_bits = 1;
                let mut code = 4;
                let mut base = 4;
                while base * 2 < distance {
                    extra_bits += 1;
                    code += 2;
                    base *= 2;
                }
                let half = base / 2;
                let delta = distance - base - 1;
                if distance <= base + half {
                    Some((code, extra_bits, delta % half))
                } else {
                    Some((code + 1, extra_bits, delta % half))
                }
            }
        } else {
            None
        }
    }
}
impl From<lz77::Code> for Symbol {
    fn from(code: lz77::Code) -> Self {
        Symbol::Code(code)
    }
}

#[derive(Debug)]
pub struct Encoder {
    literal: huffman::Encoder,
    distance: huffman::Encoder,
}
impl Encoder {
    pub fn encode<W>(&self, writer: &mut bit::BitWriter<W>, symbol: &Symbol) -> io::Result<()>
    where
        W: io::Write,
    {
        self.literal.encode(writer, symbol.code())?;
        if let Some((bits, extra)) = symbol.extra_lengh() {
            writer.write_bits(bits, extra)?;
        }
        if let Some((code, bits, extra)) = symbol.distance() {
            self.distance.encode(writer, u16::from(code))?;
            if bits > 0 {
                writer.write_bits(bits, extra)?;
            }
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct Decoder {
    literal: huffman::Decoder,
    distance: huffman::Decoder,
}
impl Decoder {
    #[inline(always)]
    pub fn decode_unchecked<R>(&self, reader: &mut bit::BitReader<R>) -> Symbol
    where
        R: io::Read,
    {
        let mut symbol = self.decode_literal_or_length(reader);
        if let Symbol::Code(lz77::Code::Pointer {
            ref mut backward_distance,
            ..
        }) = symbol
        {
            *backward_distance = self.decode_distance(reader);
        }
        symbol
    }
    #[inline(always)]
    fn decode_literal_or_length<R>(&self, reader: &mut bit::BitReader<R>) -> Symbol
    where
        R: io::Read,
    {
        let decoded = self.literal.decode_unchecked(reader);
        match decoded {
            0..=255 => Symbol::Code(lz77::Code::Literal(decoded as u8)),
            256 => Symbol::EndOfBlock,
            286 | 287 => {
                #[cfg(not(feature = "no_std"))]
                let message = format!("The value {decoded} must not occur in compressed data");
                #[cfg(feature = "no_std")]
                let message = "The value(s) [286, 287] must not occur in compressed data";
                reader.set_last_error(io::Error::new(io::ErrorKind::InvalidData, message));
                Symbol::EndOfBlock // dummy value
            }
            length_code => {
                let (base, extra_bits) = LENGTH_TABLE[length_code as usize - 257];
                let extra = reader.read_bits_unchecked(extra_bits);
                Symbol::Code(lz77::Code::Pointer {
                    length: base + extra,
                    backward_distance: 0,
                })
            }
        }
    }
    #[inline(always)]
    fn decode_distance<R>(&self, reader: &mut bit::BitReader<R>) -> u16
    where
        R: io::Read,
    {
        let decoded = self.distance.decode_unchecked(reader) as usize;
        let (base, extra_bits) = DISTANCE_TABLE[decoded];
        let extra = reader.read_bits_unchecked(extra_bits);
        base + extra
    }
}

pub trait HuffmanCodec {
    fn build(&self, symbols: &[Symbol]) -> io::Result<Encoder>;
    fn save<W>(&self, writer: &mut bit::BitWriter<W>, codec: &Encoder) -> io::Result<()>
    where
        W: io::Write;
    fn load<R>(&self, reader: &mut bit::BitReader<R>) -> io::Result<Decoder>
    where
        R: io::Read;
}

#[derive(Debug)]
pub struct FixedHuffmanCodec;
impl HuffmanCodec for FixedHuffmanCodec {
    #[allow(unused_variables)]
    fn build(&self, symbols: &[Symbol]) -> io::Result<Encoder> {
        let mut literal_builder = huffman::EncoderBuilder::new(288);
        for &(bitwidth, ref symbols, code_base) in &FIXED_LITERAL_OR_LENGTH_CODE_TABLE {
            for (code, symbol) in symbols
                .clone()
                .enumerate()
                .map(|(i, s)| (code_base + i as u16, s))
            {
                literal_builder.set_mapping(symbol, huffman::Code::new(bitwidth, code))?;
            }
        }

        let mut distance_builder = huffman::EncoderBuilder::new(30);
        for i in 0..30 {
            distance_builder.set_mapping(i, huffman::Code::new(5, i))?;
        }

        Ok(Encoder {
            literal: literal_builder.finish(),
            distance: distance_builder.finish(),
        })
    }
    #[allow(unused_variables)]
    fn save<W>(&self, writer: &mut bit::BitWriter<W>, codec: &Encoder) -> io::Result<()>
    where
        W: io::Write,
    {
        Ok(())
    }
    #[allow(unused_variables)]
    fn load<R>(&self, reader: &mut bit::BitReader<R>) -> io::Result<Decoder>
    where
        R: io::Read,
    {
        let mut literal_builder = huffman::DecoderBuilder::new(9, None, Some(END_OF_BLOCK));
        for &(bitwidth, ref symbols, code_base) in &FIXED_LITERAL_OR_LENGTH_CODE_TABLE {
            for (code, symbol) in symbols
                .clone()
                .enumerate()
                .map(|(i, s)| (code_base + i as u16, s))
            {
                literal_builder.set_mapping(symbol, huffman::Code::new(bitwidth, code))?;
            }
        }

        let mut distance_builder =
            huffman::DecoderBuilder::new(5, literal_builder.safely_peek_bitwidth(), None);
        for i in 0..30 {
            distance_builder.set_mapping(i, huffman::Code::new(5, i))?;
        }

        Ok(Decoder {
            literal: literal_builder.finish(),
            distance: distance_builder.finish(),
        })
    }
}

#[derive(Debug)]
pub struct DynamicHuffmanCodec;
impl HuffmanCodec for DynamicHuffmanCodec {
    fn build(&self, symbols: &[Symbol]) -> io::Result<Encoder> {
        let mut literal_counts = [0; 286];
        let mut distance_counts = [0; 30];
        let mut empty_distance_table = true;
        for s in symbols {
            literal_counts[s.code() as usize] += 1;
            if let Some((d, _, _)) = s.distance() {
                empty_distance_table = false;
                distance_counts[d as usize] += 1;
            }
        }
        if empty_distance_table {
            // Sets a dummy value because an empty distance table causes decoding error on Windows.
            //
            // See https://github.com/sile/libflate/issues/23 for more details.
            distance_counts[0] = 1;
        }
        Ok(Encoder {
            literal: huffman::EncoderBuilder::from_frequencies(&literal_counts, 15)?,
            distance: huffman::EncoderBuilder::from_frequencies(&distance_counts, 15)?,
        })
    }
    fn save<W>(&self, writer: &mut bit::BitWriter<W>, codec: &Encoder) -> io::Result<()>
    where
        W: io::Write,
    {
        let literal_code_count = cmp::max(257, codec.literal.used_max_symbol().unwrap_or(0) + 1);
        let distance_code_count = cmp::max(1, codec.distance.used_max_symbol().unwrap_or(0) + 1);
        let codes = build_bitwidth_codes(codec, literal_code_count, distance_code_count);

        let mut code_counts = [0; 19];
        for x in &codes {
            code_counts[x.0 as usize] += 1;
        }
        let bitwidth_encoder = huffman::EncoderBuilder::from_frequencies(&code_counts, 7)?;

        let bitwidth_code_count = cmp::max(
            4,
            BITWIDTH_CODE_ORDER
                .iter()
                .rev()
                .position(|&i| code_counts[i] != 0 && bitwidth_encoder.lookup(i as u16).width > 0)
                .map_or(0, |trailing_zeros| 19 - trailing_zeros),
        ) as u16;
        writer.write_bits(5, literal_code_count - 257)?;
        writer.write_bits(5, distance_code_count - 1)?;
        writer.write_bits(4, bitwidth_code_count - 4)?;
        for &i in BITWIDTH_CODE_ORDER
            .iter()
            .take(bitwidth_code_count as usize)
        {
            let width = if code_counts[i] == 0 {
                0
            } else {
                u16::from(bitwidth_encoder.lookup(i as u16).width)
            };
            writer.write_bits(3, width)?;
        }
        for &(code, bits, extra) in &codes {
            bitwidth_encoder.encode(writer, u16::from(code))?;
            if bits > 0 {
                writer.write_bits(bits, u16::from(extra))?;
            }
        }
        Ok(())
    }
    fn load<R>(&self, reader: &mut bit::BitReader<R>) -> io::Result<Decoder>
    where
        R: io::Read,
    {
        let literal_code_count = reader.read_bits(5)? + 257;
        let distance_code_count = reader.read_bits(5)? + 1;
        let bitwidth_code_count = reader.read_bits(4)? + 4;

        if distance_code_count as usize > MAX_DISTANCE_CODE_COUNT {
            #[cfg(not(feature = "no_std"))]
            let message = format!(
                "The value of HDIST is too big: max={MAX_DISTANCE_CODE_COUNT}, actual={distance_code_count}"
            );
            #[cfg(feature = "no_std")]
            let message = "The value of HDIST is too big: max={MAX_DISTANCE_CODE_COUNT}";
            return Err(io::Error::new(io::ErrorKind::InvalidData, message));
        }

        let mut bitwidth_code_bitwidthes = [0; 19];
        for &i in BITWIDTH_CODE_ORDER
            .iter()
            .take(bitwidth_code_count as usize)
        {
            bitwidth_code_bitwidthes[i] = reader.read_bits(3)? as u8;
        }
        let bitwidth_decoder =
            huffman::DecoderBuilder::from_bitwidthes(&bitwidth_code_bitwidthes, Some(1), None)?;

        let mut literal_code_bitwidthes = Vec::with_capacity(literal_code_count as usize);
        while literal_code_bitwidthes.len() < literal_code_count as usize {
            let c = bitwidth_decoder.decode(reader)?;
            let last = literal_code_bitwidthes.last().cloned();
            literal_code_bitwidthes.extend(load_bitwidthes(reader, c, last)?);
        }

        let mut distance_code_bitwidthes = literal_code_bitwidthes
            .drain(literal_code_count as usize..)
            .collect::<Vec<_>>();
        while distance_code_bitwidthes.len() < distance_code_count as usize {
            let c = bitwidth_decoder.decode(reader)?;
            let last = distance_code_bitwidthes
                .last()
                .cloned()
                .or_else(|| literal_code_bitwidthes.last().cloned());
            distance_code_bitwidthes.extend(load_bitwidthes(reader, c, last)?);
        }
        if distance_code_bitwidthes.len() > distance_code_count as usize {
            #[cfg(not(feature = "no_std"))]
            let message = format!(
                "The length of `distance_code_bitwidthes` is too large: actual={}, expected={}",
                distance_code_bitwidthes.len(),
                distance_code_count
            );
            #[cfg(feature = "no_std")]
            let message = "The length of `distance_code_bitwidthes` is too large";
            return Err(io::Error::new(io::ErrorKind::InvalidData, message));
        }

        let literal = huffman::DecoderBuilder::from_bitwidthes(
            &literal_code_bitwidthes,
            None,
            Some(END_OF_BLOCK),
        )?;
        let distance = huffman::DecoderBuilder::from_bitwidthes(
            &distance_code_bitwidthes,
            Some(literal.safely_peek_bitwidth()),
            None,
        )?;
        Ok(Decoder { literal, distance })
    }
}

fn load_bitwidthes<R>(
    reader: &mut bit::BitReader<R>,
    code: u16,
    last: Option<u8>,
) -> io::Result<Box<dyn Iterator<Item = u8>>>
where
    R: io::Read,
{
    Ok(match code {
        0..=15 => Box::new(iter::once(code as u8)),
        16 => {
            let count = reader.read_bits(2)? + 3;
            let last = last.ok_or_else(|| invalid_data_error!("No preceding value"))?;
            Box::new(iter::repeat(last).take(count as usize))
        }
        17 => {
            let zeros = reader.read_bits(3)? + 3;
            Box::new(iter::repeat(0).take(zeros as usize))
        }
        18 => {
            let zeros = reader.read_bits(7)? + 11;
            Box::new(iter::repeat(0).take(zeros as usize))
        }
        _ => unreachable!(),
    })
}

fn build_bitwidth_codes(
    codec: &Encoder,
    literal_code_count: u16,
    distance_code_count: u16,
) -> Vec<(u8, u8, u8)> {
    struct RunLength {
        value: u8,
        count: usize,
    }

    let mut run_lens: Vec<RunLength> = Vec::new();
    for &(e, size) in &[
        (&codec.literal, literal_code_count),
        (&codec.distance, distance_code_count),
    ] {
        for (i, c) in (0..size).map(|x| e.lookup(x).width).enumerate() {
            if i > 0 && run_lens.last().map_or(false, |s| s.value == c) {
                run_lens.last_mut().unwrap().count += 1;
            } else {
                run_lens.push(RunLength { value: c, count: 1 })
            }
        }
    }

    let mut codes: Vec<(u8, u8, u8)> = Vec::new();
    for r in run_lens {
        if r.value == 0 {
            let mut c = r.count;
            while c >= 11 {
                let n = cmp::min(138, c) as u8;
                codes.push((18, 7, n - 11));
                c -= n as usize;
            }
            if c >= 3 {
                codes.push((17, 3, c as u8 - 3));
                c = 0;
            }
            for _ in 0..c {
                codes.push((0, 0, 0));
            }
        } else {
            codes.push((r.value, 0, 0));
            let mut c = r.count - 1;
            while c >= 3 {
                let n = cmp::min(6, c) as u8;
                codes.push((16, 2, n - 3));
                c -= n as usize;
            }
            for _ in 0..c {
                codes.push((r.value, 0, 0));
            }
        }
    }
    codes
}
