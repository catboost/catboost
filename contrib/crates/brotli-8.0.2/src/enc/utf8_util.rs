use crate::enc::floatX;

fn parse_as_utf8(input: &[u8], size: usize) -> (usize, i32) {
    if (input[0] & 0x80) == 0 {
        if input[0] > 0 {
            return (1, i32::from(input[0]));
        }
    }
    if size > 1 && (input[0] & 0xe0) == 0xc0 && (input[1] & 0xc0) == 0x80 {
        let symbol = (input[0] as i32 & 0x1f) << 6 | input[1] as i32 & 0x3f;
        if symbol > 0x7f {
            return (2, symbol);
        }
    }
    if size > 2
        && (input[0] & 0xf0) == 0xe0
        && (input[1] & 0xc0) == 0x80
        && (input[2] & 0xc0) == 0x80
    {
        let symbol = (i32::from(input[0]) & 0x0f) << 12
            | (i32::from(input[1]) & 0x3f) << 6
            | i32::from(input[2]) & 0x3f;
        if symbol > 0x7ff {
            return (3, symbol);
        }
    }
    if size > 3
        && (input[0] & 0xf8) == 0xf0
        && (input[1] & 0xc0) == 0x80
        && (input[2] & 0xc0) == 0x80
        && (input[3] & 0xc0) == 0x80
    {
        let symbol = (i32::from(input[0]) & 0x07) << 18
            | (i32::from(input[1]) & 0x3f) << 12
            | (i32::from(input[2]) & 0x3f) << 6
            | i32::from(input[3]) & 0x3f;
        if symbol > 0xffff && symbol <= 0x10_ffff {
            return (4, symbol);
        }
    }

    (1, 0x11_0000 | i32::from(input[0]))
}

pub(crate) fn is_mostly_utf8(
    data: &[u8],
    pos: usize,
    mask: usize,
    length: usize,
    min_fraction: floatX,
) -> bool {
    let mut size_utf8: usize = 0;
    let mut i: usize = 0;
    while i < length {
        let (bytes_read, symbol) = parse_as_utf8(&data[(pos.wrapping_add(i) & mask)..], length - i);
        i = i.wrapping_add(bytes_read);
        if symbol < 0x11_0000 {
            size_utf8 = size_utf8.wrapping_add(bytes_read);
        }
    }
    size_utf8 as floatX > min_fraction * length as floatX
}
