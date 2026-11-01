pub(crate) const fn crc8(poly: u8, reflect: bool, mut byte: u8) -> u8 {
    if reflect {
        byte = byte.reverse_bits()
    };
    let mut value = byte;
    let mut i = 0;
    while i < 8 {
        value = (value << 1) ^ ((value >> 7) * poly);
        i += 1;
    }
    if reflect {
        value = value.reverse_bits()
    }
    value
}

pub(crate) const fn crc16(poly: u16, reflect: bool, mut byte: u8) -> u16 {
    if reflect {
        byte = byte.reverse_bits()
    };
    let mut value = (byte as u16) << 8;
    let mut i = 0;
    while i < 8 {
        value = (value << 1) ^ ((value >> 15) * poly);
        i += 1;
    }
    if reflect {
        value = value.reverse_bits()
    }
    value
}

pub(crate) const fn crc32(poly: u32, reflect: bool, mut byte: u8) -> u32 {
    if reflect {
        byte = byte.reverse_bits()
    };
    let mut value = (byte as u32) << 24;
    let mut i = 0;
    while i < 8 {
        value = (value << 1) ^ ((value >> 31) * poly);
        i += 1;
    }
    if reflect {
        value = value.reverse_bits()
    }
    value
}

pub(crate) const fn crc64(poly: u64, reflect: bool, mut byte: u8) -> u64 {
    if reflect {
        byte = byte.reverse_bits()
    };
    let mut value = (byte as u64) << 56;
    let mut i = 0;
    while i < 8 {
        value = (value << 1) ^ ((value >> 63) * poly);
        i += 1;
    }
    if reflect {
        value = value.reverse_bits()
    }
    value
}
