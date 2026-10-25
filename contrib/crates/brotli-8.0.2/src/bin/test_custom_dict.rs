#![cfg(test)]
#![allow(non_upper_case_globals)]
#![allow(dead_code)]

extern crate brotli_decompressor;
extern crate core;

use std::io::{Read, Write};

use super::brotli::concat::{BroCatli, BroCatliResult};
use super::brotli::enc::BrotliEncoderParams;
use super::integration_tests::UnlimitedBuffer;
use super::Rebox;

static RANDOM_THEN_UNICODE: &[u8] = include_bytes!("../../testdata/random_then_unicode");
static ALICE: &[u8] = include_bytes!("../../testdata/alice29.txt");

#[test]
fn test_custom_dict_minimal() {
    let mut raw = UnlimitedBuffer::new("\012345656789abcde".as_bytes());
    let mut params = BrotliEncoderParams::default();
    params.quality = 10;
    let mut br = UnlimitedBuffer::new(&[]);
    let mut rt = UnlimitedBuffer::new(&[]);
    let dict = "123456789abcde".as_bytes();
    super::compress(&mut raw, &mut br, 4096, &params, dict, 1).unwrap();
    raw.reset_read();
    eprintln!("Compressed: {:?}", &br);
    let mut vec = Vec::<u8>::new();
    vec.extend(dict);
    super::decompress(&mut br, &mut rt, 4096, Rebox::from(vec)).unwrap();
    assert_eq!(rt.data(), raw.data());
}

#[test]
fn test_custom_dict_large() {
    let mut params = BrotliEncoderParams::default();
    params.quality = 11;
    let mut br = UnlimitedBuffer::new(&[]);
    let mut rt = UnlimitedBuffer::new(&[]);
    let mut dict = [0u8;256];
    for (index, val) in dict[..].iter_mut().enumerate()
    {
        *val = index as u8;
    }
    let mut data_source = [0u8; 10823];
    let mut counter = 0usize;
    for (index, val) in data_source[..].iter_mut().enumerate()
    {
        
        *val = counter as u8;
        counter += 1;
        if counter * 10 > index 
        {
            counter -= index / 11;
        }
    }
    eprintln!("Uncompressed: {:?}", &data_source);
    let mut raw = UnlimitedBuffer::new(&data_source);
    super::compress(&mut raw, &mut br, 4096, &params, &dict, 1).unwrap();
    raw.reset_read();
    eprintln!("Compressed: {:?}", &br);
    std::fs::File::create("/tmp/compressed.br").expect("Failed").write_all(&br.data).expect("Failed to write compressed");
    std::fs::File::create("/tmp/compressed.dict").expect("Failed").write_all(&dict).expect("Failed to write dict");
    std::fs::File::create("/tmp/compressed.txt").expect("Failed").write_all(&data_source).expect("Failed to write data source");
    let mut vec = Vec::<u8>::new();
    vec.extend(dict);
    super::decompress(&mut br, &mut rt, 4096, Rebox::from(vec)).unwrap();
    assert_eq!(rt.data(), raw.data());
}

#[test]
fn test_custom_dict_medium() {
    let mut params = BrotliEncoderParams::default();
    params.quality = 11;
    let mut br = UnlimitedBuffer::new(&[]);
    let mut rt = UnlimitedBuffer::new(&[]);
    let mut dict = [0u8;256];
    for (index, val) in dict[..].iter_mut().enumerate()
    {
        *val = index as u8;
    }
    let mut data_source = [0u8; 323];
    let mut counter = 0usize;
    for (index, val) in data_source[..].iter_mut().enumerate()
    {
        
        *val = counter as u8;
        counter += 1;
        if counter * 10 > index 
        {
            counter -= index / 11;
            counter = counter.wrapping_sub(43);
            counter = usize::from(counter as u8);
        }
    }
    eprintln!("Uncompressed: {:?}", &data_source[72..]);
    let new_data_source = [148, 100, 52, 4, 5, 6, 7, 214, 165, 116, 67, 18, 225, 176, 127, 78, 29, 235, 185, 135, 85, 35, 241, 191, 141, 91, 41, 247, 196, 145, 94, 43, 248, 197, 146, 95, 44, 249, 198, 146, 94, 42, 246, 194, 142, 90, 38, 242, 190, 138, 85, 32, 235, 182, 129, 76, 23, 226, 173, 120, 67, 13, 215, 161, 107, 53, 255, 201, 147, 93, 39, 241, 186, 131, 76, 21, 222, 167, 112, 57, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 215, 158, 101, 44, 243, 186, 129, 72, 15, 16, 17, 215, 157, 99, 41, 239, 181, 123, 65, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 215, 155, 95, 35, 231, 171, 111, 51, 247, 187, 127, 66, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 216, 154, 92, 29, 222, 159, 96, 33, 226, 163, 100, 37, 230, 167, 103, 39, 231, 167, 103, 39, 231, 167, 103, 39, 231, 166, 101, 36, 227, 162, 97, 32, 223, 158, 93, 28, 218, 152, 86, 20, 21, 22, 23, 24, 25, 26, 27, 216, 149, 82, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 217, 149, 81, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 217, 147, 77, 7, 8, 9, 10, 11];
    let mut raw = UnlimitedBuffer::new(&new_data_source);
    super::compress(&mut raw, &mut br, 4096, &params, &dict, 1).unwrap();
    raw.reset_read();
    eprintln!("Compressed: {:?}", &br);
    let mut vec = Vec::<u8>::new();
    vec.extend(dict);
    super::decompress(&mut br, &mut rt, 4096, Rebox::from(vec)).unwrap();
    assert_eq!(rt.data(), raw.data());
}


#[test]
fn test_custom_dict_alice() {
    let mut raw = UnlimitedBuffer::new(ALICE);
    let mut params = BrotliEncoderParams::default();
    params.quality = 11;
    let mut br = UnlimitedBuffer::new(&[]);
    let mut rt = UnlimitedBuffer::new(&[]);
    let dict = &ALICE[12515..23411];
    super::compress(&mut raw, &mut br, 4096, &params, dict, 1).unwrap();
    raw.reset_read();
    eprintln!("Dict {:?}",   dict);
    eprintln!("Compressed: {:?}", &br);
    let mut vec = Vec::<u8>::new();
    vec.extend(dict);
    super::decompress(&mut br, &mut rt, 4096, Rebox::from(vec)).unwrap();
    assert_eq!(rt.data(), raw.data());
    if br.data().len() != 43860 { // This is for 32 bit opts
        assert_eq!(br.data().len(), 43836);
    }
}

#[test]
fn test_custom_dict_alice_9_5() {
    let mut raw = UnlimitedBuffer::new(ALICE);
    let mut params = BrotliEncoderParams::default();
    params.quality = 11;
    params.q9_5 = true;
    let mut br = UnlimitedBuffer::new(&[]);
    let mut rt = UnlimitedBuffer::new(&[]);
    let dict = &ALICE[12515..23411];
    super::compress(&mut raw, &mut br, 4096, &params, dict, 1).unwrap();
    raw.reset_read();
    eprintln!("Dict {:?}",   dict);
    eprintln!("Compressed: {:?}", &br);
    let mut vec = Vec::<u8>::new();
    vec.extend(dict);
    super::decompress(&mut br, &mut rt, 4096, Rebox::from(vec)).unwrap();
    assert_eq!(rt.data(), raw.data());
    if br.data().len() != 45710 {
        assert_eq!(br.data().len(), 45698);
    }
}

#[test]
fn test_custom_wrong_dict_fails() {
    let mut raw = UnlimitedBuffer::new(ALICE);
    let mut params = BrotliEncoderParams::default();
    params.quality = 10;
    let mut br = UnlimitedBuffer::new(&[]);
    let mut rt = UnlimitedBuffer::new(&[]);
    let dict = &ALICE[12515..19515];
    super::compress(&mut raw, &mut br, 4096, &params, dict, 1).unwrap();
    raw.reset_read();
    let mut vec = Vec::<u8>::new();
    vec.extend(&dict[1..]); // slightly offset dictionary to be wrong, and ensure the dict was being used above
    match super::decompress(&mut br, &mut rt, 4096, Rebox::from(vec)) {
        Ok(_) => panic!("Decompression should have failed"),
        Err(_) => {}
    }
    if rt.data() == raw.data() {
        panic!("they should be unequal");
    }
}

#[test]
fn test_custom_wrong_dict_fails_but_doesnt_disrupt_compression_strategy() {
    let mut raw = UnlimitedBuffer::new(ALICE);
    let mut params = BrotliEncoderParams::default();
    params.quality = 6;
    let mut br = UnlimitedBuffer::new(&[]);
    let mut rt = UnlimitedBuffer::new(&[]);
    let dict = &ALICE[12515..19515];
    super::compress(&mut raw, &mut br, 4096, &params, dict, 1).unwrap();
    raw.reset_read();
    let mut vec = Vec::<u8>::new();
    vec.extend(&dict[1..]); // slightly offset dictionary to be wrong, and ensure the dict was being used above
    super::decompress(&mut br, &mut rt, 4096, Rebox::from(vec)).unwrap();
    if rt.data() == raw.data() {
        panic!("they should be unequal");
    }
}

#[ignore = "LZ77 mode no longer is compatible with manual file splitting since it sets context to 0 instead of leaving it to the end of the dict"]
#[test]
fn test_custom_dict_for_multithreading() {
    let mut raws = [
        UnlimitedBuffer::new(&ALICE[..ALICE.len() / 3]),
        UnlimitedBuffer::new(&ALICE[ALICE.len() / 3..2 * ALICE.len() / 3]),
        UnlimitedBuffer::new(&ALICE[2 * ALICE.len() / 3..]),
    ];
    let mut params = BrotliEncoderParams::default();
    params.quality = 10;
    params.appendable = true;
    let mut brs = [
        UnlimitedBuffer::new(&[]),
        UnlimitedBuffer::new(&[]),
        UnlimitedBuffer::new(&[]),
    ];
    let mut rts = [
        UnlimitedBuffer::new(&[]),
        UnlimitedBuffer::new(&[]),
        UnlimitedBuffer::new(&[]),
    ];
    let dicts = [
        &ALICE[..0],
        &ALICE[..ALICE.len() / 3],
        &ALICE[..2 * ALICE.len() / 3],
    ];
    for (raw, (br, (rt, dict))) in raws
        .iter_mut()
        .zip(brs.iter_mut().zip(rts.iter_mut().zip(dicts.iter())))
    {
        super::compress(raw, br, 4096, &params, &[], 1).unwrap();
        raw.reset_read();
        let mut vec = Vec::<u8>::new();
        vec.extend(*dict);
        super::decompress(br, rt, 4096, Rebox::from(vec)).unwrap();
        assert_eq!(rt.data(), raw.data());
        params.catable = true;
        params.use_dictionary = false;
    }
    let mut bro_cat_li = BroCatli::new();
    let mut output = UnlimitedBuffer::new(&[]);
    let mut ibuffer = [0u8; 1];
    let mut obuffer = [0u8; 1];
    let mut ooffset = 0usize;
    for brotli in brs.iter_mut() {
        brotli.reset_read();
        bro_cat_li.new_brotli_file();
        let input = brotli;
        loop {
            let mut ioffset = 0usize;
            match input.read(&mut ibuffer[..]) {
                Err(e) => panic!("{:?}", e),
                Ok(cur_read) => {
                    if cur_read == 0 {
                        break;
                    }
                    loop {
                        match bro_cat_li.stream(
                            &ibuffer[..cur_read],
                            &mut ioffset,
                            &mut obuffer[..],
                            &mut ooffset,
                        ) {
                            BroCatliResult::NeedsMoreOutput => {
                                match output.write(&obuffer[..ooffset]) {
                                    Err(why) => panic!("couldn't write: {:}", why),
                                    Ok(count) => {
                                        assert_eq!(count, ooffset);
                                    }
                                }
                                ooffset = 0;
                            }
                            BroCatliResult::NeedsMoreInput => {
                                break;
                            }
                            BroCatliResult::Success => {
                                panic!("Unexpected state: Success when streaming before finish");
                            }
                            failure => {
                                panic!("{:?}", failure);
                            }
                        }
                    }
                }
            }
        }
    }
    loop {
        match bro_cat_li.finish(&mut obuffer[..], &mut ooffset) {
            BroCatliResult::NeedsMoreOutput => {
                match output.write(&obuffer[..ooffset]) {
                    Err(why) => panic!("couldn't write\n{:}", why),
                    Ok(count) => {
                        assert_eq!(count, ooffset);
                    }
                }
                ooffset = 0;
            }
            BroCatliResult::NeedsMoreInput => {
                panic!("Unexpected EOF");
            }
            BroCatliResult::Success => {
                if ooffset != 0 {
                    match output.write(&obuffer[..ooffset]) {
                        Err(why) => panic!("couldn't write\n{:}", why),
                        Ok(count) => {
                            assert_eq!(count, ooffset);
                        }
                    }
                }
                break;
            }
            failure => {
                panic!("{}", failure as i32)
            }
        }
    }
    let mut rt = UnlimitedBuffer::new(&[]);
    output.reset_read();
    super::decompress(&mut output, &mut rt, 4096, Rebox::default()).unwrap();
    assert_eq!(rt.data(), ALICE);
    // without setting std flag: approximation make it 4 bytes bigger
    if output.data().len() != 48568 {
        assert_eq!(output.data().len(), 48563); // as opposed to 46487 with standard settings
    }
}
