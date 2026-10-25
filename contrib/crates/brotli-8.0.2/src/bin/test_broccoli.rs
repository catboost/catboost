#![cfg(test)]
#![allow(non_upper_case_globals)]
#![allow(dead_code)]

extern crate brotli_decompressor;
extern crate core;

use core::cmp::{max, min};

use brotli_decompressor::{CustomRead, CustomWrite};

use super::brotli::concat::{BroCatli, BroCatliResult};
use super::brotli::enc::BrotliEncoderParams;
use super::integration_tests::UnlimitedBuffer;
use super::Rebox;

static RANDOM_THEN_UNICODE: &[u8] = include_bytes!("../../testdata/random_then_unicode");
static ALICE: &[u8] = include_bytes!("../../testdata/alice29.txt");
static UKKONOOA: &[u8] = include_bytes!("../../testdata/ukkonooa");
static ASYOULIKE: &[u8] = include_bytes!("../../testdata/asyoulik.txt");
static BACKWARD65536: &[u8] = include_bytes!("../../testdata/backward65536");
static DICTWORD: &[u8] = include_bytes!("../../testdata/ends_with_truncated_dictionary");
static RANDOM10K: &[u8] = include_bytes!("../../testdata/random_org_10k.bin");
static RANDOMTHENUNICODE: &[u8] = include_bytes!("../../testdata/random_then_unicode");
static QUICKFOX: &[u8] = include_bytes!("../../testdata/quickfox_repeated");
static EMPTY: &[u8] = &[];

fn concat(
    files: &mut [UnlimitedBuffer],
    brotli_files: &mut [UnlimitedBuffer],
    window_override: Option<u8>,
    bs: usize,
) {
    let mut obuffer = vec![0u8; bs];
    let mut ibuffer = vec![0u8; bs];
    let mut ooffset = 0usize;
    let mut ioffset;
    let mut uboutput = UnlimitedBuffer::new(&[]);
    {
        let mut output = super::IoWriterWrapper(&mut uboutput);
        let mut bro_cat_li = match window_override {
            Some(ws) => BroCatli::new_with_window_size(ws),
            None => BroCatli::new(),
        };
        for brotli in brotli_files.iter_mut() {
            bro_cat_li.new_brotli_file();
            {
                let mut input = super::IoReaderWrapper(brotli);
                loop {
                    ioffset = 0;
                    match input.read(&mut ibuffer[..]) {
                        Err(e) => panic!("{}", e),
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
            brotli.reset_read();
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
                    panic!("{:?}", failure)
                }
            }
        }
    }
    let mut rt = UnlimitedBuffer::new(&[]);
    match super::decompress(&mut uboutput, &mut rt, 65536, Rebox::default()) {
        Ok(_) => {}
        Err(e) => panic!("Error {:?}", e),
    }
    let mut offset = 0;
    for file in files {
        assert_eq!(&rt.data()[offset..offset + file.data().len()], file.data());
        offset += file.data().len();
    }
    assert_eq!(offset, rt.data().len());
}

fn concat_many_subsets(
    files: &mut [UnlimitedBuffer],
    brotli_files: &mut [UnlimitedBuffer],
    window_override: Option<u8>,
) {
    let test_plans: [(usize, usize); 4] =
        [(brotli_files.len(), 4096 * 1024), (4, 1), (3, 3), (2, 4096)];
    for plan_bs in test_plans.iter() {
        let files_len = files.len();
        for index in 0..(brotli_files.len() - min(plan_bs.0 - 1, files_len)) {
            let file_subset = &mut files[index..min(index + plan_bs.0, files_len)];
            let brotli_subset = &mut brotli_files[index..min(index + plan_bs.0, files_len)];
            concat(file_subset, brotli_subset, window_override, plan_bs.1);
        }
    }
}

#[cfg(debug_assertions)]
fn light_debug_test(params: &mut BrotliEncoderParams) {
    params.quality = 5;
}

#[cfg(not(debug_assertions))]
fn light_debug_test(_params: &mut BrotliEncoderParams) {}

#[cfg(debug_assertions)]
fn medium_debug_test(params: &mut BrotliEncoderParams) {
    params.quality = 9;
    params.q9_5 = false;
}

#[cfg(not(debug_assertions))]
fn medium_debug_test(_params: &mut BrotliEncoderParams) {}

#[test]
#[should_panic]
fn test_appendonly_twice_fails() {
    let mut files = [
        UnlimitedBuffer::new(UKKONOOA),
        UnlimitedBuffer::new(QUICKFOX),
    ];
    let mut ufiles = [UnlimitedBuffer::new(&[]), UnlimitedBuffer::new(&[])];
    for (src, dst) in files.iter_mut().zip(ufiles.iter_mut()) {
        let mut params0 = BrotliEncoderParams::default();
        params0.appendable = true;
        super::compress(src, dst, 4096, &params0, &[], 1).unwrap();
    }
    concat(&mut files[..], &mut ufiles[..], None, 2);
}

#[test]
fn test_append_then_empty_works() {
    let mut files = [UnlimitedBuffer::new(UKKONOOA), UnlimitedBuffer::new(&[])];
    let mut ufiles = [UnlimitedBuffer::new(&[]), UnlimitedBuffer::new(&[])];
    let mut first = true;
    for (src, dst) in files.iter_mut().zip(ufiles.iter_mut()) {
        let mut params0 = BrotliEncoderParams::default();
        params0.appendable = first;
        params0.catable = !first;
        params0.use_dictionary = first;
        super::compress(src, dst, 4096, &params0, &[], 1).unwrap();
        first = false;
    }
    concat(&mut files[..], &mut ufiles[..], None, 2);
}

#[test]
fn test_append_then_cat_works() {
    let mut files = [
        UnlimitedBuffer::new(UKKONOOA),
        UnlimitedBuffer::new(QUICKFOX),
    ];
    let mut ufiles = [UnlimitedBuffer::new(&[]), UnlimitedBuffer::new(&[])];
    let mut first = true;
    for (src, dst) in files.iter_mut().zip(ufiles.iter_mut()) {
        let mut params0 = BrotliEncoderParams::default();
        params0.appendable = first;
        params0.catable = !first;
        params0.use_dictionary = first;
        super::compress(src, dst, 4096, &params0, &[], 1).unwrap();
        first = false;
    }
    concat(&mut files[..], &mut ufiles[..], None, 2);
}

#[test]
fn test_one_byte_works() {
    let mut files = [UnlimitedBuffer::new(UKKONOOA), UnlimitedBuffer::new(&[8])];
    let mut ufiles = [UnlimitedBuffer::new(&[]), UnlimitedBuffer::new(&[])];
    let mut first = true;
    for (src, dst) in files.iter_mut().zip(ufiles.iter_mut()) {
        let mut params0 = BrotliEncoderParams::default();
        params0.appendable = first;
        params0.catable = !first;
        params0.use_dictionary = first;
        super::compress(src, dst, 4096, &params0, &[], 1).unwrap();
        first = false;
    }
    concat(&mut files[..], &mut ufiles[..], None, 2);
}

#[test]
fn test_one_byte_before_works() {
    let mut files = [UnlimitedBuffer::new(&[8]), UnlimitedBuffer::new(UKKONOOA)];
    let mut ufiles = [UnlimitedBuffer::new(&[]), UnlimitedBuffer::new(&[])];
    let mut first = true;
    for (src, dst) in files.iter_mut().zip(ufiles.iter_mut()) {
        let mut params0 = BrotliEncoderParams::default();
        params0.appendable = first;
        params0.catable = !first;
        params0.use_dictionary = first;
        super::compress(src, dst, 4096, &params0, &[], 1).unwrap();
        first = false;
    }
    concat(&mut files[..], &mut ufiles[..], None, 2);
}

#[test]
fn test_two_byte_works() {
    let mut files = [
        UnlimitedBuffer::new(UKKONOOA),
        UnlimitedBuffer::new(&[8, 9]),
    ];
    let mut ufiles = [UnlimitedBuffer::new(&[]), UnlimitedBuffer::new(&[])];
    let mut first = true;
    for (src, dst) in files.iter_mut().zip(ufiles.iter_mut()) {
        let mut params0 = BrotliEncoderParams::default();
        params0.appendable = first;
        params0.catable = !first;
        params0.use_dictionary = first;
        super::compress(src, dst, 4096, &params0, &[], 1).unwrap();
        first = false;
    }
    concat(&mut files[..], &mut ufiles[..], None, 2);
}

#[test]
fn test_two_byte_before_works() {
    let mut files = [
        UnlimitedBuffer::new(&[8, 9]),
        UnlimitedBuffer::new(UKKONOOA),
    ];
    let mut ufiles = [UnlimitedBuffer::new(&[]), UnlimitedBuffer::new(&[])];
    let mut first = true;
    for (src, dst) in files.iter_mut().zip(ufiles.iter_mut()) {
        let mut params0 = BrotliEncoderParams::default();
        params0.appendable = first;
        params0.catable = !first;
        params0.use_dictionary = first;
        super::compress(src, dst, 4096, &params0, &[], 1).unwrap();
        first = false;
    }
    concat(&mut files[..], &mut ufiles[..], None, 2);
}

#[test]
fn test_empty_then_cat_works() {
    let mut files = [UnlimitedBuffer::new(&[]), UnlimitedBuffer::new(QUICKFOX)];
    let mut ufiles = [UnlimitedBuffer::new(&[]), UnlimitedBuffer::new(&[])];
    let mut first = true;
    for (src, dst) in files.iter_mut().zip(ufiles.iter_mut()) {
        let mut params0 = BrotliEncoderParams::default();
        params0.appendable = first;
        params0.catable = !first;
        params0.use_dictionary = first;
        super::compress(src, dst, 4096, &params0, &[], 1).unwrap();
        first = false;
    }
    concat(&mut files[..], &mut ufiles[..], None, 2);
}

#[test]
fn test_concat() {
    let mut files = [
        UnlimitedBuffer::new(ALICE),
        UnlimitedBuffer::new(RANDOMTHENUNICODE),
        UnlimitedBuffer::new(UKKONOOA),
        UnlimitedBuffer::new(ASYOULIKE),
        UnlimitedBuffer::new(BACKWARD65536),
        UnlimitedBuffer::new(EMPTY),
        UnlimitedBuffer::new(DICTWORD),
        UnlimitedBuffer::new(RANDOM10K),
        UnlimitedBuffer::new(QUICKFOX),
    ];
    let mut params0 = BrotliEncoderParams::default();
    light_debug_test(&mut params0);
    let mut params1 = params0.clone();
    params1.quality = 9;
    params1.q9_5 = true;
    params1.lgwin = 22;
    params1.magic_number = true;
    medium_debug_test(&mut params1);
    let mut params2 = params0.clone();
    params2.quality = if params1.q9_5 || params1.quality > 9 {
        9
    } else {
        8
    };
    params2.lgwin = 19;
    let mut params3 = params0.clone();
    params3.quality = 8;
    params3.lgwin = 16;
    params3.magic_number = true;
    let mut params4 = params0.clone();
    params4.quality = 7;
    params4.lgwin = 14;
    let mut params4 = params0.clone();
    params4.quality = 1;
    params4.lgwin = 10;
    let mut params5 = params0.clone();
    params5.quality = 0;
    params5.lgwin = 10;
    params5.magic_number = true;
    params0.lgwin = 26;
    params0.large_window = true;

    let mut options = [params0, params1, params2, params3, params4, params5];
    for option in options.iter_mut().skip(3) {
        let mut ufiles = [
            UnlimitedBuffer::new(&[]),
            UnlimitedBuffer::new(&[]),
            UnlimitedBuffer::new(&[]),
            UnlimitedBuffer::new(&[]),
            UnlimitedBuffer::new(&[]),
            UnlimitedBuffer::new(&[]),
            UnlimitedBuffer::new(&[]),
            UnlimitedBuffer::new(&[]),
        ];
        let mut first = true;
        for (src, dst) in files.iter_mut().zip(ufiles.iter_mut()) {
            if first {
                option.appendable = true;
            } else {
                option.appendable = false;
                option.catable = true;
                option.use_dictionary = false;
            }
            super::compress(src, dst, 4096, option, &[], 1).unwrap();
            src.reset_read();
            first = false;
        }
        concat_many_subsets(&mut files[..], &mut ufiles[..], None);
        return;
    }
    let mut ufiles = [
        UnlimitedBuffer::new(&[]),
        UnlimitedBuffer::new(&[]),
        UnlimitedBuffer::new(&[]),
        UnlimitedBuffer::new(&[]),
        UnlimitedBuffer::new(&[]),
        UnlimitedBuffer::new(&[]),
        UnlimitedBuffer::new(&[]),
        UnlimitedBuffer::new(&[]),
    ];
    let options_len = options.len();
    for (index, (src, dst)) in files.iter_mut().zip(ufiles.iter_mut()).enumerate() {
        options[min(index, options_len - 1)].catable = true;
        options[min(index, options_len - 1)].use_dictionary = false;
        options[min(index, options_len - 1)].appendable = false;
        options[min(index, options_len - 1)].quality =
            max(2, options[min(index, options_len - 1)].quality);
        // ^^^ there's an artificial limitation of using 18 as the minimum window size for quality 0,1
        // since this test depends on different window sizes for each stream, exclude q={0,1}
        super::compress(
            src,
            dst,
            4096,
            &options[min(index, options_len - 1)],
            &[],
            1,
        )
        .unwrap();
        src.reset_read();
    }
    concat_many_subsets(&mut files[..], &mut ufiles[..], None);
    concat_many_subsets(&mut files[..], &mut ufiles[..], Some(28)); // FIXME: make this 28
}
