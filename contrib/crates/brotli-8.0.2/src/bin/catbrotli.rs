extern crate brotli;
extern crate core;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;
use std::{env, io};

use brotli::concat::{BroCatli, BroCatliResult};
fn usage() {
    writeln!(
        &mut ::std::io::stderr(),
        "Usage: [-w<window_size>] filename0 filename1 filename2..."
    )
    .unwrap();
}
fn read_no_interrupt<R: Read>(r: &mut R, buf: &mut [u8]) -> Result<usize, io::Error> {
    loop {
        match r.read(buf) {
            Err(e) => match e.kind() {
                io::ErrorKind::Interrupted => continue,
                _ => return Err(e),
            },
            Ok(cur_read) => return Ok(cur_read),
        }
    }
}

fn write_no_interrupt<W: Write>(w: &mut W, mut buf: &[u8]) -> Result<usize, io::Error> {
    let mut total_read = 0usize;
    loop {
        match w.write(buf) {
            Err(e) => match e.kind() {
                io::ErrorKind::Interrupted => continue,
                _ => return Err(e),
            },
            Ok(cur_read) => {
                buf = &buf[cur_read..];
                total_read += cur_read;
                if buf.is_empty() {
                    return Ok(total_read);
                }
            }
        }
    }
}

fn main() {
    let mut window_size: Option<u8> = None;
    let mut double_dash = false;
    let mut buffer_size = 4096usize;
    let mut filenames = Vec::<String>::new();
    let mut ostream = io::stdout();
    if env::args_os().len() > 1 {
        for argument in env::args().skip(1) {
            if argument.starts_with("-w") && !double_dash {
                window_size = Some(
                    argument
                        .trim_matches('-')
                        .trim_matches('w')
                        .parse::<i32>()
                        .unwrap() as u8,
                );
                continue;
            }
            if argument.starts_with("-bs") && !double_dash {
                buffer_size = argument
                    .trim_matches('-')
                    .trim_matches('b')
                    .trim_matches('s')
                    .parse::<usize>()
                    .unwrap();
                continue;
            }
            if argument == "--" {
                double_dash = true;
                continue;
            }
            filenames.push(argument);
        }
    } else {
        usage();
        return;
    }
    let mut ibuffer = vec![0u8; buffer_size];
    let mut obuffer = vec![0u8; buffer_size];
    let mut ooffset = 0;
    let mut ioffset;
    let mut bro_cat_li = match window_size {
        Some(ws) => BroCatli::new_with_window_size(ws),
        None => BroCatli::new(),
    };
    for filename in filenames {
        bro_cat_li.new_brotli_file();
        let mut input_file = match File::open(Path::new(&filename)) {
            Err(why) => panic!("couldn't open {:}\n{:}", filename, why),
            Ok(file) => file,
        };
        loop {
            ioffset = 0;
            match read_no_interrupt(&mut input_file, &mut ibuffer[..]) {
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
                                match write_no_interrupt(&mut ostream, &obuffer[..ooffset]) {
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
                                panic!(
                                    "Failed to concatenate files on {:} {:?}",
                                    filename, failure
                                );
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
                match write_no_interrupt(&mut ostream, &obuffer[..ooffset]) {
                    Err(why) => panic!("couldn't write: {:}", why),
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
                    match write_no_interrupt(&mut ostream, &obuffer[..ooffset]) {
                        Err(why) => panic!("couldn't write: {:}", why),
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
