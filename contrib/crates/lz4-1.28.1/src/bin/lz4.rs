extern crate lz4;

use std::env;
use std::fs::File;
use std::io::Read;
use std::io::Result;
use std::io::Write;
use std::iter::FromIterator;
use std::path::Path;

fn main() {
    println!("LZ4 version: {}", lz4::version());
    let suffix = ".lz4";
    for arg in Vec::from_iter(env::args())[1..].iter() {
        if arg.ends_with(suffix) {
            decompress(
                &Path::new(arg),
                &Path::new(&arg[0..arg.len() - suffix.len()]),
            )
            .unwrap();
        } else {
            compress(&Path::new(arg), &Path::new(&(arg.to_string() + suffix))).unwrap();
        }
    }
}

fn compress(src: &Path, dst: &Path) -> Result<()> {
    println!("Compressing: {:?} -> {:?}", src, dst);
    let mut fi = File::open(src)?;
    let mut fo = lz4::EncoderBuilder::new().build(File::create(dst)?)?;
    copy(&mut fi, &mut fo)?;
    match fo.finish() {
        (_, result) => result,
    }
}

fn decompress(src: &Path, dst: &Path) -> Result<()> {
    println!("Decompressing: {:?} -> {:?}", src, dst);
    let mut fi = lz4::Decoder::new(File::open(src)?)?;
    let mut fo = File::create(dst)?;
    copy(&mut fi, &mut fo)
}

fn copy(src: &mut dyn Read, dst: &mut dyn Write) -> Result<()> {
    let mut buffer: [u8; 1024] = [0; 1024];
    loop {
        let len = src.read(&mut buffer)?;
        if len == 0 {
            break;
        }
        dst.write_all(&buffer[0..len])?;
    }
    Ok(())
}
