use crate::{
    deflate::{BlockState, DeflateStream, Strategy},
    DeflateFlush,
};

use self::{huff::deflate_huff, rle::deflate_rle, stored::deflate_stored};

mod fast;
mod huff;
mod medium;
mod quick;
mod rle;
mod slow;
mod stored;

macro_rules! flush_block {
    ($stream:expr, $is_last_block:expr) => {
        $crate::deflate::flush_block_only($stream, $is_last_block);

        if $stream.avail_out == 0 {
            return match $is_last_block {
                true => BlockState::FinishStarted,
                false => BlockState::NeedMore,
            };
        }
    };
}
pub(crate) use flush_block;

pub fn run(stream: &mut DeflateStream, flush: DeflateFlush) -> BlockState {
    match stream.state.strategy {
        _ if stream.state.level == 0 => deflate_stored(stream, flush),
        Strategy::HuffmanOnly => deflate_huff(stream, flush),
        Strategy::Rle => deflate_rle(stream, flush),
        Strategy::Default | Strategy::Filtered | Strategy::Fixed => {
            (CONFIGURATION_TABLE[stream.state.level as usize].func)(stream, flush)
        }
    }
}

type CompressFunc = fn(&mut DeflateStream, flush: DeflateFlush) -> BlockState;

pub struct Config {
    pub good_length: u16, /* reduce lazy search above this match length */
    pub max_lazy: u16,    /* do not perform lazy search above this match length */
    pub nice_length: u16, /* quit search above this match length */
    pub max_chain: u16,
    pub func: CompressFunc,
}

impl Config {
    const fn new(
        good_length: u16,
        max_lazy: u16,
        nice_length: u16,
        max_chain: u16,
        func: CompressFunc,
    ) -> Self {
        Self {
            good_length,
            max_lazy,
            nice_length,
            max_chain,
            func,
        }
    }
}

pub static CONFIGURATION_TABLE: [Config; 10] = {
    [
        Config::new(0, 0, 0, 0, stored::deflate_stored), // 0 /* store only */
        Config::new(0, 0, 0, 0, quick::deflate_quick),   // 1
        Config::new(4, 4, 8, 4, fast::deflate_fast),     // 2 /* max speed, no lazy matches */
        Config::new(4, 6, 16, 6, medium::deflate_medium), // 3
        Config::new(4, 12, 32, 24, medium::deflate_medium), // 4 /* lazy matches */
        Config::new(8, 16, 32, 32, medium::deflate_medium), // 5
        Config::new(8, 16, 128, 128, medium::deflate_medium), // 6
        Config::new(8, 32, 128, 256, slow::deflate_slow), // 7
        Config::new(32, 128, 258, 1024, slow::deflate_slow), // 8
        Config::new(32, 258, 258, 4096, slow::deflate_slow), // 9 /* max compression */
    ]
};
