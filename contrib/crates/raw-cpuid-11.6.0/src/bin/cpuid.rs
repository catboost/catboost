//! The cpuid binary accompanying the library.
//!
//! The cpuid binary only compiles/runs on x86 platforms.
use std::str::FromStr;

use clap::{Parser, ValueEnum};
use raw_cpuid::{CpuId, CpuIdReaderNative};

#[derive(ValueEnum, Clone)]
enum OutputFormat {
    #[value(alias("raw"))]
    Raw,
    #[value(alias("cli"))]
    Cli,
}

impl FromStr for OutputFormat {
    type Err = &'static str;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "raw" => Ok(OutputFormat::Raw),
            "cli" => Ok(OutputFormat::Cli),
            _ => Err("no match"),
        }
    }
}

/// Prints information about the current x86 CPU to stdout using the cpuid instruction.
#[derive(Parser)]
#[clap(version = "10.2", author = "Gerd Zellweger <mail@gerdzellweger.com>")]
#[clap(disable_colored_help(true))]
struct Opts {
    /// Configures the output format.
    #[clap(short, long, default_value = "cli")]
    format: OutputFormat,
}

fn main() {
    let opts: Opts = Opts::parse();
    match opts.format {
        OutputFormat::Raw => raw_cpuid::display::raw(CpuIdReaderNative),
        OutputFormat::Cli => {
            let cpuid = CpuId::new();
            raw_cpuid::display::markdown(cpuid);
        }
    };
}
