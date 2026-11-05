//! Train a dictionary from various sources.
//!
//! A dictionary can help improve the compression of small files.
//! The dictionary must be present during decompression,
//! but can be shared across multiple "similar" files.
//!
//! Creating a dictionary using the `zstd` C library,
//! using the `zstd` command-line interface, using this library,
//! or using the `train` binary provided, should give the same result,
//! and are therefore completely compatible.
//!
//! To use, see [`Encoder::with_dictionary`] or [`Decoder::with_dictionary`].
//!
//! [`Encoder::with_dictionary`]: ../struct.Encoder.html#method.with_dictionary
//! [`Decoder::with_dictionary`]: ../struct.Decoder.html#method.with_dictionary

#[cfg(feature = "zdict_builder")]
use std::io::{self, Read};

pub use zstd_safe::{CDict, DDict};

/// Prepared dictionary for compression
///
/// A dictionary can include its own copy of the data (if it is `'static`), or it can merely point
/// to a separate buffer (if it has another lifetime).
pub struct EncoderDictionary<'a> {
    cdict: CDict<'a>,
}

impl EncoderDictionary<'static> {
    /// Creates a prepared dictionary for compression.
    ///
    /// This will copy the dictionary internally.
    pub fn copy(dictionary: &[u8], level: i32) -> Self {
        Self {
            cdict: zstd_safe::create_cdict(dictionary, level),
        }
    }
}

impl<'a> EncoderDictionary<'a> {
    #[cfg(feature = "experimental")]
    #[cfg_attr(feature = "doc-cfg", doc(cfg(feature = "experimental")))]
    /// Create prepared dictionary for compression
    ///
    /// A level of `0` uses zstd's default (currently `3`).
    ///
    /// Only available with the `experimental` feature. Use `EncoderDictionary::copy` otherwise.
    pub fn new(dictionary: &'a [u8], level: i32) -> Self {
        Self {
            cdict: zstd_safe::CDict::create_by_reference(dictionary, level),
        }
    }

    /// Returns reference to `CDict` inner object
    pub fn as_cdict(&self) -> &CDict<'a> {
        &self.cdict
    }
}

/// Prepared dictionary for decompression
pub struct DecoderDictionary<'a> {
    ddict: DDict<'a>,
}

impl DecoderDictionary<'static> {
    /// Create a prepared dictionary for decompression.
    ///
    /// This will copy the dictionary internally.
    pub fn copy(dictionary: &[u8]) -> Self {
        Self {
            ddict: zstd_safe::DDict::create(dictionary),
        }
    }
}

impl<'a> DecoderDictionary<'a> {
    #[cfg(feature = "experimental")]
    #[cfg_attr(feature = "doc-cfg", doc(cfg(feature = "experimental")))]
    /// Create prepared dictionary for decompression
    ///
    /// Only available with the `experimental` feature. Use `DecoderDictionary::copy` otherwise.
    pub fn new(dict: &'a [u8]) -> Self {
        Self {
            ddict: zstd_safe::DDict::create_by_reference(dict),
        }
    }

    /// Returns reference to `DDict` inner object
    pub fn as_ddict(&self) -> &DDict<'a> {
        &self.ddict
    }
}

/// Train a dictionary from a big continuous chunk of data, with all samples
/// contiguous in memory.
///
/// This is the most efficient way to train a dictionary,
/// since this is directly fed into `zstd`.
///
/// * `sample_data` is the concatenation of all sample data.
/// * `sample_sizes` is the size of each sample in `sample_data`.
///     The sum of all `sample_sizes` should equal the length of `sample_data`.
/// * `max_size` is the maximum size of the dictionary to generate.
///
/// The result is the dictionary data. You can, for example, feed it to [`CDict::create`].
#[cfg(feature = "zdict_builder")]
#[cfg_attr(feature = "doc-cfg", doc(cfg(feature = "zdict_builder")))]
pub fn from_continuous(
    sample_data: &[u8],
    sample_sizes: &[usize],
    max_size: usize,
) -> io::Result<Vec<u8>> {
    use crate::map_error_code;

    // Complain if the lengths don't add up to the entire data.
    if sample_sizes.iter().sum::<usize>() != sample_data.len() {
        return Err(io::Error::new(
            io::ErrorKind::Other,
            "sample sizes don't add up".to_string(),
        ));
    }

    let mut result = Vec::with_capacity(max_size);
    zstd_safe::train_from_buffer(&mut result, sample_data, sample_sizes)
        .map_err(map_error_code)?;
    Ok(result)
}

/// Train a dictionary from multiple samples.
///
/// The samples will internally be copied to a single continuous buffer,
/// so make sure you have enough memory available.
///
/// If you need to stretch your system's limits,
/// [`from_continuous`] directly uses the given slice.
///
/// [`from_continuous`]: ./fn.from_continuous.html
///
/// * `samples` is a list of individual samples to train on.
/// * `max_size` is the maximum size of the dictionary to generate.
///
/// The result is the dictionary data. You can, for example, feed it to [`CDict::create`].
#[cfg(feature = "zdict_builder")]
#[cfg_attr(feature = "doc-cfg", doc(cfg(feature = "zdict_builder")))]
pub fn from_samples<S: AsRef<[u8]>>(
    samples: &[S],
    max_size: usize,
) -> io::Result<Vec<u8>> {
    // Pre-allocate the entire required size.
    let total_length: usize =
        samples.iter().map(|sample| sample.as_ref().len()).sum();

    let mut data = Vec::with_capacity(total_length);

    // Copy every sample to a big chunk of memory
    data.extend(samples.iter().flat_map(|s| s.as_ref()).cloned());

    let sizes: Vec<_> = samples.iter().map(|s| s.as_ref().len()).collect();

    from_continuous(&data, &sizes, max_size)
}

/// Train a dictionary from multiple samples.
///
/// Unlike [`from_samples`], this does not require having a list of all samples.
/// It also allows running into an error when iterating through the samples.
///
/// They will still be copied to a continuous array and fed to [`from_continuous`].
///
/// * `samples` is an iterator of individual samples to train on.
/// * `max_size` is the maximum size of the dictionary to generate.
///
/// The result is the dictionary data. You can, for example, feed it to [`CDict::create`].
///
/// # Examples
///
/// ```rust,no_run
/// // Train from a couple of json files.
/// let dict_buffer = zstd::dict::from_sample_iterator(
///     ["file_a.json", "file_b.json"]
///         .into_iter()
///         .map(|filename| std::fs::File::open(filename)),
///     10_000,  // 10kB dictionary
/// ).unwrap();
/// ```
///
/// ```rust,no_run
/// use std::io::BufRead as _;
/// // Treat each line from stdin as a separate sample.
/// let dict_buffer = zstd::dict::from_sample_iterator(
///     std::io::stdin().lock().lines().map(|line: std::io::Result<String>| {
///         // Transform each line into a `Cursor<Vec<u8>>` so they implement Read.
///         line.map(String::into_bytes)
///             .map(std::io::Cursor::new)
///     }),
///     10_000,  // 10kB dictionary
/// ).unwrap();
/// ```
#[cfg(feature = "zdict_builder")]
#[cfg_attr(feature = "doc-cfg", doc(cfg(feature = "zdict_builder")))]
pub fn from_sample_iterator<I, R>(
    samples: I,
    max_size: usize,
) -> io::Result<Vec<u8>>
where
    I: IntoIterator<Item = io::Result<R>>,
    R: Read,
{
    let mut data = Vec::new();
    let mut sizes = Vec::new();

    for sample in samples {
        let mut sample = sample?;
        let len = sample.read_to_end(&mut data)?;
        sizes.push(len);
    }

    from_continuous(&data, &sizes, max_size)
}

/// Train a dict from a list of files.
///
/// * `filenames` is an iterator of files to load. Each file will be treated as an individual
///     sample.
/// * `max_size` is the maximum size of the dictionary to generate.
///
/// The result is the dictionary data. You can, for example, feed it to [`CDict::create`].
#[cfg(feature = "zdict_builder")]
#[cfg_attr(feature = "doc-cfg", doc(cfg(feature = "zdict_builder")))]
pub fn from_files<I, P>(filenames: I, max_size: usize) -> io::Result<Vec<u8>>
where
    P: AsRef<std::path::Path>,
    I: IntoIterator<Item = P>,
{
    from_sample_iterator(
        filenames
            .into_iter()
            .map(|filename| std::fs::File::open(filename)),
        max_size,
    )
}

#[cfg(test)]
#[cfg(feature = "zdict_builder")]
mod tests {
    use std::fs;
    use std::io;
    use std::io::Read;

    use walkdir;

    #[test]
    fn test_dict_training() {
        // Train a dictionary
        let paths: Vec<_> = walkdir::WalkDir::new("src")
            .into_iter()
            .map(|entry| entry.unwrap())
            .map(|entry| entry.into_path())
            .filter(|path| path.to_str().unwrap().ends_with(".rs"))
            .collect();

        let dict = super::from_files(&paths, 4000).unwrap();

        for path in paths {
            let mut buffer = Vec::new();
            let mut file = fs::File::open(path).unwrap();
            let mut content = Vec::new();
            file.read_to_end(&mut content).unwrap();
            io::copy(
                &mut &content[..],
                &mut crate::stream::Encoder::with_dictionary(
                    &mut buffer,
                    1,
                    &dict,
                )
                .unwrap()
                .auto_finish(),
            )
            .unwrap();

            let mut result = Vec::new();
            io::copy(
                &mut crate::stream::Decoder::with_dictionary(
                    &buffer[..],
                    &dict[..],
                )
                .unwrap(),
                &mut result,
            )
            .unwrap();

            assert_eq!(&content, &result);
        }
    }
}
