//! Geometric level generator.

use rand::{Rng, SeedableRng, rngs::SmallRng};
use thiserror::Error;

use crate::level_generator::LevelGenerator;

#[derive(Error, Debug, PartialEq, Eq)]
/// Errors that can occur when creating a [`Geometric`] level generator.
#[expect(
    clippy::module_name_repetitions,
    reason = "Using 'Error' would be too generic and may cause confusion."
)]
#[non_exhaustive]
pub enum GeometricError {
    /// The maximum number of levels must be non-zero.
    #[error("max must be non-zero.")]
    ZeroMax,
    /// The maximum number of levels must be less than `i32::MAX`.
    #[error("max must be less than i32::MAX.")]
    MaxTooLarge,
    /// The probability `$p$` must be in the range `$(0, 1)$`.
    #[error("p must be in (0, 1).")]
    InvalidProbability,
    /// Failed to initialize the random number generator.
    #[error("Failed to initialize the random number generator.")]
    RngInitFailed,
}

/// A level generator using a geometric distribution.
///
/// This distribution assumes that if a node is present at some level `$n$`,
/// then the probability that it is present at level `$n+1$` is some constant
/// `$p \in (0, 1)$`. This produces a geometric distribution, albeit truncated
/// at the maximum number of levels allowed.
#[derive(Debug)]
pub struct Geometric {
    /// The total number of levels that are assumed to exist.
    total: usize,
    /// The probability that a node is not present in the next level.
    ///
    /// While the geometric distribution is defined using the probability `$p$`,
    /// the computations needed rely on `$q = 1 - p$`.
    q: f64,
    /// The random number generator.
    rng: SmallRng,
}
impl Geometric {
    /// Create a new geometric level generator with `total` number of levels,
    /// and `p` as the probability that a given node is present in the next
    /// level.
    ///
    /// # Errors
    ///
    /// `p` must be between 0 and 1 and will panic otherwise. Similarly, `total`
    /// must be at greater or equal to 1.
    #[inline]
    pub fn new(total: usize, p: f64) -> Result<Self, GeometricError> {
        if total == 0 {
            return Err(GeometricError::ZeroMax);
        }
        match i32::try_from(total) {
            Ok(_) => {}
            Err(_) => return Err(GeometricError::MaxTooLarge),
        }
        if !(0.0 < p && p < 1.0) {
            return Err(GeometricError::InvalidProbability);
        }
        #[expect(clippy::float_arithmetic, reason = "Computing q = 1 - p is fine")]
        Ok(Geometric {
            total,
            q: 1.0 - p,
            rng: SmallRng::from_rng(&mut rand::rng()),
        })
    }
}

impl LevelGenerator for Geometric {
    #[inline]
    fn total(&self) -> usize {
        self.total
    }

    /// Generate a level for a new node using a geometric distribution.
    ///
    /// This function generate a random level in the range `$[0, \text{total})$`
    /// by sample from a uniform distribution and inverting the cumulative
    /// distribution function (CDF) of the truncated geometric distribution.
    ///
    /// The CDF of the truncated geometric distribution is
    ///
    /// ```math
    /// \text{CDF}(n) = \frac{q^n - 1}{q^{t} - 1}
    /// ```
    ///
    /// where `$q = 1 - p$` and `$t$` is the total number of levels. Inverting
    /// it for `$n$` gives:
    ///
    /// ```math
    /// n = \left\lfloor \log_q\left(1 + (q^{\text{total}} - 1) \cdot u\right) \right\rfloor
    /// ```
    ///
    /// where `$u \in [0, 1]$` is a uniformly distributed random variate.
    #[inline]
    #[expect(clippy::float_arithmetic, reason = "Computing inverse CDF")]
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "CDF domain is [0, total] so the cast is safe"
    )]
    #[expect(
        clippy::expect_used,
        reason = "Runtime check guarantees total fits in i32"
    )]
    #[expect(clippy::as_conversions, reason = "No other way to do this")]
    fn level(&mut self) -> usize {
        let u = self.rng.random::<f64>();
        (1.0 + (self
            .q
            .powi(i32::try_from(self.total).expect("total is guaranteed to fit in i32"))
            - 1.0)
            * u)
            .log(self.q)
            .floor() as usize
    }
}

#[cfg(test)]
mod tests {
    use anyhow::{Result, bail};
    use pretty_assertions::assert_eq;
    use rstest::rstest;

    use super::{Geometric, LevelGenerator};
    use crate::level_generator::geometric::GeometricError;

    #[test]
    fn invalid_max() {
        assert_eq!(Geometric::new(0, 0.5).err(), Some(GeometricError::ZeroMax));
    }

    #[test]
    fn invalid_p() {
        assert_eq!(
            Geometric::new(1, 0.0).err(),
            Some(GeometricError::InvalidProbability)
        );
        assert_eq!(
            Geometric::new(1, 1.0).err(),
            Some(GeometricError::InvalidProbability)
        );
    }

    // Miri is very slow, so we use a much smaller number of iterations, and
    // don't check for the presence of min and max level nodes.
    #[cfg(miri)]
    #[rstest]
    fn new_miri(
        #[values(1, 2, 128, 1024)] n: usize,
        #[values(0.01, 0.1, 0.5, 0.99)] p: f64,
    ) -> Result<()> {
        const MAX: usize = 10;

        let mut generator = Geometric::new(n, p)?;
        assert_eq!(generator.total(), n);
        for _ in 0..MAX {
            let level = generator.level();
            assert!((0..n).contains(&level));
        }
        Ok(())
    }

    #[cfg(not(miri))]
    #[rstest]
    fn new_small(
        #[values(1, 2, 4, 8)] n: usize,
        #[values(0.01, 0.1, 0.5, 0.8)] p: f64,
    ) -> Result<()> {
        const MAX: usize = 10_000_000;

        let mut generator = Geometric::new(n, p)?;
        assert_eq!(generator.total(), n);
        for _ in 0..1_000 {
            let level = generator.level();
            assert!((0..n).contains(&level));
        }
        // Make sure that we can produce at least one level-0 node, and one at the
        // maximum level.
        let mut found = false;
        for _ in 0..MAX {
            let level = generator.level();
            if level == 0 {
                found = true;
                break;
            }
        }
        if !found {
            bail!("Failed to generate a level-0 node.");
        }

        found = false;
        for _ in 0..MAX {
            let level = generator.level();
            if level == n.checked_sub(1).expect("n is guaranteed to be > 0") {
                found = true;
                break;
            }
        }
        if !found {
            bail!(
                "Failed to generate a level-{} node.",
                n.checked_sub(1).expect("n is guaranteed to be > 0")
            );
        }

        Ok(())
    }

    #[cfg(not(miri))]
    #[rstest]
    fn new_large(#[values(512, 1024)] n: usize, #[values(0.001, 0.01)] p: f64) -> Result<()> {
        const MAX: usize = 10_000_000;

        let mut generator = Geometric::new(n, p)?;
        assert_eq!(generator.total(), n);
        for _ in 0..1_000 {
            let level = generator.level();
            assert!((0..n).contains(&level));
        }
        // Make sure that we can produce at least one level-0 node, and one at the
        // maximum level.
        let mut found = false;
        for _ in 0..MAX {
            let level = generator.level();
            if level == 0 {
                found = true;
                break;
            }
        }
        if !found {
            bail!("Failed to generate a level-0 node.");
        }

        found = false;
        for _ in 0..MAX {
            let level = generator.level();
            if level == n.checked_sub(1).expect("n is guaranteed to be > 0") {
                found = true;
                break;
            }
        }
        if !found {
            bail!(
                "Failed to generate a level-{} node.",
                n.checked_sub(1).expect("n is guaranteed to be > 0")
            );
        }

        Ok(())
    }
}
