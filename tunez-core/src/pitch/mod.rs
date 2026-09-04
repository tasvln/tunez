// pitch/mod.rs
// Pitch detection subsystem for vocal pitch correction.
// Uses the YIN algorithm for robust fundamental frequency estimation.

pub mod yin;
pub mod postprocess;
pub mod smoothing;
pub mod musical;
pub mod target;

use std::fmt;

pub use yin::{YinDetector, YinDetectorConfig};
pub use postprocess::PostProcessor;
pub use smoothing::TemporalSmoothing;
pub use musical::*;
pub use target::TargetGenerator;

/// Result of a pitch detection analysis.
/// `frequency_hz` is valid only if `voiced` is true.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PitchEstimate {
    /// Estimated fundamental frequency in Hz.
    /// Valid when `voiced` is true.
    pub frequency_hz: f32,
    /// Confidence in the estimate: 0.0 (no pitch) to 1.0 (high confidence).
    pub confidence: f32,
    /// Whether the frame is voiced (contains periodic vocalization).
    pub voiced: bool,
}

impl fmt::Display for PitchEstimate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.voiced {
            write!(
                f,
                "{:.2} Hz, confidence={:.3}",
                self.frequency_hz, self.confidence
            )
        } else {
            write!(f, "unvoiced")
        }
    }
}

impl PitchEstimate {
    /// Unvoiced frame default.
    pub fn unvoiced() -> Self {
        Self {
            frequency_hz: 0.0,
            confidence: 0.0,
            voiced: false,
        }
    }

    /// Returns the period in samples (inverse of frequency).
    /// Only valid when voiced is true.
    pub fn period_samples(&self) -> f32 {
        if self.voiced && self.frequency_hz > 0.0 {
            self.frequency_hz.recip() * 44100.0
        } else {
            f32::INFINITY
        }
    }
}
