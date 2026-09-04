// tunez-core: shared pitch detection and DSP logic for vocal pitch correction.
//
// No host dependencies (VST3/CLAP). Designed to be independently testable.

pub mod pitch;

// Re-export the main types at crate root for convenience.
pub use pitch::{
    PitchEstimate, YinDetector, YinDetectorConfig, PostProcessor, TemporalSmoothing,
};
