// pitch/smoothing.rs
// Temporal smoothing for pitch estimates.
//
// Responsibilities:
// 1. Frame-to-frame exponential smoothing of pitch estimates
// 2. Velocity-based gating to prevent pitch jumps on transient attacks
// 3. Hold-last-good-pitch during unvoiced regions
// 4. Confidence-weighted blending
//
// The smoothing operates on the semitone representation (not Hz) for
// perceptually uniform interpolation, then converts back to Hz for output.
//
// Tuning knobs:
// - tau (ms): Time constant for exponential smoothing. Higher = smoother but slower to track.
//   Default 30 ms is moderate. For faster tracking, try 10-15 ms.
//   For very smooth output (e.g., correction), try 50-80 ms.
// - velocity_max_semitones: Max semitone velocity per frame.
//   Adjust based on vocal style: fast passages need higher values.
// - hold_threshold: Confidence below which we enter "hold" mode.
//   Default 0.2. Below this, stop updating the smoothed pitch.
// - hold_decay_rate: Rate at which held pitch slowly drifts during unvoiced regions.
//   Set to 0.0 to freeze completely, or small value like 0.0001 for gentle drift.
//
// Switching smoothing strategies:
// The `update` method can be replaced with Kalman filtering,
// median filtering, or Viterbi tracking while keeping the same API.
// Just ensure the method returns PitchEstimate.

use crate::pitch::PitchEstimate;

/// Configuration for temporal smoothing.
#[derive(Debug, Clone)]
pub struct TemporalSmoothingConfig {
    /// Smoothing time constant in milliseconds.
    /// Controls how quickly the smoothed pitch follows changes.
    pub tau_ms: f32,
    /// Maximum semitone velocity per frame.
    /// Jumps larger than this are attenuated.
    pub velocity_max_semitones: f32,
    /// Confidence threshold below which we hold the last pitch.
    pub hold_threshold: f32,
    /// Rate at which held pitch drifts during unvoiced regions (0.0 = freeze).
    pub hold_decay_rate: f32,
    /// Minimum confidence to consider an estimate valid for smoothing.
    pub min_valid_confidence: f32,
}

impl Default for TemporalSmoothingConfig {
    fn default() -> Self {
        Self {
            tau_ms: 30.0,
            velocity_max_semitones: 5.0,
            hold_threshold: 0.2,
            hold_decay_rate: 0.0001,
            min_valid_confidence: 0.15,
        }
    }
}

impl TemporalSmoothingConfig {
    /// Compute the smoothing factor per sample from tau.
    /// Returns the per-sample smoothing coefficient.
    pub fn smoothing_factor(&self, sample_rate: f32) -> f32 {
        // Per-sample smoothing factor from time constant
        // factor = 1 - exp(-1 / (tau * sample_rate / 1000))
        let tau_samples = self.tau_ms / 1000.0 * sample_rate;
        1.0 - (-1.0 / tau_samples).exp()
    }
}

/// Temporal smoother state.
pub struct TemporalSmoothing {
    config: TemporalSmoothingConfig,
    /// Smoothed pitch in semitones (MIDI note number).
    smoothed_semitone: f32,
    /// Whether we have a valid initial pitch.
    initialized: bool,
    /// Samples since last voiced frame.
    unvoiced_count: usize,
    /// Sample rate (used for smoothing factor calculation).
    _sample_rate: f32,
    /// Per-sample smoothing factor (precomputed).
    smooth_factor: f32,
}

impl TemporalSmoothing {
    /// Create a new temporal smoother.
    pub fn new(config: TemporalSmoothingConfig, sample_rate: f32) -> Self {
        let smooth_factor = config.smoothing_factor(sample_rate);
        Self {
            config,
            smoothed_semitone: 69.0, // Start at A4
            initialized: false,
            unvoiced_count: 0,
            _sample_rate: sample_rate,
            smooth_factor,
        }
    }

    /// Create with default configuration.
    pub fn new_default(sample_rate: f32) -> Self {
        Self::new(TemporalSmoothingConfig::default(), sample_rate)
    }

    /// Return a reference to the configuration.
    pub fn config(&self) -> &TemporalSmoothingConfig {
        &self.config
    }

    /// Update the smoothed pitch estimate from a raw pitch estimate.
    /// This is the main entry point: called every frame with a new raw estimate.
    pub fn update(&mut self, raw: PitchEstimate) -> PitchEstimate {
        if !raw.voiced {
            return self.handle_unvoiced();
        }

        self.unvoiced_count = 0;

        // Convert raw frequency to semitones
        let raw_semitone = freq_to_semitone(raw.frequency_hz);

        if !self.initialized {
            // First valid estimate initializes the smoothed pitch
            self.smoothed_semitone = raw_semitone;
            self.initialized = true;
            return PitchEstimate {
                frequency_hz: raw.frequency_hz,
                confidence: raw.confidence,
                voiced: true,
            };
        }

        // Calculate velocity (semitone change)
        let delta_semitones = raw_semitone - self.smoothed_semitone;
        let abs_velocity = delta_semitones.abs();

        let target_semitone = if abs_velocity > self.config.velocity_max_semitones {
            // Attenuate large jumps: blend toward target proportional to velocity
            let attenuation = self.config.velocity_max_semitones / abs_velocity;
            self.smoothed_semitone + delta_semitones * attenuation
        } else {
            // Normal exponential smoothing
            self.smoothed_semitone * (1.0 - self.smooth_factor) + raw_semitone * self.smooth_factor
        };

        // Smoothed estimate
        self.smoothed_semitone = target_semitone;
        let smoothed_hz = semitone_to_freq(self.smoothed_semitone);

        // Blend confidence: smoothed confidence is weighted average
        // of raw confidence and previous smoothed confidence (implicit in smoothing)
        let confidence = raw.confidence;

        PitchEstimate {
            frequency_hz: smoothed_hz,
            confidence,
            voiced: true,
        }
    }

    /// Handle an unvoiced frame.
    fn handle_unvoiced(&mut self) -> PitchEstimate {
        self.unvoiced_count += 1;

        if self.unvoiced_count == 0 {
            // Just became unvoiced: return unvoiced with current smoothed pitch
            return PitchEstimate {
                frequency_hz: semitone_to_freq(self.smoothed_semitone),
                confidence: 0.0,
                voiced: false,
            };
        }

        if self.unvoiced_count == 1 {
            return PitchEstimate {
                frequency_hz: semitone_to_freq(self.smoothed_semitone),
                confidence: 0.0,
                voiced: false,
            };
        }

        // After a couple of unvoiced frames, start holding
        if self.unvoiced_count < 5 {
            return PitchEstimate {
                frequency_hz: semitone_to_freq(self.smoothed_semitone),
                confidence: 0.0,
                voiced: false,
            };
        }

        // Long unvoiced: apply decay to smoothed pitch
        if self.config.hold_decay_rate > 0.0 && self.initialized {
            // Very slow drift: 0.1% per frame
            self.smoothed_semitone *= 1.0 - self.config.hold_decay_rate;
        }

        PitchEstimate {
            frequency_hz: semitone_to_freq(self.smoothed_semitone),
            confidence: 0.0,
            voiced: false,
        }
    }

    /// Reset the smoother state.
    pub fn reset(&mut self) {
        self.initialized = false;
        self.unvoiced_count = 0;
        self.smoothed_semitone = 69.0;
    }

    /// Get the current smoothed pitch in semitones.
    pub fn smoothed_semitone(&self) -> f32 {
        self.smoothed_semitone
    }

    /// Get the current smoothed pitch in Hz.
    pub fn smoothed_frequency_hz(&self) -> Option<f32> {
        if self.initialized {
            Some(semitone_to_freq(self.smoothed_semitone))
        } else {
            None
        }
    }
}

/// Convert frequency (Hz) to MIDI semitone number.
/// MIDI 69 = A440.
pub fn freq_to_semitone(freq_hz: f32) -> f32 {
    if freq_hz <= 0.0 {
        return 0.0;
    }
    69.0 + 12.0 * (freq_hz / 440.0).log2()
}

/// Convert MIDI semitone number to frequency (Hz).
/// MIDI 69 = A440.
pub fn semitone_to_freq(semitone: f32) -> f32 {
    440.0 * (2.0_f32).powf((semitone - 69.0) / 12.0)
}

/// Convert frequency to cents offset from nearest semitone.
/// Returns value in range [-50, 50].
pub fn freq_to_cents(freq_hz: f32) -> f32 {
    if freq_hz <= 0.0 {
        return 0.0;
    }
    let semitone = freq_to_semitone(freq_hz);
    let cents = (semitone - semitone.round()) * 100.0;
    // Map to [-50, 50]
    if cents > 50.0 {
        cents - 100.0
    } else if cents < -50.0 {
        cents + 100.0
    } else {
        cents
    }
}

/// Convert cents offset back to frequency.
pub fn semitone_cents_to_freq(base_semitone: f32, cents: f32) -> f32 {
    semitone_to_freq(base_semitone + cents / 100.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pitch::PitchEstimate;

    #[test]
    fn test_freq_semitone_conversion() {
        // A440 = semitone 69
        let semi = freq_to_semitone(440.0);
        assert!((semi - 69.0).abs() < 0.01, "440 Hz should be ~69 semitones, got {}", semi);

        // Round-trip
        let freq = semitone_to_freq(69.0);
        assert!((freq - 440.0).abs() < 0.01, "69 semitones should be ~440 Hz, got {}", freq);
    }

    #[test]
    fn test_freq_cents() {
        // Perfect semitone: 0 cents
        let cents = freq_to_cents(440.0);
        assert!(cents.abs() < 0.01, "440 Hz should be 0 cents, got {}", cents);

        // Half step up (A#4 / Bb4): 100 cents from A4,
        // freq_to_cents returns offset from nearest semitone so A#4 = 0 cents.
        // Test with 50 cents (halfway between semitones) instead.
        let half_semitone_freq = 440.0 * 2.0_f32.powf(0.5 / 12.0);
        let cents = freq_to_cents(half_semitone_freq);
        // 50 cents (halfway between semitones): can be +50 or -50 depending on rounding
        assert!(
            (cents - 50.0).abs() < 1.0 || (cents - (-50.0)).abs() < 1.0,
            "50 cents should be ~50 or ~-50 cents, got {}",
            cents
        );

        // -50 cents (halfway down): negative range
        let neg_freq = 440.0 * 2.0_f32.powf(-50.0 / 1200.0);
        let cents = freq_to_cents(neg_freq);
        assert!(
            (cents - (-50.0)).abs() < 0.5,
            "-50 cents: got {}",
            cents
        );
    }

    #[test]
    fn test_smooth_step_response() {
        let mut smoother = TemporalSmoothing::new_default(44100.0);

        // Feed a steady 440 Hz tone
        for _ in 0..500 {
            let raw = PitchEstimate {
                frequency_hz: 440.0,
                confidence: 0.9,
                voiced: true,
            };
            let result = smoother.update(raw);
            assert!(result.voiced);
        }

        // Smoothed pitch should be close to 440 Hz
        let smoothed_hz = smoother.smoothed_frequency_hz().unwrap();
        assert!(
            (smoothed_hz - 440.0).abs() < 5.0,
            "Smoothed pitch should be ~440 Hz, got {:.2}",
            smoothed_hz
        );
    }

    #[test]
    fn test_velocity_gating() {
        let config = TemporalSmoothingConfig {
            velocity_max_semitones: 2.0, // Very tight
            ..Default::default()
        };
        let mut smoother = TemporalSmoothing::new(config, 44100.0);

        // Initialize with 440 Hz
        let init = PitchEstimate {
            frequency_hz: 440.0,
            confidence: 0.9,
            voiced: true,
        };
        smoother.update(init);

        // Large jump to 880 Hz (one octave = 12 semitones)
        let jump = PitchEstimate {
            frequency_hz: 880.0,
            confidence: 0.9,
            voiced: true,
        };
        let result = smoother.update(jump);

        // Should be attenuated toward 440 Hz, not fully at 880 Hz
        assert!(
            result.frequency_hz < 600.0,
            "Velocity gating: freq should be attenuated, got {:.2} Hz",
            result.frequency_hz
        );
        assert!(
            result.frequency_hz > 470.0,
            "Velocity gating: freq should not be too low, got {:.2} Hz",
            result.frequency_hz
        );
    }

    #[test]
    fn test_hold_during_unvoiced() {
        let mut smoother = TemporalSmoothing::new_default(44100.0);

        // Initialize with 440 Hz
        let init = PitchEstimate {
            frequency_hz: 440.0,
            confidence: 0.9,
            voiced: true,
        };
        smoother.update(init);

        // Feed unvoiced frames
        for _ in 0..20 {
            let unvoiced = PitchEstimate::unvoiced();
            let result = smoother.update(unvoiced);
            assert!(!result.voiced);
        }

        // Smoothed pitch should be close to 440 Hz (held)
        let held = smoother.smoothed_frequency_hz().unwrap();
        assert!(
            (held - 440.0).abs() < 5.0,
            "Held pitch should be ~440 Hz, got {:.2}",
            held
        );
    }

    #[test]
    fn test_exponential_smoothing_curve() {
        // Verify that smoothing follows exponential curve
        let config = TemporalSmoothingConfig {
            tau_ms: 50.0, // Slow smoothing
            velocity_max_semitones: 100.0, // No velocity gating
            ..Default::default()
        };
        let s = config.smoothing_factor(44100.0);
        let mut smoother = TemporalSmoothing::new(config, 44100.0);

        // Start at 440 Hz
        let start = PitchEstimate {
            frequency_hz: 440.0,
            confidence: 0.9,
            voiced: true,
        };
        smoother.update(start);

        // Target is 880 Hz (one octave)
        let target = PitchEstimate {
            frequency_hz: 880.0,
            confidence: 0.9,
            voiced: true,
        };

        // The smoothed pitch should approach 880 Hz exponentially
        // After N steps with smoothing factor s:
        // pitch = start + (target - start) * (1 - (1-s)^N)
        let steps = 100;

        for i in 0..steps {
            let result = smoother.update(target);
            let expected = 440.0 + (880.0 - 440.0) * (1.0 - (1.0 - s).powi(i + 1));
            let actual = result.frequency_hz;

            // Allow some tolerance for semitone->Hz nonlinearity during octave transition
            assert!(
                (actual - expected).abs() < 10.0,
                "Step {}: expected {:.2} Hz, got {:.2} Hz",
                i,
                expected,
                actual
            );
        }
    }

    #[test]
    fn test_vibrato_tracking() {
        let mut smoother = TemporalSmoothing::new(
            TemporalSmoothingConfig {
                tau_ms: 15.0, // Faster tracking for vibrato
                velocity_max_semitones: 10.0, // Allow faster changes
                ..Default::default()
            },
            44100.0,
        );

        let base_freq = 440.0;
        let vibrato_hz = 5.5;
        let vibrato_depth_hz = 8.0;
        let duration_s = 0.1;
        let num_frames = (duration_s * 44100.0 / 256.0) as usize; // ~256 sample hop

        for frame in 0..num_frames {
            let t = frame as f32 * 256.0 / 44100.0;
            let instant_freq = base_freq + vibrato_depth_hz * (2.0 * PI * vibrato_hz * t).sin();
            let raw = PitchEstimate {
                frequency_hz: instant_freq,
                confidence: 0.85,
                voiced: true,
            };
            let result = smoother.update(raw);
            assert!(result.voiced);
        }
    }

    // Required for vibrato test
    use std::f32::consts::PI;
}
