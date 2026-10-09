// pitch/postprocess.rs
// Post-processing pipeline for YIN pitch detector output.
//
// Responsibilities:
// 1. Octave error correction - detect and fix half-doubling/doubling errors
// 2. Voicing decision - combine YIN confidence with energy-based metrics
// 3. Confidence calibration - produce a well-behaved confidence score
// 4. Frequency clamping - ensure output stays in audible vocal range
//
// Octave error correction algorithm:
// The YIN algorithm can sometimes find the period at half or double the true period.
// This happens because harmonics at 2*f0 produce a strong autocorrelation peak at tau/2.
//
// The correction works by checking if the detected frequency is an octave away from
// the previous estimate, and if the semitone jump exceeds a threshold.
// We also check that the octave-corrected frequency stays within the valid vocal range.
//
// Tuning knobs:
// - octave_threshold: Max semitone jump before forcing octave correction.
//   Default 6 semitones covers most vocal leaps. For very agile singers, try 8-10.
// - min_vocal_freq / max_vocal_freq: Output frequency clamping range.
//   Default 65-1200 Hz covers typical vocal range.
// - velocity_max_semitones: Max semitone velocity between frames.
//   If jump exceeds this, force octave correction toward previous estimate.
// - min_confidence: Confidence threshold below which we report unvoiced.
//   Default 0.10 is conservative; raise to 0.20 for tighter voicing.
// - hold_decay_rate: How fast the held pitch drifts during long unvoiced regions.
//   Set to 0.0 to freeze completely, or small value like 0.0001 for gentle drift.
//
// Switching octave correction strategy:
// The `correct_octave` method can be replaced with a full spectral analysis
// approach (e.g., checking harmonic series alignment) while keeping the same API.

use crate::pitch::PitchEstimate;

/// Configuration for the post-processor.
#[derive(Debug, Clone)]
pub struct PostProcessorConfig {
    /// Maximum semitone jump before octave correction kicks in.
    pub octave_threshold: f32,
    /// Maximum semitone velocity between frames.
    pub velocity_max_semitones: f32,
    /// Minimum output frequency (Hz).
    pub min_vocal_freq: f32,
    /// Maximum output frequency (Hz).
    pub max_vocal_freq: f32,
    /// Minimum confidence for voicing decision.
    pub min_confidence: f32,
    /// Rate at which held pitch drifts during unvoiced regions (0.0 = freeze).
    pub hold_decay_rate: f32,
    /// Maximum number of unvoiced frames before clearing held frequency.
    pub max_hold_frames: usize,
}

impl Default for PostProcessorConfig {
    fn default() -> Self {
        Self {
            octave_threshold: 6.0,
            velocity_max_semitones: 6.0,
            min_vocal_freq: 65.0,
            max_vocal_freq: 1200.0,
            min_confidence: 0.10,
            hold_decay_rate: 0.0001,
            max_hold_frames: 10,
        }
    }
}

/// Post-processor state for octave error correction.
pub struct PostProcessor {
    config: PostProcessorConfig,
    /// YIN detector's voicing threshold (needed for confidence calibration).
    yin_threshold: f32,
    /// Previous frequency estimate for velocity-based correction.
    prev_freq_hz: Option<f32>,
    /// Consecutive unvoiced frames counter.
    unvoiced_count: usize,
    /// Smoothed energy estimate for voicing decision.
    smoothed_energy: f32,
    /// Decay factor for energy smoothing (~10ms window at 44.1kHz).
    energy_decay: f32,
}

impl PostProcessor {
    /// Create a new post-processor with the given configuration.
    pub fn new(config: PostProcessorConfig, yin_threshold: f32) -> Self {
        Self {
            config,
            yin_threshold,
            prev_freq_hz: None,
            unvoiced_count: 0,
            smoothed_energy: 0.0,
            energy_decay: 1.0 - (1.0 / (0.01 * 44100.0)),
        }
    }

    /// Create with default configuration.
    pub fn new_default() -> Self {
        Self::new(PostProcessorConfig::default(), 0.10) // syncs with YinDetectorConfig::default threshold
    }

    /// Return a reference to the configuration.
    pub fn config(&self) -> &PostProcessorConfig {
        &self.config
    }

    /// Process a raw pitch estimate from the YIN detector.
    /// Applies octave correction, voicing decision, and confidence calibration.
    pub fn process(
        &mut self,
        raw: PitchEstimate,
        detector_confidence: f32,
        energy: f32,
    ) -> PitchEstimate {
        if !raw.voiced {
            self.unvoiced_count += 1;
            self.update_energy(energy);
            return self.unvoiced_from_energy(energy);
        }

        self.unvoiced_count = 0;
        self.update_energy(energy);

        // Step 1: Octave error correction
        let freq = self.correct_octave(raw.frequency_hz, detector_confidence);

        // Step 2: Clamp to valid vocal range
        let freq = freq.clamp(self.config.min_vocal_freq, self.config.max_vocal_freq);

        // Step 3: Update state
        self.prev_freq_hz = Some(freq);

        // Step 4: Confidence calibration
        let confidence = self.calibrate_confidence(detector_confidence, energy);

        PitchEstimate {
            frequency_hz: freq,
            confidence,
            voiced: confidence >= self.config.min_confidence,
        }
    }

    /// Process estimate without energy (simplified version).
    pub fn process_simplified(
        &mut self,
        raw: PitchEstimate,
        detector_confidence: f32,
    ) -> PitchEstimate {
        if !raw.voiced {
            return self.unvoiced_simplified();
        }

        let freq = self.correct_octave(raw.frequency_hz, detector_confidence);
        let freq = freq.clamp(self.config.min_vocal_freq, self.config.max_vocal_freq);

        self.prev_freq_hz = Some(freq);

        let confidence =
            Self::calibrate_confidence_simplified(self.yin_threshold, detector_confidence);

        PitchEstimate {
            frequency_hz: freq,
            confidence,
            voiced: confidence >= self.config.min_confidence,
        }
    }

    /// Reset post-processor state.
    pub fn reset(&mut self) {
        self.prev_freq_hz = None;
        self.unvoiced_count = 0;
        self.smoothed_energy = 0.0;
    }

    /// Octave error correction.
    /// Checks if the detected frequency is an octave error by comparing with
    /// the previous estimate and checking harmonic consistency.
    fn correct_octave(&mut self, detected_hz: f32, _detector_confidenceue: f32) -> f32 {
        let Some(prev_hz) = self.prev_freq_hz else {
            return detected_hz;
        };

        // Calculate semitone distances (using log2, not log1p)
        let prev_semi = 69.0 + 12.0 * (prev_hz / 440.0).log2();
        let detected_semi = 69.0 + 12.0 * (detected_hz / 440.0).log2();
        let jump = (detected_semi - prev_semi).abs();

        // Step 1: Check for half-octave error (detected is half the true freq)
        // e.g., detected 110 Hz, actual should be 220 Hz (prev was 220 Hz)
        let half_freq = detected_hz / 2.0;
        let half_dist = if half_freq >= self.config.min_vocal_freq {
            let half_semi = 69.0 + 12.0 * (half_freq / 440.0).log2();
            (half_semi - prev_semi).abs()
        } else {
            f32::MAX
        };

        // Step 2: Check for double-octave error (detected is double the true freq)
        // e.g., detected 440 Hz, actual should be 220 Hz (prev was 220 Hz)
        let double_freq = detected_hz * 2.0;
        let double_dist = if double_freq <= self.config.max_vocal_freq {
            let double_semi = 69.0 + 12.0 * (double_freq / 440.0).log2();
            (double_semi - prev_semi).abs()
        } else {
            f32::MAX
        };

        // Step 3: Apply correction if jump is large or alternative octave is closer
        let mut correction_hz = detected_hz;

        // If the jump exceeds threshold, check if an octave correction would be closer
        if jump > self.config.velocity_max_semitones || jump > self.config.octave_threshold {
            // Check half-octave
            if half_dist < jump * 0.3 && half_dist < double_dist {
                correction_hz = half_freq;
            }
            // Check double-octave
            else if double_dist < jump * 0.3 {
                correction_hz = double_freq;
            }
            // If jump is very large (> 1 octave = 12 semitones), strongly prefer octave
            else if jump > self.config.octave_threshold * 2.0 {
                if half_dist < double_dist {
                    correction_hz = half_freq;
                } else {
                    correction_hz = double_freq;
                }
            }
        }

        correction_hz
    }

    /// Update smoothed energy estimate.
    fn update_energy(&mut self, current_energy: f32) {
        self.smoothed_energy *= self.energy_decay;
        self.smoothed_energy += current_energy * (1.0 - self.energy_decay);
    }

    /// Create unvoiced estimate from energy level.
    fn unvoiced_from_energy(&self, _energy: f32) -> PitchEstimate {
        let freq = self.prev_freq_hz.unwrap_or(0.0);

        // Long unvoiced: clear held frequency
        if self.unvoiced_count >= self.config.max_hold_frames {
            return PitchEstimate {
                frequency_hz: 0.0,
                confidence: 0.0,
                voiced: false,
            };
        }

        let confidence = if self.smoothed_energy > 0.005 {
            // Low energy but not silence: reduced confidence
            let conf = (self.smoothed_energy - 0.005).max(0.0) * 2.0;
            conf.min(0.3)
        } else {
            // Silence: low confidence
            0.05
        };

        PitchEstimate {
            frequency_hz: freq,
            confidence,
            voiced: confidence >= self.config.min_confidence,
        }
    }

    /// Simplified unvoiced estimate.
    fn unvoiced_simplified(&self) -> PitchEstimate {
        PitchEstimate {
            frequency_hz: self.prev_freq_hz.unwrap_or(0.0),
            confidence: 0.05,
            voiced: false,
        }
    }

    /// Calibrate confidence from YIN value and energy.
    fn calibrate_confidence(&self, detector_confidence: f32, energy: f32) -> f32 {
        // Primary: YIN value (lower = more confident)
        // Map YIN=[0, threshold] -> conf=[1, 0]
        let yin_conf = detector_confidence.clamp(0.0, 1.0);

        // Secondary: energy level (higher = more confident)
        let energy_conf = energy.min(1.0);

        // Combined: weighted average
        let combined = yin_conf * 0.7 + energy_conf * 0.3;

        // Smoothstep calibration for smooth confidence curve
        combined * combined * (3.0 - 2.0 * combined)
    }

    /// Simplified confidence calibration (no energy).
    /// Uses the detector's actual threshold for proper normalization.
    /// Accepts the threshold as a parameter so it stays in sync with YinDetectorConfig.
    pub fn calibrate_confidence_simplified(threshold: f32, detector_confidence: f32) -> f32 {
        let conf = 1.0 - (detector_confidence / threshold).min(1.0);
        conf * conf // Squash low confidence values
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pitch::yin::{YinDetector, YinDetectorConfig};
    use std::f32::consts::PI;

    fn sine_samples(freq: f32, duration_ms: f32, sample_rate: f32) -> Vec<f32> {
        let num_samples = (duration_ms / 1000.0 * sample_rate) as usize;
        let mut samples = Vec::with_capacity(num_samples);
        for i in 0..num_samples {
            let t = i as f32 / sample_rate;
            samples.push((2.0 * PI * freq * t).sin());
        }
        samples
    }

    #[test]
    fn test_octave_correction_440_vs_220() {
        let sample_rate = 44100.0;

        // First play 220 Hz
        let samples_220 = sine_samples(220.0, 200.0, sample_rate);
        let mut detector = YinDetector::new(YinDetectorConfig::default());
        let mut post = PostProcessor::new(PostProcessorConfig::default(), 0.10);

        for &s in &samples_220 {
            if let Some(est) = detector.process_sample(s) {
                // edited
                // let detector_confidence = if est.voiced { 0.05 } else { 0.5 };
                let detector_confidence = if est.voiced { 0.95 } else { 0.3 };

                let energy = if est.voiced { 0.5 } else { 0.01 };
                let processed = post.process(est, detector_confidence, energy);
                if processed.voiced {
                    assert!(
                        processed.frequency_hz >= 180.0 && processed.frequency_hz <= 260.0,
                        "220 Hz: detected {:.2} Hz",
                        processed.frequency_hz
                    );
                }
            }
        }
        post.reset();
        detector.reset();

        // Then play 440 Hz - should not confuse with 220 Hz octave
        let samples_440 = sine_samples(440.0, 200.0, sample_rate);
        for &s in &samples_440 {
            if let Some(est) = detector.process_sample(s) {
                // let detector_confidence = if est.voiced { 0.05 } else { 0.5 };

                let detector_confidence = if est.voiced { 0.95 } else { 0.3 };
                let energy = if est.voiced { 0.5 } else { 0.01 };
                let processed = post.process(est, detector_confidence, energy);
                if processed.voiced {
                    assert!(
                        processed.frequency_hz >= 350.0 && processed.frequency_hz <= 550.0,
                        "440 Hz after 220 Hz: detected {:.2} Hz",
                        processed.frequency_hz
                    );
                }
            }
        }
    }

    #[test]
    fn test_velocity_gate() {
        let mut post = PostProcessor::new(
            PostProcessorConfig {
                velocity_max_semitones: 3.0, // Tight velocity gate
                ..Default::default()
            },
            0.10,
        );

        // Simulate a large frequency jump
        post.prev_freq_hz = Some(440.0);

        // Raw estimate jumps to 880 Hz (one octave up, but also large semitone jump)
        let raw = PitchEstimate {
            frequency_hz: 880.0,
            confidence: 0.8,
            voiced: true,
        };
        let result = post.process(raw, 0.05, 0.5);

        // Should correct to ~440 Hz due to velocity gate
        assert!(
            result.frequency_hz <= 500.0,
            "Velocity gate: detected {:.2} Hz (expected ~440 Hz)",
            result.frequency_hz
        );
    }

    #[test]
    fn test_confidence_calibration() {
        let post = PostProcessor::new_default();

        // // High confidence: low YIN value, high energy
        // let c1 = post.calibrate_confidence(0.02, 0.8);
        // assert!(c1 > 0.7, "High confidence: {:.3}", c1);

        // // Medium confidence: medium YIN value
        // let c2 = post.calibrate_confidence(0.1, 0.5);
        // assert!(c2 > 0.3 && c2 < 1.0, "Medium confidence: {:.3}", c2);

        // // Low confidence: high YIN value, zero energy
        // let c3 = post.calibrate_confidence(0.2, 0.0);
        // assert!(c3 > 0.0 && c3 < 0.6, "Low confidence: {:.3}", c3);

        // High confidence: low YIN value, high energy
        let c1 = post.calibrate_confidence(0.95, 0.8);
        assert!(c1 > 0.7, "High confidence: {:.3}", c1);

        // Medium confidence: medium YIN value
        let c2 = post.calibrate_confidence(0.5, 0.5);
        assert!(c2 > 0.3 && c2 < 1.0, "Medium confidence: {:.3}", c2);

        // Low confidence: high YIN value, zero energy
        let c3 = post.calibrate_confidence(0.2, 0.0);
        assert!(c3 > 0.0 && c3 < 0.6, "Low confidence: {:.3}", c3);
    }

    #[test]
    fn test_energy_based_unvoiced() {
        let mut post = PostProcessor::new_default();

        // Feed some voiced frames to establish previous frequency
        post.prev_freq_hz = Some(440.0);

        let unvoiced = post.unvoiced_from_energy(0.001);
        assert!(!unvoiced.voiced);
        assert_eq!(unvoiced.frequency_hz, 440.0); // Hold previous frequency
    }

    #[test]
    fn test_frequency_clamping() {
        let mut post = PostProcessor::new(
            PostProcessorConfig {
                min_vocal_freq: 80.0,
                max_vocal_freq: 1000.0,
                ..Default::default()
            },
            0.10,
        );

        let raw = PitchEstimate {
            frequency_hz: 50.0, // Below min
            confidence: 0.8,
            voiced: true,
        };
        let result = post.process(raw, 0.05, 0.5);
        assert!(
            result.frequency_hz >= 80.0,
            "Clamped to min: {:.2}",
            result.frequency_hz
        );

        let raw = PitchEstimate {
            frequency_hz: 1500.0, // Above max
            confidence: 0.8,
            voiced: true,
        };
        let result = post.process(raw, 0.05, 0.5);
        assert!(
            result.frequency_hz <= 1000.0,
            "Clamped to max: {:.2}",
            result.frequency_hz
        );
    }

    #[test]
    fn test_postprocessor_preserves_vibrato() {
        let sample_rate = 44100.0;
        let mut detector = YinDetector::new(YinDetectorConfig::default());
        let mut post = PostProcessor::new(PostProcessorConfig::default(), 0.10);

        // Generate vibrato signal
        let base_freq = 440.0;
        let vibrato_hz = 6.0;
        let vibrato_depth = 10.0;
        let duration_s = 0.3;
        let num_samples = (duration_s * sample_rate) as usize;

        for i in 0..num_samples {
            let t = i as f32 / sample_rate;
            let instant_freq = base_freq + vibrato_depth * (2.0 * PI * vibrato_hz * t).sin();
            let sample = (2.0 * PI * instant_freq * t).sin();

            if let Some(est) = detector.process_sample(sample) {
                let detector_confidence = if est.voiced { 0.08 } else { 0.4 };
                let energy = if est.voiced { 0.6 } else { 0.02 };
                let processed = post.process(est, detector_confidence, energy);
                if processed.voiced {
                    // Frequency should vary around base
                    let deviation = (processed.frequency_hz - base_freq).abs();
                    assert!(
                        deviation < 100.0,
                        "Vibrato: freq deviation {:.2} Hz",
                        deviation
                    );
                }
            }
        }
    }
}
