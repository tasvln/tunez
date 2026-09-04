// pitch/target.rs
// Temporal stability layer for target-note selection.
//
// Takes raw PitchEstimate frames and produces temporally stable target notes.
// Accounts for pitch jitter, vibrato, note transitions, brief detection errors,
// and unvoiced regions.

use crate::pitch::PitchEstimate;
use crate::pitch::musical::*;

/// Target note generator state.
/// Maintains internal history for temporal stability.
pub struct TargetGenerator {
    config: TargetConfig,
    /// Last processed raw estimate (for continuity).
    last_raw: Option<PitchEstimate>,
    /// Smoothed target frequency (Hz).
    smoothed_target: f32,
    /// Whether we have a valid initial target.
    initialized: bool,
    /// Number of consecutive unvoiced frames.
    unvoiced_count: usize,
    /// Smoothing state per frame.
    target_semitone: f32,
}

impl TargetGenerator {
    /// Create a new target generator with the given configuration.
    pub fn new(config: TargetConfig) -> Self {
        Self {
            config,
            last_raw: None,
            smoothed_target: A4_FREQ,
            initialized: false,
            unvoiced_count: 0,
            target_semitone: A4_MIDI as f32,
        }
    }

    /// Process a raw pitch estimate and return the corrected target.
    pub fn process_frame(&mut self, raw: PitchEstimate) -> PitchEstimate {
        self.last_raw = Some(raw);

        if !raw.voiced {
            return self.handle_unvoiced();
        }

        // Step 1: Find the target MIDI note
        let target_midi = if self.config.mode == TargetMode::Chromatic {
            nearest_midi(raw.frequency_hz)
        } else {
            nearest_scale_note(raw.frequency_hz, self.config.key_chroma, self.config.scale_type)
        };

        let target_freq = midi_note_to_freq(target_midi);

        // Step 2: Calculate correction amount
        let cents_offset = freq_to_cents(raw.frequency_hz);
        let correction = if cents_offset.abs() >= self.config.min_correction_cents {
            let mut correction_ratio = self.config.correction_amount;
            // Scale correction by how far off the note is (up to the threshold)
            let ratio = cents_offset.abs() / self.config.min_correction_cents;
            correction_ratio = correction_ratio * ratio.min(2.0) / 2.0;
            correction_ratio.clamp(0.0, 1.0)
        } else {
            0.0
        };

        // Step 3: Apply correction (blend between raw and target)
        let corrected_freq = raw.frequency_hz + (target_freq - raw.frequency_hz) * correction;

        // Step 4: Temporal smoothing
        let new_semitone = freq_to_midi(corrected_freq);
        let smoothed_semitone = if !self.initialized {
            self.initialized = true;
            self.target_semitone = new_semitone;
            new_semitone
        } else {
            // Exponential smoothing with adaptive factor based on jump size
            let semitone_jump = (new_semitone - self.target_semitone).abs();
            let adaptive_factor = if semitone_jump > 3.0 {
                // Large jump (note transition): less smoothing for responsiveness
                self.config.smoothing_factor * 0.3
            } else {
                // Small movement (jitter/vibrato): more smoothing
                self.config.smoothing_factor
            };
            self.target_semitone * (1.0 - adaptive_factor) + new_semitone * adaptive_factor
        };

        self.smoothed_target = semitone_to_freq(smoothed_semitone);
        self.unvoiced_count = 0;

        PitchEstimate {
            frequency_hz: self.smoothed_target,
            confidence: raw.confidence,
            voiced: true,
        }
    }

    /// Handle an unvoiced frame: hold last target with reduced confidence.
    fn handle_unvoiced(&mut self) -> PitchEstimate {
        self.unvoiced_count += 1;

        // Hold the last target for up to 3 unvoiced frames, then fade
        if self.unvoiced_count < 3 {
            return PitchEstimate {
                frequency_hz: self.smoothed_target,
                confidence: self.last_raw.map(|r| r.confidence * 0.5).unwrap_or(0.0),
                voiced: true,
            };
        }

        // Longer unvoiced: return unvoiced estimate
        PitchEstimate {
            frequency_hz: self.smoothed_target,
            confidence: 0.0,
            voiced: false,
        }
    }

    /// Reset the generator state.
    pub fn reset(&mut self) {
        self.last_raw = None;
        self.smoothed_target = A4_FREQ;
        self.initialized = false;
        self.unvoiced_count = 0;
        self.target_semitone = A4_MIDI as f32;
    }

    /// Get the current smoothed target frequency.
    pub fn current_target(&self) -> Option<f32> {
        if self.initialized {
            Some(self.smoothed_target)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    #[test]
    fn test_chromatic_target_basic() {
        let config = TargetConfig {
            key_chroma: 0,
            scale_type: ScaleType::Major,
            mode: TargetMode::Chromatic,
            correction_amount: 1.0,
            min_correction_cents: 5.0,
            smoothing_factor: 0.1,
        };
        let mut target = TargetGenerator::new(config);

        // Process a clean 440 Hz note.
        let raw = PitchEstimate {
            frequency_hz: 440.0,
            confidence: 0.9,
            voiced: true,
        };
        let result = target.process_frame(raw);
        assert!(result.voiced);
        assert!((result.frequency_hz - 440.0).abs() < 0.1);
    }

    #[test]
    fn test_scale_target_snapping() {
        // A natural minor key.
        let config = TargetConfig {
            key_chroma: 9,
            scale_type: ScaleType::NaturalMinor,
            mode: TargetMode::Scale,
            correction_amount: 1.0,
            min_correction_cents: 5.0,
            smoothing_factor: 0.1,
        };
        let mut target = TargetGenerator::new(config);

        // C (349.23 Hz) is in A minor.
        let raw = PitchEstimate {
            frequency_hz: 349.23,
            confidence: 0.9,
            voiced: true,
        };
        let result = target.process_frame(raw);
        assert!(result.voiced);

        // Should snap to C4 (MIDI 60 = 261.63 Hz) or C5 (MIDI 72 = 523.25 Hz)
        // 349.23 Hz is F4, nearest in A minor is F4 (MIDI 65).
        let expected_note = nearest_scale_note(349.23, 9, ScaleType::NaturalMinor);
        let expected_freq = midi_note_to_freq(expected_note);
        assert!(
            (result.frequency_hz - expected_freq).abs() < 0.1,
            "Expected ~{expected_freq:.2} Hz, got {:.2}",
            result.frequency_hz
        );
    }

    #[test]
    fn test_unvoiced_holding() {
        let config = TargetConfig::default();
        let mut target = TargetGenerator::new(config);

        // Establish a target.
        let raw = PitchEstimate {
            frequency_hz: 440.0,
            confidence: 0.9,
            voiced: true,
        };
        target.process_frame(raw);

        // Unvoiced frames should hold.
        for _ in 0..3 {
            let unvoiced = PitchEstimate::unvoiced();
            let result = target.process_frame(unvoiced);
            assert!(result.voiced, "Should hold target during unvoiced");
        }
    }

    #[test]
    fn test_unvoiced_fade() {
        let config = TargetConfig::default();
        let mut target = TargetGenerator::new(config);

        target.process_frame(PitchEstimate {
            frequency_hz: 440.0,
            confidence: 0.9,
            voiced: true,
        });

        // After many unvoiced frames, should fade to unvoiced.
        for _ in 0..10 {
            let unvoiced = PitchEstimate::unvoiced();
            let _result = target.process_frame(unvoiced);
        }

        assert!(!target.process_frame(PitchEstimate::unvoiced()).voiced);
    }

    #[test]
    fn test_note_transition() {
        let config = TargetConfig {
            key_chroma: 0,
            scale_type: ScaleType::Major,
            mode: TargetMode::Scale,
            correction_amount: 1.0,
            min_correction_cents: 5.0,
            smoothing_factor: 0.05, // Low smoothing for responsiveness
        };
        let mut target = TargetGenerator::new(config);

        // Start on C4 (MIDI 60 = 261.63 Hz).
        target.process_frame(PitchEstimate {
            frequency_hz: 261.63,
            confidence: 0.9,
            voiced: true,
        });

        // Jump to G4 (MIDI 67 = 392.00 Hz).
        let raw = PitchEstimate {
            frequency_hz: 392.00,
            confidence: 0.9,
            voiced: true,
        };
        let result = target.process_frame(raw);
        assert!(result.voiced);

        // Should be close to G4 despite smoothing.
        assert!(
            (result.frequency_hz - 392.00).abs() < 5.0,
            "Should track note transition: got {:.2} Hz",
            result.frequency_hz
        );
    }

    #[test]
    fn test_jitter_rejection() {
        let config = TargetConfig {
            key_chroma: 9,
            scale_type: ScaleType::NaturalMinor,
            mode: TargetMode::Scale,
            correction_amount: 1.0,
            min_correction_cents: 5.0,
            smoothing_factor: 0.15,
        };
        let mut target = TargetGenerator::new(config);

        // Stable A (440 Hz) with small jitter.
        for i in 0..50 {
            let jitter = ((i % 10) as f32 - 5.0) * 0.5; // +/- 2.5 Hz jitter
            let freq = 440.0 + jitter;
            let raw = PitchEstimate {
                frequency_hz: freq,
                confidence: 0.9,
                voiced: true,
            };
            let result = target.process_frame(raw);
            assert!(result.voiced);

            // Smoothed output should stay close to 440 Hz.
            let deviation = (result.frequency_hz - 440.0).abs();
            assert!(
                deviation < 2.0,
                "Jitter rejection: deviation {deviation:.2} Hz (frame {i})"
            );
        }
    }

    #[test]
    fn test_all_keys_chromatic() {
        for key in 0..12 {
            let config = TargetConfig {
                key_chroma: key,
                scale_type: ScaleType::Major,
                mode: TargetMode::Chromatic,
                correction_amount: 1.0,
                min_correction_cents: 5.0,
                smoothing_factor: 0.1,
            };
            let mut target = TargetGenerator::new(config);

            // Test a range of frequencies.
            for &freq in &[220.0, 261.63, 329.63, 440.0, 523.25, 659.25] {
                let raw = PitchEstimate {
                    frequency_hz: freq,
                    confidence: 0.9,
                    voiced: true,
                };
                let result = target.process_frame(raw);
                assert!(result.voiced, "Key {key} freq {freq} should be voiced");
                assert!(result.frequency_hz > 0.0);
            }
        }
    }

    #[test]
    fn test_all_keys_scale() {
        for key in 0..12 {
            let config = TargetConfig {
                key_chroma: key,
                scale_type: ScaleType::NaturalMinor,
                mode: TargetMode::Scale,
                correction_amount: 1.0,
                min_correction_cents: 5.0,
                smoothing_factor: 0.1,
            };
            let mut target = TargetGenerator::new(config);

            // Test a range of frequencies.
            for &freq in &[220.0, 261.63, 329.63, 440.0, 523.25, 659.25] {
                let raw = PitchEstimate {
                    frequency_hz: freq,
                    confidence: 0.9,
                    voiced: true,
                };
                let result = target.process_frame(raw);
                assert!(result.voiced, "Key {key} freq {freq} should be voiced");
                assert!(result.frequency_hz > 0.0);
            }
        }
    }

    #[test]
    fn test_target_outside_scale() {
        // A minor key, non-scale note F#4.
        let config = TargetConfig {
            key_chroma: 9,
            scale_type: ScaleType::NaturalMinor,
            mode: TargetMode::Scale,
            correction_amount: 1.0,
            min_correction_cents: 5.0,
            smoothing_factor: 0.1,
        };
        let mut target = TargetGenerator::new(config);

        // F#4 = 369.99 Hz (NOT in A natural minor).
        let raw = PitchEstimate {
            frequency_hz: 369.99,
            confidence: 0.9,
            voiced: true,
        };
        let result = target.process_frame(raw);
        assert!(result.voiced);

        // Should snap to nearest scale note: F4 (349.23 Hz) or G4 (392.00 Hz).
        // F4 is closer (20.76 Hz vs 22.01 Hz), so should snap to F4.
        assert!(
            (result.frequency_hz - 349.23).abs() < 1.0
                || (result.frequency_hz - 392.00).abs() < 1.0,
            "F# in A minor should snap to F4 or G4, got {:.2}",
            result.frequency_hz
        );
    }

    #[test]
    fn test_correction_amount_zero() {
        let config = TargetConfig {
            key_chroma: 0,
            scale_type: ScaleType::Major,
            mode: TargetMode::Scale,
            correction_amount: 0.0,
            min_correction_cents: 5.0,
            smoothing_factor: 0.1,
        };
        let mut target = TargetGenerator::new(config);

        // A note 20 cents flat of C4.
        let freq = 261.63 * 2.0_f32.powf(-20.0 / 1200.0);
        let raw = PitchEstimate {
            frequency_hz: freq,
            confidence: 0.9,
            voiced: true,
        };
        let result = target.process_frame(raw);

        // With correction_amount=0, should not snap.
        assert!(
            (result.frequency_hz - freq).abs() < 0.5,
            "Zero correction: got {:.2}, expected ~{:.2}",
            result.frequency_hz,
            freq
        );
    }

    #[test]
    fn test_correction_amount_full() {
        let config = TargetConfig {
            key_chroma: 0,
            scale_type: ScaleType::Major,
            mode: TargetMode::Scale,
            correction_amount: 1.0,
            min_correction_cents: 5.0,
            smoothing_factor: 0.1,
        };
        let mut target = TargetGenerator::new(config);

        // A note 20 cents flat of C4.
        let freq = 261.63 * 2.0_f32.powf(-20.0 / 1200.0);
        let raw = PitchEstimate {
            frequency_hz: freq,
            confidence: 0.9,
            voiced: true,
        };
        let result = target.process_frame(raw);

        // With correction_amount=1.0, should snap to nearest scale note (C4 = 261.63).
        assert!(
            (result.frequency_hz - 261.63).abs() < 0.5,
            "Full correction: got {:.2}, expected ~261.63",
            result.frequency_hz
        );
    }

    #[test]
    fn test_vibrato_tracking() {
        let config = TargetConfig {
            key_chroma: 9,
            scale_type: ScaleType::NaturalMinor,
            mode: TargetMode::Scale,
            correction_amount: 1.0,
            min_correction_cents: 5.0,
            smoothing_factor: 0.2, // Responsive for vibrato
        };
        let mut target = TargetGenerator::new(config);

        let base_freq = 440.0;
        let vibrato_hz = 5.5;
        let vibrato_depth = 10.0;
        let duration_s = 0.15;
        let sample_rate = 44100.0;
        let num_frames = (duration_s * sample_rate / 256.0) as usize;

        for frame in 0..num_frames {
            let t = frame as f32 * 256.0 / sample_rate;
            let instant_freq = base_freq + vibrato_depth * (2.0 * PI * vibrato_hz * t).sin();
            let raw = PitchEstimate {
                frequency_hz: instant_freq,
                confidence: 0.9,
                voiced: true,
            };
            let result = target.process_frame(raw);
            assert!(result.voiced);

            // Output should stay near base frequency with minimal deviation.
            let deviation = (result.frequency_hz - base_freq).abs();
            assert!(
                deviation < 5.0,
                "Vibrato: deviation {deviation:.2} Hz at frame {frame}"
            );
        }
    }

    #[test]
    fn test_reset() {
        let config = TargetConfig::default();
        let mut target = TargetGenerator::new(config);

        target.process_frame(PitchEstimate {
            frequency_hz: 440.0,
            confidence: 0.9,
            voiced: true,
        });

        assert!(target.current_target().is_some());
        target.reset();
        assert!(target.current_target().is_none());
    }

    /// Every chromatic note in every key produces a valid target.
    #[test]
    fn test_chromatic_notes_in_all_keys() {
        for key in 0..12 {
            let config = TargetConfig {
                key_chroma: key,
                scale_type: ScaleType::Major,
                mode: TargetMode::Scale,
                correction_amount: 1.0,
                min_correction_cents: 5.0,
                smoothing_factor: 0.1,
            };
            let mut target = TargetGenerator::new(config);

            for midi in 0..=127 {
                let freq = midi_to_freq(midi as f32);
                let raw = PitchEstimate {
                    frequency_hz: freq,
                    confidence: 0.9,
                    voiced: true,
                };
                let result = target.process_frame(raw);
                assert!(
                    result.voiced,
                    "Key {key} MIDI {midi} ({freq:.2} Hz) should be voiced"
                );
                assert!(result.frequency_hz > 0.0);
            }
        }
    }

    /// Minor scales: every chromatic note in every key produces a valid target.
    #[test]
    fn test_chromatic_notes_in_minor_keys() {
        for key in 0..12 {
            let config = TargetConfig {
                key_chroma: key,
                scale_type: ScaleType::NaturalMinor,
                mode: TargetMode::Scale,
                correction_amount: 1.0,
                min_correction_cents: 5.0,
                smoothing_factor: 0.1,
            };
            let mut target = TargetGenerator::new(config);

            for midi in 0..=127 {
                let freq = midi_to_freq(midi as f32);
                let raw = PitchEstimate {
                    frequency_hz: freq,
                    confidence: 0.9,
                    voiced: true,
                };
                let result = target.process_frame(raw);
                assert!(
                    result.voiced,
                    "Key {key} minor MIDI {midi} ({freq:.2} Hz) should be voiced"
                );
                assert!(result.frequency_hz > 0.0);
            }
        }
    }
}
