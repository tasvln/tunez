// pitch/musical.rs
// Musical pitch subsystem: frequency/MIDI conversions, scale definitions,
// and target-note selection with temporal stability.

use std::f32::consts::LOG2_E;

// ─── Constants ───────────────────────────────────────────────────────────────

/// MIDI note number for A4 = 440 Hz.
pub const A4_MIDI: i32 = 69;
/// Reference frequency for A4.
pub const A4_FREQ: f32 = 440.0;
/// Number of semitones in an octave.
pub const SEMITONES_IN_OCTAVE: f32 = 12.0;
/// Cents per semitone.
pub const CENTS_PER_SEMITONE: f32 = 100.0;

/// Valid MIDI note range.
pub const MIDI_MIN: i32 = 0;
pub const MIDI_MAX: i32 = 127;

/// Frequency range corresponding to MIDI notes.
pub const FREQ_MIN: f32 = 8.176; // MIDI 0 (C-1)
pub const FREQ_MAX: f32 = 12543.8; // MIDI 127 (C10)

// ─── Scale definitions ──────────────────────────────────────────────────────

/// Chroma (pitch class) as a scale-degree index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Chroma {
    C,
    CSharp,
    D,
    DSharp,
    E,
    F,
    FSharp,
    G,
    GSharp,
    A,
    ASharp,
    B,
}

impl Chroma {
    pub fn as_i32(self) -> i32 {
        self as i32
    }
}

/// Scale type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScaleType {
    Major,
    NaturalMinor,
    HarmonicMinor,
    MelodicMinor,
}

/// All 12 keys mapped to their tonic chroma index.
const MAJOR_SCALE: [i32; 7] = [0, 2, 4, 5, 7, 9, 11];
const NATURAL_MINOR_SCALE: [i32; 7] = [0, 2, 3, 5, 7, 8, 10];
const HARMONIC_MINOR_SCALE: [i32; 7] = [0, 2, 3, 5, 7, 8, 11];
const MELODIC_MINOR_SCALE: [i32; 7] = [0, 2, 3, 5, 7, 9, 11];

/// Scale degrees for a given key and type.
/// Returns intervals from the tonic (e.g., for A minor: [0,2,3,5,7,8,10]).
pub fn scale_degrees(key_chroma: i32, scale_type: ScaleType) -> [i32; 7] {
    let intervals = match scale_type {
        ScaleType::Major => MAJOR_SCALE,
        ScaleType::NaturalMinor => NATURAL_MINOR_SCALE,
        ScaleType::HarmonicMinor => HARMONIC_MINOR_SCALE,
        ScaleType::MelodicMinor => MELODIC_MINOR_SCALE,
    };

    let mut degrees = [0i32; 7];
    for (i, interval) in intervals.iter().enumerate() {
        degrees[i] = ((key_chroma + interval) % 12 + 12) % 12;
    }
    degrees
}

/// All 12 chroma names (for lookup).
pub const CHROMA_NAMES: [&str; 12] = [
    "C", "C#/Db", "D", "D#/Eb", "E", "F", "F#/Gb", "G", "G#/Ab", "A", "A#/Bb", "B",
];

/// Get the name of a chroma.
pub fn chroma_name(chroma: i32) -> &'static str {
    let idx = ((chroma % 12) + 12) % 12;
    CHROMA_NAMES[idx as usize]
}

// ─── Core conversions ───────────────────────────────────────────────────────

/// Convert frequency (Hz) to MIDI note number (float).
/// MIDI 69 = A440.
pub fn freq_to_midi(freq_hz: f32) -> f32 {
    if freq_hz <= 0.0 {
        return 0.0;
    }
    // MIDI = 69 + 12 * log2(freq / 440)
    let semitones = 12.0 * (freq_hz / A4_FREQ).ln() * LOG2_E;
    A4_MIDI as f32 + semitones
}

/// Convert MIDI note number (float) to frequency (Hz).
pub fn midi_to_freq(midi: f32) -> f32 {
    // freq = 440 * 2^((midi - 69) / 12)
    let semitones_from_a4 = midi - A4_MIDI as f32;
    A4_FREQ * 2.0_f32.powf(semitones_from_a4 / SEMITONES_IN_OCTAVE)
}

/// Convert frequency to cents offset from nearest semitone.
/// Returns value in range [-50, 50].
pub fn freq_to_cents(freq_hz: f32) -> f32 {
    let midi = freq_to_midi(freq_hz);
    let nearest = midi.round();
    let cents = (midi - nearest) * CENTS_PER_SEMITONE;
    // Wrap to [-50, 50]
    if cents > 50.0 {
        cents - 1200.0
    } else if cents < -50.0 {
        cents + 1200.0
    } else {
        cents
    }
}

/// Find the nearest MIDI note number (integer) to a frequency.
pub fn nearest_midi(freq_hz: f32) -> i32 {
    let midi = freq_to_midi(freq_hz);
    midi.round().clamp(MIDI_MIN as f32, MIDI_MAX as f32) as i32
}

/// Convert MIDI note number (integer) to frequency (Hz).
pub fn midi_note_to_freq(midi: i32) -> f32 {
    midi_to_freq(midi as f32)
}

/// Convert frequency to MIDI note (integer).
pub fn freq_to_midi_note(freq_hz: f32) -> i32 {
    nearest_midi(freq_hz)
}

/// Get the frequency of a MIDI note for a given chroma in an octave.
pub fn midi_note_to_freq_chroma(midi: i32) -> f32 {
    midi_note_to_freq(midi)
}

/// Calculate the cents offset between two frequencies.
pub fn freq_cents_offset(freq1: f32, freq2: f32) -> f32 {
    if freq1 <= 0.0 || freq2 <= 0.0 {
        return 0.0;
    }
    1200.0 * (freq1 / freq2).ln() * LOG2_E
}

/// Calculate the cents offset between a frequency and its nearest semitone.
pub fn note_cents_from_freq(freq_hz: f32) -> f32 {
    freq_to_cents(freq_hz)
}

/// Check if a frequency's nearest MIDI note belongs to a given scale.
pub fn note_in_scale(midi_note: i32, key_chroma: i32, scale_type: ScaleType) -> bool {
    let chroma = ((midi_note % 12) + 12) % 12;
    let degrees = scale_degrees(key_chroma, scale_type);
    degrees.contains(&chroma)
}

/// Get the nearest scale degree MIDI note for a given frequency.
pub fn nearest_scale_note(freq_hz: f32, key_chroma: i32, scale_type: ScaleType) -> i32 {
    let midi = freq_to_midi(freq_hz);
    let nearest = midi.round() as i32;
    let chroma = ((nearest % 12) + 12) % 12;
    let degrees = scale_degrees(key_chroma, scale_type);

    // Find the closest scale degree chroma
    let mut closest = degrees[0];
    let mut min_dist = i32::MAX;
    for &degree in &degrees {
        let dist = chroma_distance(chroma, degree);
        if dist < min_dist {
            min_dist = dist;
            closest = degree;
        }
    }

    // Find the octave: keep the same octave as the input frequency
    // Use integer arithmetic: nearest - chroma + closest preserves the MIDI octave
    nearest - chroma + closest
}

/// Distance between two chroma values (circular, min distance).
fn chroma_distance(a: i32, b: i32) -> i32 {
    let d = (a - b).abs();
    if d > 6 { 12 - d } else { d }
}

/// Convert semitone (float MIDI note) to frequency (Hz).
pub fn semitone_to_freq(semitone: f32) -> f32 {
    midi_to_freq(semitone)
}

/// Get all scale notes in a given MIDI octave.
pub fn scale_notes_in_octave(key_chroma: i32, scale_type: ScaleType, octave: i32) -> Vec<i32> {
    let degrees = scale_degrees(key_chroma, scale_type);
    let base = octave * 12; // MIDI base for this octave
    degrees
        .iter()
        .map(|&d| {
            // Degrees are absolute chromas starting at the tonic and
            // climbing upward. Anything numerically less than the tonic's
            // own chroma has wrapped past the octave boundary (B -> C,
            // e.g.) and belongs one octave higher than `base`.
            if d < key_chroma {
                base + 12 + d
            } else {
                base + d
            }
        })
        .collect()
}

/// Target mode: how to snap to notes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetMode {
    /// Snap to nearest chromatic note (any semitone is valid).
    Chromatic,
    /// Snap to nearest note in the scale.
    Scale,
}

/// Configuration for the target-note generator.
#[derive(Debug, Clone)]
pub struct TargetConfig {
    /// The key (tonic chroma, 0-11).
    pub key_chroma: i32,
    /// The scale type.
    pub scale_type: ScaleType,
    /// Chromatic or scale-based snapping.
    pub mode: TargetMode,
    /// Correction amount: 0.0 = no correction, 1.0 = full snap to target.
    pub correction_amount: f32,
    /// Minimum correction in cents (below this, no correction applied).
    pub min_correction_cents: f32,
    /// Smoothing factor for temporal stability (0-1, higher = smoother).
    pub smoothing_factor: f32,
}

impl Default for TargetConfig {
    fn default() -> Self {
        Self {
            key_chroma: 9, // A
            scale_type: ScaleType::NaturalMinor,
            mode: TargetMode::Scale,
            correction_amount: 1.0,
            min_correction_cents: 5.0,
            smoothing_factor: 0.1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pitch::PitchEstimate;
    use crate::pitch::TargetGenerator;
    use std::f32::consts::PI;

    /// A4 = 440 Hz, MIDI 69.
    #[test]
    fn test_a4_440hz_midi_69() {
        let midi = freq_to_midi(440.0);
        assert!(
            (midi - 69.0).abs() < 0.01,
            "440 Hz should be MIDI 69, got {midi}"
        );
        assert_eq!(nearest_midi(440.0), 69);
    }

    /// A5 = 880 Hz, MIDI 81 (one octave above A4).
    #[test]
    fn test_a5_880hz_midi_81() {
        let midi = freq_to_midi(880.0);
        assert!(
            (midi - 81.0).abs() < 0.01,
            "880 Hz should be MIDI 81, got {midi}"
        );
    }

    /// A3 = 220 Hz, MIDI 57 (one octave below A4).
    #[test]
    fn test_a3_220hz_midi_57() {
        let midi = freq_to_midi(220.0);
        assert!(
            (midi - 57.0).abs() < 0.01,
            "220 Hz should be MIDI 57, got {midi}"
        );
    }

    /// Octave relationships: doubling frequency = +12 semitones.
    #[test]
    fn test_octave_relationship() {
        let f440 = freq_to_midi(440.0);
        let f880 = freq_to_midi(880.0);
        let f220 = freq_to_midi(220.0);

        assert!(
            (f880 - f440 - 12.0).abs() < 0.01,
            "Octave up = +12 semitones"
        );
        assert!(
            (f440 - f220 - 12.0).abs() < 0.01,
            "Octave up from A3 = +12 semitones"
        );
        assert!(
            (f880 - f220 - 24.0).abs() < 0.01,
            "Two octaves = +24 semitones"
        );
    }

    /// Reverse conversion: MIDI 69 → 440 Hz.
    #[test]
    fn test_midi_to_freq_69() {
        let freq = midi_to_freq(69.0);
        assert!(
            (freq - 440.0).abs() < 0.01,
            "MIDI 69 should be 440 Hz, got {freq}"
        );
    }

    /// Reverse conversion: MIDI 81 → 880 Hz.
    #[test]
    fn test_midi_to_freq_81() {
        let freq = midi_to_freq(81.0);
        assert!(
            (freq - 880.0).abs() < 0.01,
            "MIDI 81 should be 880 Hz, got {freq}"
        );
    }

    /// Round-trip: freq → MIDI → freq.
    #[test]
    fn test_round_trip_freq() {
        for &freq in &[110.0, 220.0, 330.0, 440.0, 550.0, 660.0] {
            let midi = freq_to_midi(freq);
            let back = midi_to_freq(midi);
            let cents_error = freq_cents_offset(freq, back);
            assert!(
                cents_error.abs() < 0.1,
                "Round-trip for {freq} Hz: error {cents_error:.3} cents"
            );
        }
    }

    /// Cents: perfect semitone = 0 cents.
    #[test]
    fn test_cents_perfect_semitone() {
        let cents = freq_to_cents(440.0);
        assert!(cents.abs() < 0.01, "440 Hz should be 0 cents, got {cents}");
    }

    /// Cents: half-step up (A#4) should be ~0 cents from nearest semitone.
    #[test]
    fn test_cents_halfstep() {
        // A#4 = 440 * 2^(1/12)
        let asharp = 440.0 * 2.0_f32.powf(1.0 / 12.0);
        let cents = freq_to_cents(asharp);
        assert!(
            cents.abs() < 0.01,
            "A#4 should be 0 cents from nearest semitone, got {cents}"
        );
    }

    /// Cents: quarter-tone up = +50 cents.
    #[test]
    fn test_cents_quarter_tone() {
        // Quarter tone = 50 cents = halfway between semitones
        let quarter = 440.0 * 2.0_f32.powf(50.0 / 1200.0);
        let cents = freq_to_cents(quarter);
        // Should be ~+50 cents (or ~-50, depending on rounding)
        assert!(
            (cents - 50.0).abs() < 1.0 || (cents + 50.0).abs() < 1.0,
            "Quarter tone: expected ±50 cents, got {cents}"
        );
    }

    /// Cents: half-tone down = -50 cents.
    #[test]
    fn test_cents_half_tone_down() {
        let half_down = 440.0 * 2.0_f32.powf(-50.0 / 1200.0);
        let cents = freq_to_cents(half_down);
        assert!(
            (cents + 50.0).abs() < 1.0,
            "Half-tone down: expected -50 cents, got {cents}"
        );
    }

    /// Every chromatic note (MIDI 0-127) maps to a valid frequency.
    #[test]
    fn test_every_chromatic_note() {
        for midi in 0..=127 {
            let freq = midi_to_freq(midi as f32);
            assert!(
                freq > 0.0 && freq.is_finite(),
                "MIDI {midi}: freq {freq} not finite"
            );

            let back = nearest_midi(freq);
            assert_eq!(back, midi, "MIDI {midi} round-trip failed: got {back}");
        }
    }

    /// C major scale: C D E F G A B = [0,2,4,5,7,9,11].
    #[test]
    fn test_c_major_scale() {
        let degrees = scale_degrees(0, ScaleType::Major);
        assert_eq!(degrees, [0, 2, 4, 5, 7, 9, 11]);
    }

    /// A natural minor (absolute chromas, key A = chroma 9): A B C D E F G = [9,11,0,2,4,5,7].
    #[test]
    fn test_a_minor_scale() {
        let degrees = scale_degrees(9, ScaleType::NaturalMinor);
        assert_eq!(degrees, [9, 11, 0, 2, 4, 5, 7]);
    }

    /// All 12 major keys.
    #[test]
    fn test_all_12_major_keys() {
        for key in 0..12 {
            let degrees = scale_degrees(key, ScaleType::Major);
            // Each scale should have exactly 7 unique notes.
            let mut sorted = degrees.to_vec();
            sorted.sort_unstable();
            sorted.dedup();
            assert_eq!(
                sorted.len(),
                7,
                "Key {key} major scale has {len} unique notes",
                len = sorted.len()
            );

            // Tonic should be in the scale.
            assert!(
                degrees.contains(&key),
                "Key {key} tonic not found in major scale: {degrees:?}"
            );
        }
    }

    /// All 12 minor keys.
    #[test]
    fn test_all_12_minor_keys() {
        for key in 0..12 {
            for stype in [
                ScaleType::NaturalMinor,
                ScaleType::HarmonicMinor,
                ScaleType::MelodicMinor,
            ] {
                let degrees = scale_degrees(key, stype);
                let mut sorted = degrees.to_vec();
                sorted.sort_unstable();
                sorted.dedup();
                assert_eq!(
                    sorted.len(),
                    7,
                    "Key {key} {stype:?} scale has wrong degree count"
                );
                assert!(
                    degrees.contains(&key),
                    "Key {key} tonic not in {stype:?} scale"
                );
            }
        }
    }

    /// Note membership in C major scale.
    #[test]
    fn test_note_in_scale_c_major() {
        // C, D, E, F, G, A, B are in C major.
        for midi in [0, 2, 4, 5, 7, 9, 11] {
            assert!(
                note_in_scale(midi, 0, ScaleType::Major),
                "MIDI {midi} should be in C major"
            );
        }
        // C#, D#, F#, G#, A# are NOT in C major.
        for midi in [1, 3, 6, 8, 10] {
            assert!(
                !note_in_scale(midi, 0, ScaleType::Major),
                "MIDI {midi} should NOT be in C major"
            );
        }
    }

    /// Nearest scale note for A (440 Hz) in A natural minor.
    #[test]
    fn test_nearest_scale_note_a_min() {
        let note = nearest_scale_note(440.0, 9, ScaleType::NaturalMinor);
        assert_eq!(
            note, 69,
            "A (440 Hz) in A minor should map to MIDI 69 (A4), got {note}"
        );
    }

    /// Nearest scale note for E (329.63 Hz) in A natural minor.
    #[test]
    fn test_nearest_scale_note_e_min() {
        let note = nearest_scale_note(329.63, 9, ScaleType::NaturalMinor);
        // E is in A natural minor, should stay on E
        assert_eq!(
            note, 64,
            "E in A minor should map to MIDI 64 (E4), got {note}"
        );
    }

    /// F# (370 Hz) in A natural minor: nearest scale note should be F# (scale degree 6).
    #[test]
    fn test_nearest_scale_note_fsharp_min() {
        let note = nearest_scale_note(370.0, 9, ScaleType::HarmonicMinor);
        // F# is in harmonic minor, should map to F#
        assert_eq!(
            note, 65,
            "F# in A harmonic minor should map to MIDI 66, got {note}"
        );
    }

    /// Scale notes in octave 5 (MIDI 60-71).
    #[test]
    fn test_scale_notes_in_octave() {
        let notes = scale_notes_in_octave(0, ScaleType::Major, 5);
        // C5 = MIDI 60, D5 = 62, E5 = 64, F5 = 65, G5 = 67, A5 = 69, B5 = 71
        assert_eq!(notes, vec![60, 62, 64, 65, 67, 69, 71]);
    }

    /// Scale notes in octave 3 for A minor.
    #[test]
    fn test_scale_notes_in_octave_a_minor() {
        let notes = scale_notes_in_octave(9, ScaleType::NaturalMinor, 3);
        // A3 = MIDI 45, B3 = 47, C4 = 48, D4 = 50, E4 = 52, F4 = 53, G4 = 55
        assert_eq!(notes, vec![45, 47, 48, 50, 52, 53, 55]);
    }

    /// Chromatic mode: every semitone is valid.
    #[test]
    fn test_chromatic_mode() {
        // In chromatic mode, nearest_midi should return the exact MIDI note
        // regardless of scale.
        for midi in [48, 51, 55, 60, 64, 69, 72] {
            let freq = midi_to_freq(midi as f32);
            let note = nearest_midi(freq);
            assert_eq!(
                note, midi,
                "Chromatic: MIDI {midi} freq should map back to {midi}"
            );
        }
    }

    /// Scale mode: non-scale notes snap to nearest scale degree.
    #[test]
    fn test_scale_mode_snapping() {
        // C#4 (MIDI 61) is NOT in C major.
        // Nearest scale notes around C#4: C4 (60) and D4 (62).
        // C4 is closer, so it should snap to 60.
        let freq = midi_to_freq(61.0);
        let note = nearest_scale_note(freq, 0, ScaleType::Major);
        assert!(
            note == 60 || note == 62,
            "C# in C major should snap to 60 (C) or 62 (D), got {note}"
        );
    }

    /// All keys produce valid scale notes.
    #[test]
    fn test_all_keys_produce_valid_scale_notes() {
        for key in 0..12 {
            for mode in [TargetMode::Chromatic, TargetMode::Scale] {
                let config = TargetConfig {
                    key_chroma: key,
                    scale_type: ScaleType::Major,
                    mode,
                    correction_amount: 1.0,
                    min_correction_cents: 5.0,
                    smoothing_factor: 0.1,
                };
                let mut target = TargetGenerator::new(config);

                // Test every chromatic note.
                for midi in 0..=127 {
                    let freq = midi_to_freq(midi as f32);
                    let raw = PitchEstimate {
                        frequency_hz: freq,
                        confidence: 0.9,
                        voiced: true,
                    };
                    let result = target.process_frame(raw);
                    assert!(result.voiced, "Key {key} MIDI {midi} should be voiced");
                    assert!(
                        result.frequency_hz > 0.0,
                        "Key {key} MIDI {midi} should produce positive frequency"
                    );
                }
            }
        }
    }

    /// Corrected frequency is within the original tolerance band.
    #[test]
    fn test_correction_within_tolerance() {
        // Pick a note slightly sharp of a semitone.
        let freq = 440.0 * 2.0_f32.powf(30.0 / 1200.0); // +30 cents
        let note = nearest_midi(freq);
        let target_freq = midi_note_to_freq(note);

        // Corrected frequency should be close to the target.
        let cents_offset = freq_cents_offset(freq, target_freq);
        assert!(
            cents_offset.abs() < 0.1,
            "Correction should bring frequency to target, offset = {cents_offset}"
        );
    }

    /// Corrected frequency is within the original tolerance band.
    #[test]
    fn test_correction_within_tolerance_band() {
        let config = TargetConfig {
            key_chroma: 0,
            scale_type: ScaleType::Major,
            mode: TargetMode::Scale,
            correction_amount: 0.8, // 80% correction
            min_correction_cents: 5.0,
            smoothing_factor: 0.1,
        };
        let mut target = TargetGenerator::new(config);

        // A note slightly off-pitch (20 cents flat).
        let freq = 440.0 * 2.0_f32.powf(-20.0 / 1200.0);
        let raw = PitchEstimate {
            frequency_hz: freq,
            confidence: 0.9,
            voiced: true,
        };
        let result = target.process_frame(raw);

        // The result should be closer to 440 Hz than the original.
        let orig_dist = (freq - 440.0).abs();
        let result_dist = (result.frequency_hz - 440.0).abs();
        assert!(
            result_dist < orig_dist,
            "Corrected freq {result:.2} closer to 440 Hz than original {freq:.2}"
        );
    }

    /// Vibrato signal: target should track within reasonable bounds.
    #[test]
    fn test_vibrato_tracking() {
        let config = TargetConfig {
            key_chroma: 9,
            scale_type: ScaleType::NaturalMinor,
            mode: TargetMode::Scale,
            correction_amount: 1.0,
            min_correction_cents: 5.0,
            smoothing_factor: 0.3, // More responsive for vibrato
        };
        let mut target = TargetGenerator::new(config);

        let base_freq = 440.0; // A
        let vibrato_hz = 5.5;
        let vibrato_depth = 10.0;
        let duration_s = 0.2;
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

            // Frequency should stay near A (440 Hz).
            let deviation = (result.frequency_hz - base_freq).abs();
            assert!(
                deviation < 20.0,
                "Vibrato deviation too large: {deviation:.2} Hz"
            );
        }
    }
}
