# TUNEZ — Architecture Specification

Vocal pitch-correction audio plugin (VST3 + CLAP). Real-time, commercial-quality, monophonic vocal pipeline.

---

## 1. DIRECTORY STRUCTURE

```
tunez/
├── Cargo.toml
├── ARCHITECTURE.md          (this file)
├── docs/
│   ├── algorithms.md        (detailed algorithm descriptions)
│   ├── testing.md           (test strategy overview)
│   └── latency.md           (latency analysis)
│
├── tunez-core/              (shared logic, no host deps)
│   └── src/
│       ├── lib.rs
│       ├── config.rs        (app-level configuration)
│       │
│       ├── pitch/
│       │   ├── mod.rs
│       │   ├── yin.rs         (pitch detection — YIN/FFT-YIN)
│       │   └── postprocess.rs (peak picking, octave correction)
│       │
│       ├── tracking/
│       │   ├── mod.rs
│       │   ├── tracker.rs     (stateful pitch tracker)
│       │   └── viterbi.rs     (optional: Viterbi tracking for stability)
│       │
│       ├── musical/
│       │   ├── mod.rs
│       │   ├── note.rs        (note definitions, frequency mapping)
│       │   ├── scale.rs       (scale degree lookup)
│       │   └── key.rs         (key + scale presets)
│       │
│       ├── correction/
│       │   ├── mod.rs
│       │   ├── target.rs      (target note generation)
│       │   ├── flex_tune.rs   (correction threshold / flex-tune)
│       │   ├── retune.rs      (retune speed / attack-release)
│       │   └── humanize.rs    (micro-pitch randomization)
│       │
│       ├── dsp/
│       │   ├── mod.rs
│       │   ├── pitch_shift.rs (phase-locked vocoder / PSOLA)
│       │   ├── formant.rs     (formant preservation / shift)
│       │   ├── delay_line.rs  (circular buffer for DSP)
│       │   ├── fft.rs         (FFT utilities, FFT-YIN)
│       │   └── interpolators.rs (sinc/lagrange)
│       │
│       ├── params/
│       │   ├── mod.rs
│       │   ├── model.rs       (parameter model — all params)
│       │   ├── smoothed.rs    (parameter smoothing)
│       │   ├── id.rs          (parameter IDs, ranges)
│       │   └── automation.rs  (automation point handling)
│       │
│       ├── latency/
│       │   ├── mod.rs
│       │   ├── manager.rs     (latency calculation)
│       │   └── compensation.rs (input-output delay alignment)
│       │
│       └── test_utils.rs      (shared test helpers)
│
├── tunez-dsp/               (real-time pipeline, no GUI)
│   └── src/
│       ├── lib.rs
│       ├── pipeline.rs      (main processing pipeline)
│       ├── state.rs         (DSPState — owned by audio thread)
│       ├── callback.rs      (process_audio entry point)
│       ├── scheduler.rs     (parameter scheduling / smoothing)
│       └── metrics.rs       (pitch tracking metrics, debug output)
│
├── tunez-plugin/            (VST3 + CLAP plugin hosts)
│   └── src/
│       ├── lib.rs
│       ├── plugin.rs        (base plugin trait impl)
│       ├── audio.rs         (audio processing callback)
│       ├── midi.rs          (MIDI note input / key tracking)
│       ├── process.rs       (buffer processing setup)
│       │
│       ├── vst3/
│       │   ├── mod.rs
│       │   ├── component.rs (VST3 component)
│       │   ├── params.rs    (VST3 parameter handling)
│       │   └── latency.rs   (VST3 latency ports)
│       │
│       ├── clap/
│       │   ├── mod.rs
│       │   ├── component.rs (CLAP plugin)
│       │   ├── params.rs    (CLAP parameter handling)
│       │   ├── ext_audio_ports.rs
│       │   ├── ext_params.rs
│       │   ├── ext_render.rs (latency)
│       │   └── ext_state.rs (CLAP state)
│       │
│       └── presets/
│           ├── mod.rs
│           ├── loader.rs    (load/save presets)
│           └── format.rs    (preset serialization format)
│
├── tunez-editor/            (GUI / editor)
│   └── src/
│       ├── lib.rs
│       ├── editor.rs        (base editor trait impl)
│       ├── ui/
│       │   ├── mod.rs
│       │   ├── controls.rs  (knob, slider, button widgets)
│       │   ├── display.rs   (pitch display / visualization)
│       │   ├── keyboard.rs  (virtual piano keyboard)
│       │   ├── scale_map.rs (scale key selection UI)
│       │   └── panels.rs    (panel layout)
│       ├── state.rs         (editor state — GUI side)
│       └── message.rs       (GUI<->DSP message passing)
│
├── tunez-standalone/        (standalone app for dev/test)
│   └── src/
│       ├── main.rs
│       ├── app.rs           (standalone app setup)
│       └── audio_backend.rs (portaudio / rodio / cpal)
│
├── tunez-tools/             (offline analysis / testing)
│   └── src/
│       ├── main.rs
│       ├── analyze.rs       (analyze audio file pitch)
│       ├── compare.rs       (compare output vs reference)
│       └── waveform.rs      (waveform rendering for debugging)
│
├── tunez-tests/             (integration tests)
│   └── tests/
│       ├── pitch_detection.rs
│       ├── tracker.rs
│       ├── correction.rs
│       ├── pitch_shifting.rs
│       ├── formant.rs
│       ├── latency.rs
│       └── automation.rs
│
└── tests/                   (integration / end-to-end)
    ├── test_audio/          (test WAV files)
    ├── reference_output/    (golden reference outputs)
    ├── pitch_correct.rs
    └── preset_roundtrip.rs
```

---

## 2. MODULE ARCHITECTURE

### Layer Overview (dependency direction: left to right)

```
  tunez-core          tunez-dsp           tunez-plugin          tunez-editor
  [shared logic]  ->  [real-time pipe]  ->  [host bindings]  ->  [GUI]
      ^                     ^
      |                     |
  tunez-core          tunez-tools
  [testable units]  ->  [offline analysis]
```

Each module in `tunez-core` is independently testable with no host dependencies. `tunez-dsp` consumes core modules for the real-time pipeline. `tunez-plugin` wraps DSP with VST3/CLAP host integration. `tunez-editor` connects to the plugin's parameter model.

---

## 3. DATA FLOW

### Audio Processing Path (per sample frame)

```
Input PCM
  |
  v
+------------------+
| Pre-emphasis     |  (optional, high-shelf to boost HF for pitch detection)
+------------------+
  |
  v
+------------------+
| Pitch Detection  |  YIN/FFT-YIN on buffered chunks (see latency section)
+------------------+
  |  -> detected_pitch (Hz, per-chunk)
  v
+------------------+
| Pitch Tracker    |  Stabilize detected pitch: octave correction,
|                  |  velocity filtering, confidence scoring
+------------------+
  |  -> tracked_pitch (Hz, smoothed)
  v
+------------------+
| Musical Mapping  |  Convert pitch to semitone cents from note
|                  |  in selected key/scale
+------------------+
  |  -> semitone, cents
  v
+------------------+
| Target Generator |  Find nearest scale-degree note. Handle:
|                  |  chromatic, diatonic, flex-tune threshold
+------------------+
  |  -> target_pitch (Hz)
  v
+------------------+
| Retune Smoothing |  Crossfade between input and target over
|                  |  retune speed (attack/release time)
+------------------+
  |  -> target_pitch (Hz, time-varying)
  v
+------------------+
| Humanize         |  Apply micro-randomization to target pitch
|                  |  based on deviation magnitude
+------------------+
  |  -> final_pitch (Hz)
  v
+------------------+
| Pitch Shifter    |  Phase-locked vocoder: shift samples by
|                  |  final_pitch / tracked_pitch ratio
+------------------+
  |  -> shifted PCM
  v
+------------------+
| Formant Processor|  Preserve or shift formants independently
|                  |  of pitch shift
+------------------+
  |  -> processed PCM
  v
+------------------+
| Dry/Wet Mix      |  Crossfade between dry input and wet output
+------------------+
  |
  v
Output PCM + gain + latency offset

Parameter updates (scheduled, not per-sample):
  retune_speed, flex_tune, humanize_amount, formant_shift, dry_wet, gain, key, scale
  |
  v
  ParameterScheduler (read from smoothed values in callback)
```

### Parameter Update Path (GUI -> DSP)

```
GUI (tunez-editor)
  |
  | set_param(id, value)
  v
+------------------+
| ParamModel       |  Validate, clamp, update target value
| (tunez-core)     |
+------------------+
  |
  | smoothed value (per-sample via exponential smoothing)
  v
+------------------+
| Scheduler        |  Queue param changes, resolve in audio callback
| (tunez-dsp)      |  via parameter scheduling
+------------------+
  |
  | SmoothedParam<T> per parameter
  v
+------------------+
| DSPState         |  Read smoothed values each process cycle
| (tunez-dsp)      |  in real-time safe manner
+------------------+
```

---

## 4. THREADING MODEL

```
+---------------------------------------------------+
|                  Host Process                     |
|                                                   |
|  +------------------+       +-------------------+  |
|  |  GUI / Editor    |       |  Audio Callback   |  |
|  |  (high priority) |       |  (real-time)      |  |
|  |                  |       |                   |  |
|  |  reads params    |<----->|  reads smoothed   |  |
|  |  sends changes   |  Arc  |  params from      |  |
|  |                  |  Mutex|  DSPState         |  |
|  +------------------+       +-------------------+  |
|         |                              |            |
|         |     +----------------+       |            |
|         +---->| ParamModel     |<------+            |
|               | (ArcMutex)     |                   |
|               +----------------+                   |
|                                                   |
|  +---------------------------------------------+  |
|  |  Preset Loader (off-main thread)            |  |
|  |  (file I/O, deserialization)                |  |
|  +---------------------------------------------+  |
+---------------------------------------------------+
```

Key rules:

- `DSPState` wrapped in `Arc<Mutex<DSPState>>` or `Atomic` for lock-free read in audio thread.
- `ParamModel` shared via `Arc` between editor and audio thread.
- Audio callback holds mutex for at most one short lock cycle to snapshot all smoothed parameters.
- No allocations in audio callback — pre-allocated circular buffers.
- No blocking in audio callback — `parking_lot::Mutex` (fast mutex, no OS contention).
- Preset loading happens on editor thread; audio thread reads via `ArcSwap` or atomic snapshot.

---

## 5. PARAMETER MODEL

### Parameter Categories

```
STRUCT ParamModel {
    // --- Pitch Detection ---
    fft_size:          NonZeroU32,      // 512, 1024, 2048, 4096
    hop_size:          NonZeroU32,       // fraction of fft_size

    // --- Pitch Tracking ---
    tracking_mode:     TrackingMode,     // FAST, SMOOTH, VITERBI

    // --- Musical ---
    key:               Key,              // C, C#, D, ..., B (12)
    scale:             Scale,            // CHROMATIC, MAJOR, MINOR, etc. (8+ scales)

    // --- Correction ---
    retune_speed_ms:   f32,              // 0.05ms (hard) to 200ms (auto-tune slow)
    flex_tune:         f32,              // 0.0 (full correction) to 1.0 (no correction)
    correction_threshold_db: f32,        // below this cents threshold, use input pitch
    humanize_amount:   f32,              // 0.0 (no) to 1.0 (max micro-randomization)
    vibrato_amount:    f32,              // 0.0 (disable) to 1.0 (full vibrato)
    vibrato_speed_hz:  f32,              // 4-8 Hz typical
    vibrato_depth_cents: f32,            // 5-40 cents typical

    // --- Formants ---
    formant_shift:     f32,              // 0.5 to 2.0 (1.0 = preserve)
    formant_preserve:  bool,             // quick toggle

    // --- Output ---
    dry_wet:           f32,              // 0.0 (dry) to 1.0 (wet)
    bypass:            bool,
    input_gain_db:     f32,              // -24 to +24 dB
    output_gain_db:    f32,              // -24 to +24 dB

    // --- MIDI (optional) ---
    midi_key_input:    bool,             // follow MIDI note for key/scale
}
```

### Parameter Smoothing

Each numeric parameter uses an exponential moving average (EMA) smoother:

```rust
struct SmoothedParam {
    current: f32,      // current interpolated value (updated per sample)
    target:  f32,      // target value (set by GUI)
    samples_remaining: usize,  // samples until target reached
    sample_rate: f32,
}
```

Smoothing formula: `current += (target - current) * (1.0 - exp(-1.0 / (samples_remaining / sample_rate)))`

This ensures zero-crossing-free parameter transitions, no clicks.

### Parameter IDs

Each parameter has a unique integer ID for host automation and preset serialization. IDs are stable across versions.

---

## 6. DSP PROCESSING PIPELINE

### Block Processing

Audio is processed in blocks of `BLOCK_SIZE` samples (typically 64 or 128). The pitch detection operates on chunks of `HOP_SIZE` samples, which may span multiple blocks.

```
Block processing (per block):
  1. Read BLOCK_SIZE samples from input buffer
  2. Accumulate into pitch detection ring buffer
  3. If ring buffer has HOP_SIZE samples:
     a. Run YIN on the hop window
     b. Run tracker update on detected pitch
     c. Update musical mapping
     d. Generate correction target
     e. Update smoothed parameters
  4. Apply pitch shift with current parameters
  5. Apply formant processing
  6. Apply dry/wet mix and gain
  7. Write to output buffer

Latency = FFT half-overlap + tracker delay + smoothing delay
  = HOP_SIZE/2 + ~5ms tracker + ~0ms smoothing
  = typically 10-25ms depending on FFT size
```

### Pitch Detection Pipeline

```
Audio window (HOP_SIZE samples)
  |
  v
+------------------+
| Apply Hann window|  w[n] = 0.5 * (1 - cos(2*pi*n/(N-1)))
+------------------+
  |
  v
+------------------+
| YIN Algorithm    |  Step 1: Difference function
|                  |  d[n] = sum((x[i] - x[i+n])^2) for i in window
|                  |  Step 2: Cumulative mean normalized difference
|                  |  YIN[n] = d[n] * sum(d[0..n]) / (n * sum(d[0..n]))
|                  |  Step 3: Absolute threshold comparison
|                  |  Step 4: Parabolic interpolation for finer resolution
+------------------+
  |  -> f0_hz (detected fundamental)
  v
+------------------+
| Post-processing  |  - Clip to audible range (50-2000 Hz)
|                  |  - Octave error detection via harmonic consistency
|                  |  - Confidence score
+------------------+
  |
  v
  f0_hz, confidence
```

### Pitch Tracking Pipeline

```
Input: raw f0_hz, confidence from pitch detector
Output: smoothed, octave-corrected f0_hz

  +------------------+
  | Octave Correction|  Compare consecutive frames; if jump > 1 octave,
  |                  |  multiply/divide by 2 to find nearest plausible
  +------------------+
  |
  v
+------------------+
| Velocity Filter  |  If velocity of pitch change exceeds threshold,
|                  |  apply exponential smoothing to avoid jumps
+------------------+
  |
  v
+------------------+
| Confidence Gating|  If confidence < threshold, hold last good pitch
|                  |  (don't reset, just maintain)
+------------------+
  |
  v
  tracked_f0_hz
```

### Musical Mapping Pipeline

```
Input: tracked_f0_hz
Output: semitone, cents_from_note, scale_degree, target_note_hz

  +------------------+
  | Convert to       |  semitone = 69 + 12 * log2(f0_hz / 440.0)
  | Semitone+Cents   |  cents = (semitone - round(semitone)) * 100
+------------------+
  |
  v
+------------------+
  | Scale Degree     |  For diatonic modes: find nearest note in scale
  | Lookup           |  For chromatic: always use nearest semitone
  |                  |  scale_degree = find_nearest_scale_note(key, scale, semitone)
+------------------+
  |
  v
+------------------+
| Flex-Tune          |  If |cents| < flex_tune_threshold:
| Threshold          |    use input pitch (no correction)
|                  |  Else:
|                  |    target = nearest scale note
+------------------+
  |
  v
  target_note_hz, semitone, cents
```

### Pitch Correction (Retune) Pipeline

```
Input: input_f0_hz, target_note_hz
Output: corrected_f0_hz (time-varying)

  +------------------+
  | Calculate        |  distance = |input_f0 - target_f0| in cents
  | Distance         |
+------------------+
  |
  v
+------------------+
  | Retune Speed     |  Over retune_speed_ms samples:
  | Crossfade        |    - Start blending from input toward target
  | (Attack)         |    - Full correction at attack end
  |                  |    - Exponential curve for natural feel
+------------------+
  |
  v
+------------------+
  | Humanize         |  If humanize_amount > 0:
  | Micro-randomize  |    offset = random(-max_cents, +max_cents) * humanize_amount
  | Target           |    max_cents scales with distance to target
+------------------+
  |
  v
  corrected_f0_hz
```

### Pitch Shifting Pipeline (Phase-Locked Vocoder)

```
Input: input_samples, correction_ratio = target_f0 / input_f0
Output: shifted_samples

  +------------------+
  | Analyze Stage    |  Process in overlapping analysis frames:
  | (FFT-YIN already  |  - Compute phase of each FFT bin
  | done above)      |  - Track phase progression per bin
  +------------------+
  |
  v
+------------------+
  | Synthesize Stage |  For each synthesis frame:
  |                  |  - Adjust phase increment per bin:
  |                  |    new_phase[n] = old_phase[n] + bin_freq * correction_ratio
  |                  |  - Sum sinusoidal components
  |                  |  - Apply synthesis window, overlap-add
  +------------------+
  |
  v
  shifted_samples
```

Alternative: PSOLA (Pitch Synchronization Overlap-Add) for simpler implementation.

### Formant Processing Pipeline

```
Input: shifted_audio
Output: formant_processed_audio

  +------------------+
  | Linear Prediction|  Compute LPC coefficients from shifted audio
  | (LPC)            |  (order ~10-15 for vocal formants)
  +------------------+
  |
  v
+------------------+
  | Formant          |  Extract formant frequencies F1, F2, F3, ...
  | Extraction       |  from LPC poles
  +------------------+
  |
  v
+------------------+
  | Formant Shift    |  Apply multiplicative shift:
  |                  |    F1' = F1 * formant_shift
  |                  |    F2' = F2 * formant_shift
  |                  |    (formant_shift = 1.0 preserves)
  +------------------+
  |
  v
+------------------+
  | Resynthesis      |  Rebuild audio from shifted formants
  | (LP filter)      |  using excitation source
  +------------------+
  |
  v
  formant_processed_audio
```

Simpler alternative for MVP: All-pass phase vocoder formant shift (simpler, less natural).

---

## 7. RECOMMENDED IMPLEMENTATION ORDER

### Phase 1: Foundation (Week 1-2)
1. **Cargo workspace** setup with all crates
2. **ParamModel** — parameter definitions, IDs, ranges, defaults
3. **Musical module** — note definitions, scale lookup, key mapping
4. **SmoothedParam** — parameter smoothing implementation
5. **Circular buffer** — ring buffer for audio/DSP

### Phase 2: Pitch Detection (Week 3-4)
6. **YIN algorithm** — difference function, cumulative mean normalized difference, parabolic interpolation
7. **Post-processing** — octave correction, confidence scoring
8. **Unit tests** against known sine waves and recorded vocals

### Phase 3: Pitch Tracking (Week 5)
9. **Tracker** — stateful tracking with velocity filtering, confidence gating
10. **Integration test** — track pitch through a vocal melody

### Phase 4: Musical Correction (Week 6)
11. **Target generator** — nearest note lookup, flex-tune threshold
12. **Retune smoothing** — crossfade between input and target
13. **Humanize** — micro-randomization
14. **Integration test** — correct pitch to a scale

### Phase 5: Pitch Shifting (Week 7-8)
15. **Phase-locked vocoder** — analysis/synthesis with phase tracking
16. **Unit tests** — verify pitch shift preserves formants (before formant module)
17. **Frequency response tests** — verify no artifacts

### Phase 6: Formants (Week 9)
18. **LPC-based formant extraction and shift**
19. **Fallback: all-pass phase vocoder** formant preservation
20. **Tests** — verify formant shift on vowels

### Phase 7: Pipeline Integration (Week 10)
21. **DSP pipeline** — wire all modules together
22. **Latency management** — calculate and report processing delay
23. **Dry/wet mix** — crossfade implementation
24. **Gain staging** — input/output gain

### Phase 8: Plugin Hosts (Week 11-12)
25. **VST3 plugin** — using `vst3` crate, parameter handling, latency ports
26. **CLAP plugin** — using `clap` crate, param extensions, state
27. **Audio backend** — portaudio or cpal for both hosts

### Phase 9: Editor (Week 13-14)
28. **Editor trait** — plugin/editor communication
29. **UI controls** — knob, slider, button widgets
30. **Pitch visualization** — pitch display, scale map
31. **Keyboard input** — virtual piano for key/scale selection

### Phase 10: Polish (Week 15+)
32. **Preset system** — save/load, JSON format
33. **Standalone app** — for testing without hosts
34. **Tools** — offline analysis, reference comparison
35. **Vibrato preservation**
36. **MIDI key input**
37. **Integration tests** — full pipeline with test audio
38. **Latency optimization** — FFT size tuning, block size
39. **Performance profiling** — CPU usage < X% at block size Y

---

## 8. MAJOR TECHNICAL RISKS

### Risk 1: YIN on vocals is unreliable
- **Problem**: YIN works well on pure tones; vocals have harmonics, breath noise, vibrato
- **Mitigation**: Use FFT-YIN variant; add harmonic consistency check; combine with autocorrelation; use confidence gating to hold pitch during low-confidence regions
- **Fallback**: Adaptive threshold YIN tuned for vocal spectra

### Risk 2: Formant preservation quality
- **Problem**: Simple pitch shifting shifts formants too (chipmunk effect)
- **Mitigation**: LPC-based formant extraction is accurate; all-pass phase vocoder is simpler but less natural
- **Risk level**: Medium. May need to iterate on LPC order and windowing

### Risk 3: Latency vs. accuracy trade-off
- **Problem**: Smaller FFT = lower latency but worse pitch resolution. Large FFT = good accuracy but high latency (>30ms is noticeable)
- **Mitigation**: Adaptive FFT size based on pitch range; minimum 256 samples hop; target <20ms end-to-end latency
- **Risk level**: Medium. Will need profiling on real hardware

### Risk 4: Monophonic assumption
- **Problem**: Chorus, harmonies, or breath sounds confuse pitch detection
- **Mitigation**: Design for solo vocal; use confidence gating; add "breath mode" that bypasses correction during low-energy regions
- **Risk level**: Low — documented limitation

### Risk 5: Parameter smoothing click-free transitions
- **Problem**: Discontinuous parameter changes cause pops/clicks
- **Mitigation**: All parameters use exponential smoothing with configurable time constant; pitch correction crossfade is always active during retune transitions
- **Risk level**: Low — standard technique

### Risk 6: Rust audio plugin ecosystem maturity
- **Problem**: Rust VST3/CLAP crates may have gaps or bugs
- **Mitigation**: Use `vst3` (mature) and `clap` (growing); fall back to raw FFI if needed; target CLAP first (simpler API)
- **Risk level**: Medium. CLAP spec is simpler than VST3

---

## 9. MEASURABLE ACCEPTANCE CRITERIA

### DSP Quality
- [ ] Pitch detection accuracy: <5 cents RMS error on sustained pure tones (100-1000 Hz)
- [ ] Pitch detection accuracy: <15 cents median absolute error on sustained vocal phrases
- [ ] Octave error rate: <2% on solo vocal phrases (tested on 10 sample recordings)
- [ ] Corrected output: within 3 cents of target note for held notes >200ms
- [ ] Formant preservation (formant_shift=1.0): spectral centroid shift <100 Hz on vowels

### Performance
- [ ] CPU usage: <20% single-core on 2020 Mac Mini (or equivalent) at 44.1kHz / 1024-sample block
- [ ] CPU usage: <35% single-core at 48kHz / 256-sample block (real-time headroom)
- [ ] Memory: <50 MB RSS during processing
- [ ] No allocations in audio callback (verified with jemalloc profiling)

### Latency
- [ ] End-to-end latency: <20ms with FFT size 1024 at 44.1kHz
- [ ] Latency reported correctly to host (VST3/CLAP latency port)
- [ ] Input-output alignment within 1 sample at 48kHz

### Plugin Compatibility
- [ ] Loads and processes audio in at least 2 DAWs (e.g., Reaper, Logic)
- [ ] Parameter automation recorded and replayed accurately
- [ ] Preset load/save: 100% round-trip fidelity (all parameters preserved)
- [ ] Bypass: zero-gain bypass produces identical output to direct signal (<0.1 dB difference)

### Editor
- [ ] Pitch visualization: displays tracked pitch with <10ms visual latency
- [ ] Parameter changes reflected in audio within one block
- [ ] Virtual keyboard responds to mouse within 50ms
- [ ] UI scales at 100% and 200% (DPI)

### Reliability
- [ ] No crashes on silence input (all-zero samples)
- [ ] No crashes on DC offset input
- [ ] No crashes on sample rate changes (44.1kHz, 48kHz, 96kHz)
- [ ] No audio glitches during parameter changes at audio rate (1000 changes/sec stress test)

---

## ALGORITHMIC DETAILS (Summary)

| Algorithm | Location | Complexity | Notes |
|-----------|----------|------------|-------|
| YIN pitch detection | `tunez-core/pitch/yin.rs` | O(N log N) via FFT | Use FFT-YIN variant for speed |
| LPC formant analysis | `tunez-core/dsp/formant.rs` | O(N * order) | Burg or autocorrelation method |
| Phase-locked vocoder | `tunez-core/dsp/pitch_shift.rs` | O(N log N) per block | Harmonic phase tracking |
| Parameter smoothing | `tunez-core/params/smoothed.rs` | O(1) per param | Exponential moving average |
| Scale degree lookup | `tunez-core/musical/scale.rs` | O(1) | Precomputed semitone table |
| Viterbi tracking | `tunez-core/tracking/viterbi.rs` | O(states * transitions) | Optional, for hard vocals |
| Flex-tune threshold | `tunez-core/correction/flex_tune.rs` | O(1) | Simple cents comparison |
| Retune crossfade | `tunez-core/correction/retune.rs` | O(1) | Per-block exponential blend |

---

## DEPENDENCIES (planned)

```toml
# tunez-core
[dependencies]
approx = "0.5"          # floating point comparison with tolerance
bytemuck = "1.14"       # safe buffer transmutation
num-complex = "0.4"     # complex numbers for FFT
smallvec = "1.11"       # stack-allocated small vectors
itertools = "0.12"      # iterator utilities
rand = "0.8"            # for humanize randomization (seeded)

# tunez-dsp
[dependencies]
tunez-core = { path = "tunez-core" }
portaudio = "0.10"      # or cpal for standalone
crossbeam = "0.8"       # thread communication
parking_lot = "0.12"    # fast mutex for real-time

# tunez-plugin
[dependencies]
tunez-core = { path = "tunez-core" }
tunez-dsp = { path = "tunez-dsp" }
vst3 = "0.6"            # VST3 host framework
clap = "0.15"           # CLAP host framework

# tunez-editor
[dependencies]
tunez-core = { path = "tunez-core" }
iced = "0.12"           # Rust GUI framework (lightweight, pure Rust)
# or: winit + raw-window-handle for custom rendering

# tunez-tools
[dependencies]
tunez-core = { path = "tunez-core" }
hound = "3.5"           # WAV file reading
serde = "1.0"           # preset serialization
serde_json = "1.0"      # preset serialization
```

---

## DATA STRUCTURES (Key Types)

```rust
// Core audio sample format
pub type AudioSample = f32;
pub type AudioBuffer = Vec<AudioSample>;

// Frequency representation
pub struct Pitch {
    pub hz: f32,          // frequency in Hz
    pub semitone: f32,    // semitone number (69 = A440)
    pub cents: f32,       // cents off from nearest semitone (-50 to 50)
    pub confidence: f32,  // 0.0 (uncertain) to 1.0 (certain)
}

// Musical note
pub struct Note {
    pub name: NoteName,   // C4, D#5, etc.
    pub hz: f32,          // frequency in Hz
    pub semitone: i8,     // MIDI note number
}

// Musical scale
pub struct Scale {
    pub intervals: [i8; 12],  // which semitone degrees are present
    pub name: ScaleName,       // major, minor, chromatic, etc.
}

// Processing state (owned by audio thread)
pub struct DSPState {
    pub pitch_detector: PitchDetector,
    pub tracker: PitchTracker,
    pub correction: CorrectionEngine,
    pub pitch_shifter: PitchShifter,
    pub formant_processor: FormantProcessor,
    pub mixer: DryWetMixer,
    pub scheduler: ParameterScheduler,
    pub latency: LatencyManager,
}

// Parameter model (shared between GUI and audio thread)
pub struct ParamModel {
    // See section 5 above
}

// Plugin state (shared via Arc)
pub struct PluginState {
    pub params: Arc<ParamModel>,
    pub dsp: Arc<Mutex<DSPState>>,
    pub presets: Arc<PresetStore>,
}
```

---

## END

This is the engineering specification. Implementation follows in the subsequent sessions, proceeding through the phases outlined in section 7.
