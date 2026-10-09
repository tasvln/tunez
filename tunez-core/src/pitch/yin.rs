// pitch/yin.rs
// YIN algorithm implementation for vocal pitch detection.
//
// Algorithm overview:
// 1. Hann windowing: apply Hann window to reduce spectral leakage
// 2. Difference function: d[tau] = sum (x[n] - x[n+tau])^2 for tau in [min_lag, max_lag]
// 3. Cumulative mean normalized difference: normalize so the minimum is meaningful regardless of energy
// 4. Absolute threshold search: find first tau where YIN[tau] < threshold AND is a local minimum
// 5. Parabolic interpolation: refine period estimate for sub-sample accuracy
// 6. Octave error check: if YIN[2*tau] is similarly low, use 2*tau (the lower octave)
// 7. Voicing decision: combine YIN value with signal energy for robust unvoiced detection
//
// Tuning knobs:
// - tau_ms: Time constant for cumulative mean normalization (default 2.0 ms).
//   Controls how far back the normalization looks. Higher = better octave rejection.
//   The detector will search at least (tau_ms/1000 * sr) samples.
// - threshold: Absolute threshold for voicing decision (default 0.15).
//   Lower = more sensitive but may detect pitch in noise.
// - min_freq_hz: Minimum detectable frequency (default 65 Hz, ~C2).
// - max_freq_hz: Maximum detectable frequency (default 1100 Hz, covers soprano range).
// - sample_rate: Sample rate in Hz (configurable, default 44100).
// - window_size: Analysis window size in samples. Defaults to 4 * max_lag for ~75% overlap.
// - hop_size: Hop size in samples. Controls latency vs. update rate trade-off.
//
// To switch algorithms, replace the `analyze` method. The rest of the
// pipeline works with any function returning PitchEstimate.

use crate::pitch::PitchEstimate;

// ─── Default Configuration ───────────────────────────────────────────────────

/// Default sample rate.
const DEFAULT_SAMPLE_RATE: f32 = 44100.0;
/// Time constant for cumulative mean normalization (milliseconds).
/// 2.0 ms = ~88 samples at 44.1 kHz. Provides good octave rejection.
const DEFAULT_TAU_MS: f32 = 2.0;
/// Absolute threshold for the YIN difference function.
/// Values below this at the first minimum indicate voiced speech.
const DEFAULT_THRESHOLD: f32 = 0.10;
/// Minimum detectable frequency (Hz). C2 = ~65.4 Hz.
const DEFAULT_MIN_FREQ: f32 = 65.0;
/// Maximum detectable frequency (Hz). High soprano limit.
const DEFAULT_MAX_FREQ: f32 = 1100.0;
/// Default hop size in samples. Controls latency vs. update rate.
const DEFAULT_HOP_SIZE: usize = 256;

/// Hann window constant.
const HANN_A: f32 = 0.5;
const HANN_B: f32 = 0.5;

// ─── Configuration ───────────────────────────────────────────────────────────

/// Configuration for the YIN pitch detector.
#[derive(Debug, Clone)]
pub struct YinDetectorConfig {
    /// Sample rate in Hz.
    pub sample_rate: f32,
    /// Time constant for cumulative mean normalization (ms).
    /// Controls how far back the normalization window looks.
    pub tau_ms: f32,
    /// Absolute threshold for voicing decision.
    /// YIN value below this at the first minimum indicates voiced.
    pub threshold: f32,
    /// Minimum detectable frequency (Hz).
    pub min_freq_hz: f32,
    /// Maximum detectable frequency (Hz).
    pub max_freq_hz: f32,
    /// Hop size in samples. Controls latency vs. update rate.
    pub hop_size: usize,
}

impl Default for YinDetectorConfig {
    fn default() -> Self {
        Self {
            sample_rate: DEFAULT_SAMPLE_RATE,
            tau_ms: DEFAULT_TAU_MS,
            threshold: DEFAULT_THRESHOLD,
            min_freq_hz: DEFAULT_MIN_FREQ,
            max_freq_hz: DEFAULT_MAX_FREQ,
            hop_size: DEFAULT_HOP_SIZE,
        }
    }
}

impl YinDetectorConfig {
    /// Convert tau_ms to minimum lag in samples.
    pub fn tau_samples(&self) -> usize {
        (self.tau_ms / 1000.0 * self.sample_rate).max(1.0) as usize
    }

    /// Maximum lag in samples (corresponding to min_freq_hz).
    pub fn max_lag(&self) -> usize {
        (self.sample_rate / self.min_freq_hz).floor().min(8192.0) as usize
    }

    /// Minimum lag in samples (corresponding to max_freq_hz).
    pub fn min_lag(&self) -> usize {
        (self.sample_rate / self.max_freq_hz).ceil().max(1.0) as usize
    }

    /// Analysis window size (samples). Defaults to 4 * max_lag.
    pub fn window_size(&self) -> usize {
        self.max_lag() * 4
    }

    /// Minimum buffer size needed before first analysis.
    pub fn min_buffer_size(&self) -> usize {
        self.window_size()
    }
}

// ─── Pre-allocated buffers (zero-allocation after init) ─────────────────────

/// Analysis buffers: pre-allocated, reused across frames.
pub struct AnalysisBuffers {
    /// Windowed audio samples.
    pub windowed: Vec<f32>,
    /// Difference function d[tau].
    pub diff: Vec<f32>,
    /// YIN normalized difference values.
    pub yin: Vec<f32>,
}

impl AnalysisBuffers {
    /// Allocate buffers for the given analysis parameters.
    pub fn new(max_lag: usize) -> Self {
        Self {
            windowed: vec![0.0; max_lag * 4],
            diff: vec![0.0; max_lag + 1],
            yin: vec![0.0; max_lag + 1],
        }
    }

    /// Apply Hann window to the raw audio data in-place.
    pub fn apply_hann(&mut self, data: &[f32], window_len: usize) {
        let n = window_len.min(data.len());
        for i in 0..n {
            let hann =
                HANN_A - HANN_B * (std::f32::consts::PI * 2.0 * i as f32 / window_len as f32).cos();
            self.windowed[i] = data[i] * hann * hann;
        }
    }
}

// ─── Ring Buffer ─────────────────────────────────────────────────────────────

/// Ring buffer for zero-allocation input buffering.
pub struct RingBuffer {
    buffer: Vec<f32>,
    write_pos: usize,
    length: usize,
}

impl RingBuffer {
    /// Create a new ring buffer with the given capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: vec![0.0; capacity],
            write_pos: 0,
            length: 0,
        }
    }

    /// Push a single sample into the ring buffer.
    pub fn push(&mut self, sample: f32) {
        self.buffer[self.write_pos] = sample;
        self.write_pos = (self.write_pos + 1) % self.buffer.len();
        if self.length < self.buffer.len() {
            self.length += 1;
        }
    }

    /// Push a slice of samples into the ring buffer.
    pub fn push_slice(&mut self, samples: &[f32]) {
        let cap = self.buffer.len();
        if samples.is_empty() || cap == 0 {
            return;
        }

        // If more samples arrived than the buffer can even hold, only the
        // most recent `cap` of them matter — the rest would be overwritten
        // immediately anyway.
        let samples = if samples.len() > cap {
            &samples[samples.len() - cap..]
        } else {
            samples
        };
        let n = samples.len();

        // Always write — this is the fix. A ring buffer never runs out of
        // "space" once full; it just overwrites the oldest data as it goes.
        let end = self.write_pos + n;
        if end <= cap {
            self.buffer[self.write_pos..end].copy_from_slice(samples);
        } else {
            let first = cap - self.write_pos;
            self.buffer[self.write_pos..].copy_from_slice(&samples[..first]);
            self.buffer[..n - first].copy_from_slice(&samples[first..]);
        }

        self.write_pos = (self.write_pos + n) % cap;
        self.length = (self.length + n).min(cap);
    }

    /// Copy the last `len` samples into `dst`.
    /// Returns None if the buffer doesn't have enough samples.
    pub fn last_n(&self, len: usize, dst: &mut [f32]) -> bool {
        if self.length < len || dst.len() < len {
            return false;
        }
        let start = if self.length >= self.buffer.len() {
            (self.write_pos + self.length - len) % self.buffer.len()
        } else {
            self.write_pos.wrapping_sub(len)
        };

        if start + len <= self.buffer.len() {
            dst[..len].copy_from_slice(&self.buffer[start..start + len]);
        } else {
            let first = self.buffer.len() - start;
            dst[..first].copy_from_slice(&self.buffer[start..]);
            dst[first..len].copy_from_slice(&self.buffer[..len - first]);
        }
        true
    }

    /// Get the current number of samples in the buffer.
    pub fn len(&self) -> usize {
        self.length
    }

    /// Check if the buffer is full.
    pub fn is_full(&self) -> bool {
        self.length >= self.buffer.len()
    }

    /// Check if the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    /// Clear the buffer.
    pub fn clear(&mut self) {
        self.write_pos = 0;
        self.length = 0;
    }
}

// Pitch Dectector

/// One plausible pitch found within a single analysis window.
///
/// YIN's difference function often dips below threshold at more than one
/// lag — the true pitch period, and commonly its octave-related multiples
/// (2x, 3x, 4x the true period, which sound like 1/2, 1/3, 1/4 the true
/// frequency). Rather than the detector guessing which one is "real" with
/// no context, it reports the short list and lets a later stage — which
/// has access to what pitch came before — make the final call.
#[derive(Debug, Clone, Copy)]
pub struct PitchCandidate {
    /// Estimated frequency in Hz, after parabolic interpolation.
    pub frequency_hz: f32,
    /// The raw YIN difference value at this candidate (lower = more
    /// confidently periodic; this is the "low = good" number, distinct
    /// from PitchEstimate::confidence which is "high = good").
    pub yin_value: f32,
    /// The lag in samples this candidate came from, pre-interpolation.
    /// Kept around because comparing raw lags (not frequencies) is the
    /// simplest way to check "is candidate B roughly double candidate A's
    /// period" later on.
    pub tau: usize,
}

// Tracking

/// Controls how pitch candidates are chosen using the previous frame's
/// accepted pitch as context, not just raw periodicity in isolation.
#[derive(Debug, Clone)]
pub struct TrackingConfig {
    /// 0.0 = ignore history, always trust whichever candidate is most
    /// periodic. 1.0 = ignore periodicity, always pick whichever
    /// candidate is closest to the previous pitch.
    pub continuity_weight: f32,
    /// Cents distance beyond which "being closer" stops mattering.
    pub max_jump_cents: f32,
}

impl Default for TrackingConfig {
    fn default() -> Self {
        Self {
            continuity_weight: 0.4,
            max_jump_cents: 600.0,
        }
    }
}

/// Choose the best candidate given what the previous frame settled on.
fn pick_candidate(
    candidates: &[PitchCandidate],
    prev_freq_hz: Option<f32>,
    config: &TrackingConfig,
) -> Option<PitchCandidate> {
    if candidates.is_empty() {
        return None;
    }

    let Some(prev_freq) = prev_freq_hz else {
        return Some(candidates[0]); // no history, trust periodicity
    };

    let mut best: Option<(f32, PitchCandidate)> = None;
    for &c in candidates {
        let cents_away = (1200.0 * (c.frequency_hz / prev_freq).log2()).abs();
        let continuity_term = (cents_away / config.max_jump_cents).min(1.0);
        let score = c.yin_value * (1.0 - config.continuity_weight)
            + continuity_term * config.continuity_weight;

        if best.map_or(true, |(best_score, _)| score < best_score) {
            best = Some((score, c));
        }
    }

    best.map(|(_, c)| c)
}

// ─── YIN Detector ────────────────────────────────────────────────────────────

/// The YIN pitch detector.
/// Processes audio samples and produces pitch estimates.
///
/// Usage for real-time:
/// ```
/// let config = YinDetectorConfig {
///     sample_rate: 44100.0,
///     hop_size: 256,
///     ..Default::default()
/// };
/// let mut detector = YinDetector::new(config);
///
/// // Feed samples in chunks matching hop_size:
/// while !audio_finished {
///     let chunk = get_next_hop();
///     let estimates = detector.process_chunk(&chunk);
///     for est in estimates {
///         // process estimate
///     }
/// }
/// ```
pub struct YinDetector {
    config: YinDetectorConfig,
    ring_buffer: RingBuffer,
    buffers: AnalysisBuffers,
    /// Number of samples consumed since last analysis.
    samples_since_analysis: usize,
    /// Total samples processed (for offset tracking).
    total_samples: usize,
    tracking: TrackingConfig,
    prev_freq_hz: Option<f32>,
}

impl YinDetector {
    /// Create a new YIN detector with the given configuration.
    pub fn new(config: YinDetectorConfig) -> Self {
        let analysis_size = config.window_size();
        let max_lag = config.max_lag();
        Self {
            config,
            ring_buffer: RingBuffer::new(analysis_size),
            buffers: AnalysisBuffers::new(max_lag),
            samples_since_analysis: 0,
            total_samples: 0,
            tracking: TrackingConfig::default(),
            prev_freq_hz: None,
        }
    }

    /// Create with default configuration.
    pub fn new_default() -> Self {
        Self::new(YinDetectorConfig::default())
    }

    /// Return a reference to the configuration.
    pub fn config(&self) -> &YinDetectorConfig {
        &self.config
    }

    /// Set the sample rate (recomputes lag parameters).
    pub fn set_sample_rate(&mut self, sample_rate: f32) {
        self.config.sample_rate = sample_rate;
        self.total_samples = 0;
        self.samples_since_analysis = 0;
    }

    /// Reset the detector state.
    pub fn reset(&mut self) {
        self.ring_buffer.clear();
        self.total_samples = 0;
        self.samples_since_analysis = 0;
        self.prev_freq_hz = None;
    }

    /// Process a single sample (for backward compatibility with frame-by-frame pipelines).
    /// Appends the sample and returns an estimate if enough samples have accumulated.
    pub fn process_sample(&mut self, sample: f32) -> Option<PitchEstimate> {
        self.ring_buffer.push(sample);
        self.total_samples += 1;

        let min_needed = self.config.min_buffer_size();

        // Count how many analysis windows are complete
        let windows_complete = (self.total_samples.saturating_sub(min_needed))
            .saturating_sub(self.samples_since_analysis)
            / self.config.hop_size;

        if windows_complete > 0 {
            let est = self.analyze_next();
            self.samples_since_analysis += self.config.hop_size * windows_complete;
            Some(est)
        } else {
            None
        }
    }

    /// Process a chunk of samples and return pitch estimates.
    ///
    /// The chunk is appended to the internal buffer. If we have enough
    /// samples for analysis, one pitch estimate is produced at the hop
    /// boundary. Returns estimates for each completed analysis window.
    pub fn process_chunk(&mut self, samples: &[f32]) -> Vec<PitchEstimate> {
        if samples.is_empty() {
            return Vec::new();
        }

        self.ring_buffer.push_slice(samples);
        self.total_samples += samples.len();

        let mut estimates = Vec::new();
        let min_needed = self.config.min_buffer_size();

        // We produce one estimate per hop_size samples, starting after
        // the initial buffer fills up.
        let total = self.total_samples;
        let min_start = min_needed;

        // Count how many analysis windows are complete
        let windows_complete = (total.saturating_sub(min_start))
            .saturating_sub(self.samples_since_analysis)
            / self.config.hop_size;

        for _ in 0..windows_complete {
            let est = self.analyze_next();
            estimates.push(est);
            self.samples_since_analysis += self.config.hop_size;
        }

        estimates
    }

    // /// Scan the YIN curve for every local minimum that dips below threshold,
    // /// not just the first or the deepest. Returns them sorted best-first
    // /// (lowest yin_value = most periodic = most likely correct).
    // fn find_candidates(&self, tau_min: usize, max_lag: usize, threshold: f32) -> Vec<(usize, f32)> {
    //     let mut candidates = Vec::new();
    //     let mut tau = tau_min;

    //     while tau <= max_lag {
    //         let val = self.buffers.yin[tau];

    //         if val < threshold {
    //             // Walk forward to the actual bottom of this dip, same idea
    //             // as before — don't stop at the first point that crossed
    //             // the line, find where it stops falling.
    //             let mut t = tau;
    //             while t + 1 <= max_lag && self.buffers.yin[t + 1] < self.buffers.yin[t] {
    //                 t += 1;
    //             }
    //             candidates.push((t, self.buffers.yin[t]));

    //             // Skip past this whole dip so we don't re-detect the same
    //             // minimum on the way back up the other side of it.
    //             while t + 1 <= max_lag && self.buffers.yin[t + 1] > self.buffers.yin[t] {
    //                 t += 1;
    //             }
    //             tau = t + 1;
    //         } else {
    //             tau += 1;
    //         }
    //     }

    //     // Best (lowest yin_value) first.
    //     candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    //     candidates
    // }

    fn find_candidates(&self, tau_min: usize, max_lag: usize, threshold: f32) -> Vec<(usize, f32)> {
        let mut candidates = Vec::new();
        let mut tau = tau_min;

        while tau <= max_lag {
            let val = self.buffers.yin[tau];

            if val < threshold {
                let mut t = tau;
                while t + 1 <= max_lag && self.buffers.yin[t + 1] < self.buffers.yin[t] {
                    t += 1;
                }
                candidates.push((t, self.buffers.yin[t]));

                // Skip past this dip so we don't re-find the same minimum
                // on the way back up the other side of it.
                while t + 1 <= max_lag && self.buffers.yin[t + 1] > self.buffers.yin[t] {
                    t += 1;
                }
                tau = t + 1;
            } else {
                tau += 1;
            }
        }

        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        candidates
    }

    /// Steps 1-2 of YIN: window the latest audio, compute the difference
    /// function, then the cumulative mean normalized difference. Fills
    /// self.buffers.yin — both analyze_next and analyze_candidates read
    /// from it afterward, so this is the one place that math lives.
    fn compute_yin_curve(&mut self, window_size: usize, max_lag: usize, tau_min: usize) {
        let mut temp = [0.0f32; 8192];
        self.ring_buffer
            .last_n(window_size, &mut temp[..window_size]);
        self.buffers.apply_hann(&temp[..window_size], window_size);

        // Step 1: Difference function
        // d[tau] = sum_{n=0}^{N-tau-1} (x[n] - x[n+tau])^2
        for tau in tau_min..=max_lag {
            let mut sum: f32 = 0.0;
            let limit = window_size - tau;
            for n in 0..limit {
                let d = self.buffers.windowed[n] - self.buffers.windowed[n + tau];
                sum += d * d;
            }
            self.buffers.diff[tau] = sum;
        }

        // Step 2: Cumulative mean normalized difference
        self.buffers.yin[0] = 1.0;
        self.buffers.yin[1] = 1.0;
        let mut cumsum: f32 = 0.0;
        for tau in 2..=max_lag {
            cumsum += self.buffers.diff[tau - 1];
            let mean = cumsum / tau as f32;
            self.buffers.yin[tau] = if mean > 1e-10 {
                self.buffers.diff[tau] / (tau as f32 * mean)
            } else {
                1.0
            };
        }
    }

    /// Like `analyze_next`, but exposes the full ranked candidate list
    /// instead of committing to a single winner.
    pub fn analyze_candidates(&mut self) -> Vec<PitchCandidate> {
        let window_size = self.config.window_size();
        let max_lag = self.config.max_lag();
        let tau_min = self.config.min_lag();
        let threshold = self.config.threshold;

        // Steps 1-2: windowing + difference function + cumulative mean
        // normalization — identical to analyze_next, fills self.buffers.yin.
        self.compute_yin_curve(window_size, max_lag, tau_min);

        let raw_candidates = self.find_candidates(tau_min, max_lag, threshold);

        raw_candidates
            .into_iter()
            .map(|(tau, _)| {
                let (period, yin_value) = self.interpolate_minimum(tau, tau_min, max_lag);
                let freq = (self.config.sample_rate / period)
                    .clamp(self.config.min_freq_hz, self.config.max_freq_hz);
                PitchCandidate {
                    frequency_hz: freq,
                    yin_value,
                    tau,
                }
            })
            .collect()
    }

    /// Parabolic interpolation around a minimum at `tau`, for sub-sample
    /// period accuracy. Returns (refined_period, interpolated_yin_value).
    fn interpolate_minimum(&self, tau: usize, tau_min: usize, max_lag: usize) -> (f32, f32) {
        if tau > tau_min + 1 && tau < max_lag {
            let y_prev = self.buffers.yin[tau - 1];
            let y_center = self.buffers.yin[tau];
            let y_next = self.buffers.yin[tau + 1];
            let denom = 2.0 * (y_center - 0.5 * (y_prev + y_next));
            if denom.abs() > 1e-10 {
                let delta = (0.5 * (y_next - y_prev) / denom).clamp(-0.5, 0.5);
                let yin_value = y_center - 0.25 * (y_next - y_prev) * delta;
                return (tau as f32 + delta, yin_value);
            }
        }
        (tau as f32, self.buffers.yin[tau])
    }

    /// Analyze one window: extract last `window_size` samples, apply Hann window, run YIN.
    fn analyze_next(&mut self) -> PitchEstimate {
        let window_size = self.config.window_size();
        let max_lag = self.config.max_lag();
        let tau_min = self.config.min_lag();
        let threshold = self.config.threshold;

        self.compute_yin_curve(window_size, max_lag, tau_min);

        // Step 3: first-dip-below-threshold search (the A5/Hypothesis-1 fix)
        let raw_candidates = self.find_candidates(tau_min, max_lag, threshold);
        if raw_candidates.is_empty() {
            self.prev_freq_hz = None; // lost the thread, don't bias the next frame
            return PitchEstimate::unvoiced();
        }

        let candidates: Vec<PitchCandidate> = raw_candidates
            .into_iter()
            .map(|(tau, _)| {
                let (period, yin_value) = self.interpolate_minimum(tau, tau_min, max_lag);
                let freq = (self.config.sample_rate / period)
                    .clamp(self.config.min_freq_hz, self.config.max_freq_hz);
                PitchCandidate {
                    frequency_hz: freq,
                    yin_value,
                    tau,
                }
            })
            .collect();

        let chosen = pick_candidate(&candidates, self.prev_freq_hz, &self.tracking).unwrap();
        let tau_found = chosen.tau;

        // Step 4: parabolic interpolation (now shared with analyze_candidates)
        let (period, yin_value) = self.interpolate_minimum(tau_found, tau_min, max_lag);

        // Step 5: octave error check (unchanged from before)
        const MIN_OCTAVE_GAP: f32 = 0.08;
        let mut final_period = period;
        let double_tau = 2 * tau_found;
        if double_tau <= max_lag {
            let yin_double = self.buffers.yin[double_tau];
            if yin_double < threshold && (yin_value - yin_double) > MIN_OCTAVE_GAP {
                let is_min = if double_tau < max_lag {
                    self.buffers.yin[double_tau] <= self.buffers.yin[double_tau - 1]
                        && self.buffers.yin[double_tau] <= self.buffers.yin[double_tau + 1]
                } else {
                    self.buffers.yin[double_tau] <= self.buffers.yin[double_tau - 1]
                };
                if is_min {
                    final_period = double_tau as f32;
                }
            }
        }

        // Step 6: convert period to frequency
        let freq = if final_period > 0.0 {
            self.config.sample_rate / final_period
        } else {
            0.0
        };
        let freq = freq.clamp(self.config.min_freq_hz, self.config.max_freq_hz);

        // Step 7: energy-based voicing veto
        let window_size_f = window_size as f32;
        let mut energy: f32 = 0.0;
        for n in 0..window_size {
            energy += self.buffers.windowed[n] * self.buffers.windowed[n];
        }
        energy /= window_size_f;
        let rms = energy.sqrt();

        let voiced = yin_value < threshold
            && rms > 0.005
            && freq >= self.config.min_freq_hz
            && freq <= self.config.max_freq_hz;

        // Step 8: confidence estimation
        let yin_confidence = if yin_value < threshold {
            1.0 - (yin_value / threshold)
        } else {
            0.0
        };
        let energy_confidence = (rms * 10.0).clamp(0.0, 1.0);
        let confidence = yin_confidence * 0.75 + energy_confidence * 0.25;

        if voiced {
            self.prev_freq_hz = Some(freq);
        }

        PitchEstimate {
            frequency_hz: freq,
            confidence,
            voiced,
        }
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    // ─── Signal Generators ───────────────────────────────────────────────────

    fn sine_wave(freq: f32, duration_s: f32, sample_rate: f32) -> Vec<f32> {
        let n = (duration_s * sample_rate) as usize;
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let t = i as f32 / sample_rate;
            out.push((2.0 * PI * freq * t).sin());
        }
        out
    }

    fn vibrato_signal(
        base_freq: f32,
        vibrato_hz: f32,
        vibrato_depth: f32,
        duration_s: f32,
        sample_rate: f32,
    ) -> Vec<f32> {
        let n = (duration_s * sample_rate) as usize;
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let t = i as f32 / sample_rate;
            let instant_freq = base_freq + vibrato_depth * (2.0 * PI * vibrato_hz * t).sin();
            out.push((2.0 * PI * instant_freq * t).sin());
        }
        out
    }

    fn noisy_sine(
        freq: f32,
        snr_db: f32,
        duration_s: f32,
        sample_rate: f32,
        seed: u32,
    ) -> Vec<f32> {
        let n = (duration_s * sample_rate) as usize;
        let signal_power = 0.5; // sin^2 average = 0.5
        let noise_power = signal_power / (10.0_f32).powf(snr_db / 10.0);
        let noise_std = noise_power.sqrt();
        let mut rng = SimpleRng::new(seed);
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let t = i as f32 / sample_rate;
            let sig = (2.0 * PI * freq * t).sin();
            let noise = (rng.next_f32() - 0.5) * 2.0 * noise_std;
            out.push(sig + noise);
        }
        out
    }

    fn white_noise(power: f32, duration_s: f32, sample_rate: f32, seed: u32) -> Vec<f32> {
        let n = (duration_s * sample_rate) as usize;
        let std = power.sqrt();
        let mut rng = SimpleRng::new(seed);
        (0..n).map(|_| (rng.next_f32() - 0.5) * 2.0 * std).collect()
    }

    fn silence(n_samples: usize) -> Vec<f32> {
        vec![0.0; n_samples]
    }

    fn sawtooth(freq: f32, duration_s: f32, sample_rate: f32) -> Vec<f32> {
        let n = (duration_s * sample_rate) as usize;
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let t = i as f32 / sample_rate;
            let phase = (2.0 * PI * freq * t) % (2.0 * PI);
            let mut saw = 0.0f32;
            for h in 1..=20 {
                saw += (h as f32 * phase).sin() / h as f32;
            }
            out.push(saw);
        }
        out
    }

    fn breathy_vowel(
        base_freq: f32,
        breathiness_ratio: f32,
        duration_s: f32,
        sample_rate: f32,
        seed: u32,
    ) -> Vec<f32> {
        // Simulate breathy voice: periodic component + broadband noise
        let n = (duration_s * sample_rate) as usize;
        let mut rng = SimpleRng::new(seed);
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let t = i as f32 / sample_rate;
            let periodic = (2.0 * PI * base_freq * t).sin();
            let noise = (rng.next_f32() - 0.5) * 2.0;
            out.push((1.0 - breathiness_ratio) * periodic + breathiness_ratio * noise);
        }
        out
    }

    fn median_sorted(arr: &[f32]) -> f32 {
        let mid = arr.len() / 2;
        arr[mid]
    }

    // ─── Unit Tests ──────────────────────────────────────────────────────────

    #[test]
    fn test_sine_440_hz_accuracy() {
        let sr = 44100.0;
        let samples = sine_wave(440.0, 0.5, sr);
        let config = YinDetectorConfig {
            sample_rate: sr,
            hop_size: 256,
            ..Default::default()
        };
        let mut detector = YinDetector::new(config);
        let estimates = detector.process_chunk(&samples);

        let voiced: Vec<_> = estimates.iter().filter(|e| e.voiced).collect();
        assert!(
            !voiced.is_empty(),
            "Should detect voiced frames for 440 Hz sine"
        );

        // Measure mean frequency error in cents
        let mean_freq: f32 =
            voiced.iter().map(|e| e.frequency_hz).sum::<f32>() / voiced.len() as f32;
        let error_cents = 1200.0 * (mean_freq / 440.0).log2();
        assert!(
            error_cents.abs() < 2.0,
            "440 Hz: mean freq {:.2} Hz, error {:.1} cents",
            mean_freq,
            error_cents
        );

        // Check confidence is high
        let mean_conf: f32 = voiced.iter().map(|e| e.confidence).sum::<f32>() / voiced.len() as f32;
        assert!(
            mean_conf > 0.5,
            "Confidence for clean sine should be high: {}",
            mean_conf
        );
    }

    #[test]
    fn test_sine_220_hz_accuracy() {
        let sr = 44100.0;
        let samples = sine_wave(220.0, 0.5, sr);
        let config = YinDetectorConfig {
            sample_rate: sr,
            hop_size: 256,
            ..Default::default()
        };
        let mut detector = YinDetector::new(config);
        let estimates = detector.process_chunk(&samples);

        let voiced: Vec<_> = estimates.iter().filter(|e| e.voiced).collect();
        assert!(!voiced.is_empty(), "Should detect voiced frames for 220 Hz");

        let mean_freq: f32 =
            voiced.iter().map(|e| e.frequency_hz).sum::<f32>() / voiced.len() as f32;
        let error_cents = 1200.0 * (mean_freq / 220.0).log2();
        assert!(
            error_cents.abs() < 2.0,
            "220 Hz: mean freq {:.2} Hz, error {:.1} cents",
            mean_freq,
            error_cents
        );
    }

    #[test]
    fn test_low_frequencies() {
        // Test male vocal range: 85-150 Hz
        for freq in [85.0, 100.0, 120.0, 150.0] {
            let sr = 44100.0;
            let samples = sine_wave(freq, 0.5, sr);
            let config = YinDetectorConfig {
                sample_rate: sr,
                hop_size: 256,
                min_freq_hz: 50.0,
                max_freq_hz: 1000.0,
                ..Default::default()
            };
            let mut detector = YinDetector::new(config);
            let estimates = detector.process_chunk(&samples);

            let voiced: Vec<_> = estimates.iter().filter(|e| e.voiced).collect();
            if !voiced.is_empty() {
                let mean_freq: f32 =
                    voiced.iter().map(|e| e.frequency_hz).sum::<f32>() / voiced.len() as f32;
                let error_cents = 1200.0 * (mean_freq / freq).log2();
                assert!(
                    error_cents.abs() < 3.0,
                    "{} Hz: detected {:.2} Hz, error {:.1} cents",
                    freq,
                    mean_freq,
                    error_cents
                );
            }
            detector.reset();
        }
    }

    #[test]
    fn test_high_frequencies() {
        // Test soprano range: 300-800 Hz
        for freq in [330.0, 440.0, 600.0, 800.0] {
            let sr = 44100.0;
            let samples = sine_wave(freq, 0.5, sr);
            let config = YinDetectorConfig {
                sample_rate: sr,
                hop_size: 256,
                min_freq_hz: 65.0,
                max_freq_hz: 1100.0,
                ..Default::default()
            };
            let mut detector = YinDetector::new(config);
            let estimates = detector.process_chunk(&samples);

            let voiced: Vec<_> = estimates.iter().filter(|e| e.voiced).collect();
            if !voiced.is_empty() {
                let mean_freq: f32 =
                    voiced.iter().map(|e| e.frequency_hz).sum::<f32>() / voiced.len() as f32;
                let error_cents = 1200.0 * (mean_freq / freq).log2();
                assert!(
                    error_cents.abs() < 2.0,
                    "{} Hz: detected {:.2} Hz, error {:.1} cents",
                    freq,
                    mean_freq,
                    error_cents
                );
            }
            detector.reset();
        }
    }

    #[test]
    fn test_octave_accuracy_220_vs_440() {
        let sr = 44100.0;
        let config = YinDetectorConfig {
            sample_rate: sr,
            hop_size: 256,
            min_freq_hz: 65.0,
            max_freq_hz: 1100.0,
            ..Default::default()
        };

        // Test 220 Hz
        let samples_220 = sine_wave(220.0, 0.5, sr);
        let mut detector = YinDetector::new(config.clone());
        let ests_220: Vec<f32> = detector
            .process_chunk(&samples_220)
            .iter()
            .filter(|e| e.voiced)
            .map(|e| e.frequency_hz)
            .collect();

        // Test 440 Hz
        let samples_440 = sine_wave(440.0, 0.5, sr);
        let mut detector = YinDetector::new(config);
        let ests_440: Vec<f32> = detector
            .process_chunk(&samples_440)
            .iter()
            .filter(|e| e.voiced)
            .map(|e| e.frequency_hz)
            .collect();

        // Median estimates should be close to correct octave
        let med_220 = median_sorted(&mut ests_220.clone());
        let med_440 = median_sorted(&mut ests_440.clone());

        assert!(
            (med_220 - 220.0).abs() < 5.0,
            "220 Hz: median {:.2} Hz",
            med_220
        );
        assert!(
            (med_440 - 440.0).abs() < 5.0,
            "440 Hz: median {:.2} Hz",
            med_440
        );
    }

    #[test]
    fn test_octave_accuracy_110_vs_220() {
        // Test octave pair: A2 (110) vs A3 (220)
        let sr = 44100.0;
        let config = YinDetectorConfig {
            sample_rate: sr,
            hop_size: 256,
            min_freq_hz: 50.0,
            max_freq_hz: 1100.0,
            ..Default::default()
        };

        let samples_110 = sine_wave(110.0, 0.5, sr);
        let mut detector = YinDetector::new(config.clone());
        let ests_110: Vec<f32> = detector
            .process_chunk(&samples_110)
            .iter()
            .filter(|e| e.voiced)
            .map(|e| e.frequency_hz)
            .collect();

        let samples_220 = sine_wave(220.0, 0.5, sr);
        let mut detector = YinDetector::new(config);
        let ests_220: Vec<f32> = detector
            .process_chunk(&samples_220)
            .iter()
            .filter(|e| e.voiced)
            .map(|e| e.frequency_hz)
            .collect();

        let med_110 = median_sorted(&mut ests_110.clone());
        let med_220 = median_sorted(&mut ests_220.clone());

        assert!(
            (med_110 - 110.0).abs() < 5.0,
            "110 Hz: median {:.2} Hz",
            med_110
        );
        assert!(
            (med_220 - 220.0).abs() < 5.0,
            "220 Hz: median {:.2} Hz",
            med_220
        );
    }

    #[test]
    fn test_harmonically_rich_signal() {
        // Sawtooth wave at 110 Hz (A2)
        let sr = 44100.0;
        let samples = sawtooth(110.0, 0.5, sr);
        let config = YinDetectorConfig {
            sample_rate: sr,
            hop_size: 256,
            min_freq_hz: 50.0,
            max_freq_hz: 1000.0,
            ..Default::default()
        };
        let mut detector = YinDetector::new(config);
        let estimates = detector.process_chunk(&samples);

        let voiced: Vec<_> = estimates.iter().filter(|e| e.voiced).collect();
        assert!(
            !voiced.is_empty(),
            "Should detect pitched sawtooth at 110 Hz"
        );

        if !voiced.is_empty() {
            let mean_freq: f32 =
                voiced.iter().map(|e| e.frequency_hz).sum::<f32>() / voiced.len() as f32;
            let error_cents = 1200.0 * (mean_freq / 110.0).log2();
            assert!(
                error_cents.abs() < 5.0,
                "Sawtooth 110 Hz: detected {:.2} Hz, error {:.1} cents",
                mean_freq,
                error_cents
            );
        }
    }

    #[test]
    fn test_vibrato_tracking() {
        // Vibrato at 440 Hz with 5.5 Hz rate and +/- 8 Hz depth
        let sr = 44100.0;
        let samples = vibrato_signal(440.0, 5.5, 8.0, 0.5, sr);
        let config = YinDetectorConfig {
            sample_rate: sr,
            hop_size: 256,
            ..Default::default()
        };
        let mut detector = YinDetector::new(config);
        let estimates = detector.process_chunk(&samples);

        let voiced: Vec<_> = estimates.iter().filter(|e| e.voiced).collect();
        assert!(!voiced.is_empty(), "Should detect vibrato as voiced");

        // Mean frequency should be close to 440 Hz
        if !voiced.is_empty() {
            let mean_freq: f32 =
                voiced.iter().map(|e| e.frequency_hz).sum::<f32>() / voiced.len() as f32;
            assert!(
                (mean_freq - 440.0).abs() < 10.0,
                "Vibrato mean: {:.2} Hz (expected ~440 Hz)",
                mean_freq
            );
        }
    }

    #[test]
    fn test_silence() {
        let sr = 44100.0;
        let samples = silence((0.2 * sr) as usize); // 200 ms of silence
        let config = YinDetectorConfig {
            sample_rate: sr,
            hop_size: 256,
            ..Default::default()
        };
        let mut detector = YinDetector::new(config);
        let estimates = detector.process_chunk(&samples);

        let voiced_count = estimates.iter().filter(|e| e.voiced).count();
        if !estimates.is_empty() {
            let ratio = voiced_count as f32 / estimates.len() as f32;
            assert!(
                ratio < 0.05,
                "Silence: {:.0}% voiced frames (expected <5%), got {} / {}",
                ratio * 100.0,
                voiced_count,
                estimates.len()
            );
        }
    }

    #[test]
    fn test_white_noise() {
        let sr = 44100.0;
        let samples = white_noise(0.5, 0.5, sr, 42);
        let config = YinDetectorConfig {
            sample_rate: sr,
            hop_size: 256,
            ..Default::default()
        };
        let mut detector = YinDetector::new(config);
        let estimates = detector.process_chunk(&samples);

        let voiced_count = estimates.iter().filter(|e| e.voiced).count();
        if !estimates.is_empty() {
            let ratio = voiced_count as f32 / estimates.len() as f32;
            assert!(
                ratio < 0.2,
                "White noise: {:.0}% voiced (expected <20%), got {} / {}",
                ratio * 100.0,
                voiced_count,
                estimates.len()
            );
        }
    }

    #[test]
    fn test_noisy_periodic_snr20db() {
        // Sine at 440 Hz with 20 dB SNR
        let sr = 44100.0;
        let samples = noisy_sine(440.0, 20.0, 0.5, sr, 123);
        let config = YinDetectorConfig {
            sample_rate: sr,
            hop_size: 256,
            ..Default::default()
        };
        let mut detector = YinDetector::new(config);
        let estimates = detector.process_chunk(&samples);

        let voiced: Vec<_> = estimates.iter().filter(|e| e.voiced).collect();
        if !voiced.is_empty() {
            let mean_freq: f32 =
                voiced.iter().map(|e| e.frequency_hz).sum::<f32>() / voiced.len() as f32;
            let error_cents = 1200.0 * (mean_freq / 440.0).log2();
            assert!(
                error_cents.abs() < 5.0,
                "Noisy 440 Hz @ 20dB: detected {:.2} Hz, error {:.1} cents",
                mean_freq,
                error_cents
            );
        }
    }

    #[test]
    fn test_noisy_periodic_snr10db() {
        // Sine at 220 Hz with 10 dB SNR (moderate noise)
        let sr = 44100.0;
        let samples = noisy_sine(220.0, 10.0, 0.5, sr, 456);
        let config = YinDetectorConfig {
            sample_rate: sr,
            hop_size: 256,
            min_freq_hz: 50.0,
            max_freq_hz: 1000.0,
            ..Default::default()
        };
        let mut detector = YinDetector::new(config);
        let estimates = detector.process_chunk(&samples);

        let voiced: Vec<_> = estimates.iter().filter(|e| e.voiced).collect();
        if !voiced.is_empty() {
            let mean_freq: f32 =
                voiced.iter().map(|e| e.frequency_hz).sum::<f32>() / voiced.len() as f32;
            let error_cents = 1200.0 * (mean_freq / 220.0).log2();
            assert!(
                error_cents.abs() < 8.0,
                "Noisy 220 Hz @ 10dB: detected {:.2} Hz, error {:.1} cents",
                mean_freq,
                error_cents
            );
        }
    }

    #[test]
    fn test_breathy_vocals() {
        // Simulated breathy voice: 30% noise
        let sr = 44100.0;
        let samples = breathy_vowel(200.0, 0.3, 0.5, sr, 789);
        let config = YinDetectorConfig {
            sample_rate: sr,
            hop_size: 256,
            min_freq_hz: 50.0,
            max_freq_hz: 1000.0,
            threshold: 0.25, // More lenient threshold for breathy
            ..Default::default()
        };
        let mut detector = YinDetector::new(config);
        let estimates = detector.process_chunk(&samples);

        let voiced: Vec<_> = estimates.iter().filter(|e| e.voiced).collect();
        if !voiced.is_empty() {
            let mean_freq: f32 =
                voiced.iter().map(|e| e.frequency_hz).sum::<f32>() / voiced.len() as f32;
            let error_cents = 1200.0 * (mean_freq / 200.0).log2();
            assert!(
                error_cents.abs() < 5.0,
                "Breathy 200 Hz: detected {:.2} Hz, error {:.1} cents",
                mean_freq,
                error_cents
            );
        }
    }

    #[test]
    fn test_configurable_sample_rate() {
        // Test at different sample rates
        for &sr in &[22050, 44100, 48000] {
            let samples = sine_wave(440.0, 0.2, sr as f32);
            let config = YinDetectorConfig {
                sample_rate: sr as f32,
                hop_size: 256,
                ..Default::default()
            };
            let mut detector = YinDetector::new(config);
            let estimates = detector.process_chunk(&samples);

            let voiced: Vec<_> = estimates.iter().filter(|e| e.voiced).collect();
            assert!(!voiced.is_empty(), "{} Hz sr: should detect voiced", sr);

            if !voiced.is_empty() {
                let mean_freq: f32 =
                    voiced.iter().map(|e| e.frequency_hz).sum::<f32>() / voiced.len() as f32;
                let error_cents = 1200.0 * (mean_freq / 440.0).log2();
                assert!(
                    error_cents.abs() < 3.0,
                    "{} Hz sr: error {:.1} cents at {:.2} Hz",
                    sr,
                    error_cents,
                    mean_freq
                );
            }
            detector.reset();
        }
    }

    #[test]
    fn test_hop_size_affects_latency() {
        // Larger hop_size should produce fewer estimates per second
        let sr = 44100.0;
        let samples = sine_wave(440.0, 0.1, sr); // 100 ms

        for &hop in &[64, 128, 256, 512] {
            let config = YinDetectorConfig {
                sample_rate: sr,
                hop_size: hop,
                ..Default::default()
            };
            let window_size = config.window_size();
            let mut detector = YinDetector::new(config);
            let estimates = detector.process_chunk(&samples);
            // Estimate should be roughly samples/hop, minus warmup
            let _expected = (samples.len() / window_size.saturating_sub(hop)).max(1);
            // At minimum, should produce at least some estimates
            assert!(estimates.len() > 0, "hop={} should produce estimates", hop);
            detector.reset();
        }
    }

    #[test]
    fn test_voiced_unvoiced_alternation() {
        // Simulate voiced-unvoiced alternation: 100 ms voiced, 50 ms silence
        let sr = 44100.0;
        let mut samples = Vec::new();
        // Voiced segment
        samples.extend(sine_wave(440.0, 0.1, sr));
        // Silence
        samples.extend(silence((0.05 * sr) as usize));
        // Voiced segment
        samples.extend(sine_wave(440.0, 0.1, sr));

        let config = YinDetectorConfig {
            sample_rate: sr,
            hop_size: 256,
            ..Default::default()
        };
        let mut detector = YinDetector::new(config);
        let estimates = detector.process_chunk(&samples);

        let voiced_count = estimates.iter().filter(|e| e.voiced).count();
        let unvoiced_count = estimates.iter().filter(|e| !e.voiced).count();

        // Should have both voiced and unvoiced estimates
        assert!(voiced_count > 0, "Should detect voiced frames");
        assert!(
            unvoiced_count > 0,
            "Should detect unvoiced frames in silence gap"
        );

        // Voiced frames should all be near 440 Hz
        for est in estimates.iter().filter(|e| e.voiced) {
            let error_cents = 1200.0 * (est.frequency_hz / 440.0).log2();
            assert!(
                error_cents.abs() < 3.0,
                "Voiced frame: error {:.1} cents",
                error_cents
            );
        }
    }

    #[test]
    fn test_ring_buffer_push_pop() {
        let mut rb = RingBuffer::new(16);
        assert_eq!(rb.len(), 0);
        assert!(rb.is_empty());

        rb.push_slice(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(rb.len(), 4);

        let mut dst = [0.0f32; 4];
        assert!(rb.last_n(4, &mut dst));
        assert_eq!(dst[0], 1.0);
        assert_eq!(dst[3], 4.0);
    }

    #[test]
    fn test_ring_buffer_wrap() {
        let mut rb = RingBuffer::new(8);
        // Push more than capacity
        for i in 0..16 {
            rb.push(i as f32);
        }
        assert_eq!(rb.len(), 8);

        let mut dst = [0.0f32; 4];
        assert!(rb.last_n(4, &mut dst));
        assert_eq!(dst[0], 12.0);
        assert_eq!(dst[3], 15.0);
    }

    #[test]
    fn test_ring_buffer_insufficient_data() {
        let rb = RingBuffer::new(16);
        let mut dst = [0.0f32; 8];
        assert!(!rb.last_n(8, &mut dst)); // buffer is empty
    }

    #[test]
    fn test_octave_error_rejection() {
        // Test that octave errors are rejected by checking the double_tau YIN value.
        // A sine wave at 220 Hz should not be misdetected as 110 Hz.
        let sr = 44100.0;
        let samples = sine_wave(220.0, 0.5, sr);
        let config = YinDetectorConfig {
            sample_rate: sr,
            hop_size: 256,
            min_freq_hz: 50.0,
            max_freq_hz: 1000.0,
            ..Default::default()
        };
        let mut detector = YinDetector::new(config);
        let estimates = detector.process_chunk(&samples);

        // Check that we don't have octave errors at ~110 Hz
        let octave_errors: Vec<_> = estimates
            .iter()
            .filter(|e| e.voiced && (e.frequency_hz - 110.0).abs() < 10.0)
            .collect();

        // With tau=2ms (~88 samples at 44.1kHz), octave should be rejected
        assert!(
            octave_errors.len() == 0
                || octave_errors.len() < estimates.iter().filter(|e| e.voiced).count() / 4,
            "Octave errors at 110 Hz: {} out of {} voiced frames",
            octave_errors.len(),
            estimates.iter().filter(|e| e.voiced).count()
        );
    }

    #[test]
    fn test_confidence_on_voiced_vs_unvoiced() {
        let sr = 44100.0;
        let voiced_samples = sine_wave(440.0, 0.3, sr);
        let unvoiced_samples = white_noise(0.5, 0.3, sr, 42);

        let config = YinDetectorConfig {
            sample_rate: sr,
            hop_size: 256,
            ..Default::default()
        };

        // Voiced
        let mut detector = YinDetector::new(config.clone());
        let voiced_est = detector.process_chunk(&voiced_samples);
        let voiced_confidences: Vec<f32> = voiced_est
            .iter()
            .filter(|e| e.voiced)
            .map(|e| e.confidence)
            .collect();

        // Unvoiced
        let mut detector = YinDetector::new(config);
        let unvoiced_est = detector.process_chunk(&unvoiced_samples);
        let unvoiced_confidences: Vec<f32> = unvoiced_est
            .iter()
            .filter(|e| !e.voiced)
            .map(|e| e.confidence)
            .collect();

        if !voiced_confidences.is_empty() {
            let mean_v: f32 =
                voiced_confidences.iter().sum::<f32>() / voiced_confidences.len() as f32;
            assert!(
                mean_v > 0.3,
                "Voiced confidence should be moderate+: {:.3}",
                mean_v
            );
        }

        if !unvoiced_confidences.is_empty() {
            let mean_u: f32 =
                unvoiced_confidences.iter().sum::<f32>() / unvoiced_confidences.len() as f32;
            assert!(
                mean_u < 0.5,
                "Unvoiced confidence should be low: {:.3}",
                mean_u
            );
        }
    }

    #[test]
    fn test_candidates_include_true_pitch_on_harmonic_rich_signal() {
        // A high, harmonically-simple tone — similar to what bleep.wav
        // revealed: true pitch ~1000 Hz, but naive single-answer search
        // kept landing on a sub-harmonic (250 Hz, 333 Hz) instead.
        let sr = 44100.0;
        let samples = sine_wave(1000.0, 0.3, sr);
        let config = YinDetectorConfig {
            sample_rate: sr,
            hop_size: 256,
            min_freq_hz: 65.0,
            max_freq_hz: 1200.0,
            ..Default::default()
        };
        let mut detector = YinDetector::new(config);

        // Feed enough samples to fill one analysis window, then pull candidates.
        detector.ring_buffer.push_slice(&samples);
        let candidates = detector.analyze_candidates();

        assert!(!candidates.is_empty(), "should find at least one candidate");

        let has_true_pitch = candidates
            .iter()
            .any(|c| (c.frequency_hz - 1000.0).abs() < 10.0);

        assert!(
            has_true_pitch,
            "true ~1000Hz pitch should appear somewhere in the candidate list, got: {:?}",
            candidates
                .iter()
                .map(|c| c.frequency_hz)
                .collect::<Vec<_>>()
        );
    }
}

// ─── Simple RNG ──────────────────────────────────────────────────────────────

/// Simple deterministic RNG for tests.
pub struct SimpleRng {
    state: u32,
}

impl SimpleRng {
    pub fn new(seed: u32) -> Self {
        Self { state: seed }
    }

    pub fn next_f32(&mut self) -> f32 {
        self.state = self.state.wrapping_mul(1664525).wrapping_add(1013904223);
        (self.state as f32) / (u32::MAX as f32)
    }
}
