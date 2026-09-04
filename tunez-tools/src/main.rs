// tunez-tools: offline pitch analysis tool for testing and evaluation.
//
// Usage:
//   tunez-analyze <input.wav> [--output <output.csv>] [--metrics] [--verbose]
//   tunez-analyze <input.wav> --compare <reference.csv> [--metrics]
//
// Modes:
//   Standard: reads WAV, outputs timestamped pitch estimates as CSV
//   --metrics: prints summary statistics to stderr after CSV
//   --compare: loads a reference CSV and computes error metrics

use std::io::Write;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut input_path: Option<String> = None;
    let mut output_path: Option<String> = None;
    let mut compare_path: Option<String> = None;
    let mut show_metrics = false;
    let mut verbose = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--output" | "-o" => {
                i += 1;
                if i < args.len() {
                    output_path = Some(args[i].clone());
                }
            }
            "--compare" | "-c" => {
                i += 1;
                if i < args.len() {
                    compare_path = Some(args[i].clone());
                }
            }
            "--metrics" | "-m" => {
                show_metrics = true;
            }
            "--verbose" | "-v" => {
                verbose = true;
            }
            "--help" | "-h" => {
                println!("tunez-analyze: offline pitch analysis tool");
                println!();
                println!("Usage: tunez-analyze <input.wav> [OPTIONS]");
                println!();
                println!("Options:");
                println!("  -o, --output <file>   Write CSV output to file");
                println!("  -c, --compare <file>  Compare against reference CSV");
                println!("  -m, --metrics         Show summary statistics");
                println!("  -v, --verbose         Verbose output to stderr");
                println!("  -h, --help            Show this help");
                std::process::exit(0);
            }
            _ => {
                if input_path.is_none() {
                    input_path = Some(args[i].clone());
                }
            }
        }
        i += 1;
    }

    let input_path = match input_path {
        Some(p) => p,
        None => {
            eprintln!("Usage: tunez-analyze <input.wav> [--output <output.csv>] [--metrics]");
            std::process::exit(1);
        }
    };

    let start = Instant::now();

    let (sample_rate, samples) = match read_wav(&input_path) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Error reading WAV file '{}': {}", input_path, e);
            std::process::exit(1);
        }
    };

    if verbose {
        eprintln!(
            "Read {} samples at {} Hz from {}",
            samples.len(),
            sample_rate,
            input_path
        );
    }

    let results = process_samples(sample_rate, &samples, verbose);

    let elapsed = start.elapsed();
    if verbose {
        eprintln!(
            "Pitch detection took {:.2} ms for {} samples ({} xRt)",
            elapsed.as_secs_f64() * 1000.0,
            samples.len(),
            (samples.len() as f64 / sample_rate as f64 / elapsed.as_secs_f64())
        );
    }

    // Write CSV
    if let Some(ref path) = output_path {
        match write_csv(path, &results) {
            Ok(()) => {
                if verbose {
                    eprintln!("Wrote {} estimates to {}", results.len(), path);
                }
            }
            Err(e) => {
                eprintln!("Error writing CSV '{}': {}", path, e);
                std::process::exit(1);
            }
        }
    }

    // If no output file specified, print CSV to stdout
    if output_path.is_none() {
        println!("timestamp_s,frequency_hz,confidence,voiced");
        for (ts, freq, conf, voiced) in &results {
            println!("{:.6},{:.4},{:.4},{}", ts, freq, conf, voiced);
        }
    }

    // Metrics
    if show_metrics {
        print_metrics(&results);
    }

    // Compare against reference
    if let Some(ref ref_path) = compare_path {
        match load_reference(ref_path) {
            Ok(reference) => {
                print_comparison(&results, &reference);
            }
            Err(e) => {
                eprintln!("Warning: could not load reference '{}': {}", ref_path, e);
            }
        }
    }
}

// ─── WAV Reader ──────────────────────────────────────────────────────────────

fn read_wav(path: &str) -> Result<(f32, Vec<f32>), String> {
    let mut reader = hound::WavReader::open(path).map_err(|e| format!("Hound error: {}", e))?;
    let sample_rate = reader.spec().sample_rate as f32;
    let bits_per_sample = reader.spec().bits_per_sample;

    if bits_per_sample != 16 && bits_per_sample != 24 && bits_per_sample != 32 {
        return Err(format!(
            "Unsupported bit depth: {} (need 16/24/32)",
            bits_per_sample
        ));
    }

    let samples: Vec<f32> = match bits_per_sample {
        16 => reader
            .samples::<i16>()
            .filter_map(|s| s.ok())
            .map(|s: i16| s as f32 / 32768.0)
            .collect(),
        24 => {
            let raw: Vec<i32> = reader
                .samples::<i32>()
                .filter_map(|s| s.ok())
                .collect();
            raw.into_iter()
                .map(|val| val as f32 / 8388608.0)
                .collect()
        }
        32 => reader
            .samples::<i32>()
            .filter_map(|s| s.ok())
            .map(|s: i32| s as f32 / 2147483647.0)
            .collect(),
        _ => return Err(format!("Unsupported bit depth: {}", bits_per_sample)),
    };

    Ok((sample_rate, samples))
}

// ─── Pitch Detection ─────────────────────────────────────────────────────────

fn process_samples(sample_rate: f32, samples: &[f32], verbose: bool) -> Vec<(f32, f32, f32, bool)> {
    use tunez_core::pitch::yin::{YinDetector, YinDetectorConfig};
    use tunez_core::pitch::postprocess::PostProcessor;
    use tunez_core::pitch::smoothing::TemporalSmoothing;

    let config = YinDetectorConfig {
        sample_rate,
        hop_size: 256,
        ..Default::default()
    };

    let mut detector = YinDetector::new(config);
    let mut post = PostProcessor::new_default();
    let mut smoothing = TemporalSmoothing::new_default(sample_rate);

    let hop_size = 256usize;
    let mut results: Vec<(f32, f32, f32, bool)> = Vec::new();

    let mut offset = 0;
    while offset + hop_size <= samples.len() {
        let chunk = &samples[offset..offset + hop_size];
        let estimates = detector.process_chunk(chunk);

        // Process all estimates from this chunk
        for raw in estimates {
            let yin_value = raw.confidence;
            let energy = estimate_energy(chunk);

            let processed = post.process(raw, yin_value, energy);
            let smoothed = smoothing.update(processed);

            let timestamp = offset as f32 / sample_rate + (results.len() as f32) * (256.0 / sample_rate);

            results.push((
                timestamp,
                smoothed.frequency_hz,
                smoothed.confidence,
                smoothed.voiced,
            ));

            if verbose {
                eprintln!(
                    "t={:.4}s: {:.2} Hz, conf={:.3}, voiced={}",
                    timestamp, smoothed.frequency_hz, smoothed.confidence, smoothed.voiced
                );
            }
        }

        offset += hop_size;
    }

    results
}

fn estimate_energy(chunk: &[f32]) -> f32 {
    let mut sum_sq = 0.0f32;
    for &s in chunk {
        sum_sq += s * s;
    }
    (sum_sq / chunk.len() as f32).sqrt().min(1.0)
}

// ─── CSV I/O ─────────────────────────────────────────────────────────────────

fn write_csv(path: &str, results: &[(f32, f32, f32, bool)]) -> Result<(), String> {
    let mut file = std::fs::File::create(path).map_err(|e| format!("Cannot create file: {}", e))?;

    writeln!(file, "timestamp_s,frequency_hz,confidence,voiced")
        .map_err(|e| format!("Write error: {}", e))?;

    for (ts, freq, conf, voiced) in results {
        writeln!(
            file,
            "{:.6},{:.4},{:.4},{}",
            ts, freq, conf, voiced
        )
        .map_err(|e| format!("Write error: {}", e))?;
    }

    Ok(())
}

/// Load a reference CSV for comparison (same format: timestamp_s,frequency_hz,confidence,voiced).
fn load_reference(path: &str) -> Result<Vec<(f32, f32)>, String> {
    let content = std::fs::read_to_string(path).map_err(|e| format!("Cannot read reference: {}", e))?;
    let mut ref_data = Vec::new();
    for (i, line) in content.lines().enumerate() {
        if i == 0 {
            continue; // Skip header
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() >= 2 {
            if let (Ok(ts), Ok(freq)) = (parts[0].parse::<f32>(), parts[1].parse::<f32>()) {
                ref_data.push((ts, freq));
            }
        }
    }
    Ok(ref_data)
}

// ─── Metrics ─────────────────────────────────────────────────────────────────

fn print_metrics(results: &[(f32, f32, f32, bool)]) {
    let total = results.len();
    if total == 0 {
        eprintln!("No estimates to analyze.");
        return;
    }

    let voiced_count = results.iter().filter(|&r| r.3).count();
    let unvoiced_count = total - voiced_count;

    let voiced_results: Vec<_> = results.iter().filter(|&r| r.3 && r.1 > 0.0).collect();

    eprintln!("\n=== Pitch Detection Metrics ===");
    eprintln!("Total estimates:     {}", total);
    eprintln!("Voiced frames:       {} ({:.1}%)", voiced_count, voiced_count as f64 / total as f64 * 100.0);
    eprintln!("Unvoiced frames:     {} ({:.1}%)", unvoiced_count, unvoiced_count as f64 / total as f64 * 100.0);

    if !voiced_results.is_empty() {
        let freqs: Vec<f32> = voiced_results.iter().map(|r| r.1).collect();
        let mean_freq = freqs.iter().sum::<f32>() / freqs.len() as f32;

        let mut sorted_freqs = freqs.clone();
        sorted_freqs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median_freq = sorted_freqs[sorted_freqs.len() / 2];

        let variance = freqs.iter().map(|f| (f - mean_freq).powi(2)).sum::<f32>() / freqs.len() as f32;
        let std_freq = variance.sqrt();

        eprintln!("\nVoiced frequency stats:");
        eprintln!("  Mean:              {:.2} Hz", mean_freq);
        eprintln!("  Median:            {:.2} Hz", median_freq);
        eprintln!("  Std dev:           {:.2} Hz", std_freq);
        eprintln!("  Min:               {:.2} Hz", sorted_freqs.first().copied().unwrap_or(0.0));
        eprintln!("  Max:               {:.2} Hz", sorted_freqs.last().copied().unwrap_or(0.0));

        // Confidence stats
        let confs: Vec<f32> = voiced_results.iter().map(|r| r.2).collect();
        let mean_conf = confs.iter().sum::<f32>() / confs.len() as f32;
        eprintln!("\nConfidence stats:");
        eprintln!("  Mean confidence:   {:.3}", mean_conf);
    }
}

fn print_comparison(results: &[(f32, f32, f32, bool)], reference: &[(f32, f32)]) {
    // Simple nearest-reference lookup for each result
    let mut total_error_cents = 0.0;
    let mut error_count = 0;
    let mut octave_errors = 0;

    for (ts, freq, _, voiced) in results {
        if !voiced || *freq <= 0.0 {
            continue;
        }

        // Find nearest reference by timestamp
        if let Some((ref_ts, ref_freq)) = find_nearest_reference(*ts, reference) {
            if ref_freq <= 0.0 {
                continue;
            }

            // Compute cents error
            let cents = 1200.0 * (freq / ref_freq).log2();
            total_error_cents += cents.abs();
            error_count += 1;

            if cents.abs() > 300.0 {
                octave_errors += 1;
            }
        }
    }

    if error_count > 0 {
        let mean_error = total_error_cents / error_count as f32;
        eprintln!("\n=== Comparison with Reference ===");
        eprintln!("Reference points:      {}", reference.len());
        eprintln!("Matched estimates:     {}", error_count);
        eprintln!("Mean absolute error:   {:.1} cents", mean_error);
        eprintln!("Octave errors:         {}", octave_errors);
        eprintln!("Octave error rate:     {:.1}%", octave_errors as f64 / error_count as f64 * 100.0);
    }
}

fn find_nearest_reference(ts: f32, reference: &[(f32, f32)]) -> Option<(f32, f32)> {
    let mut best = None;
    let mut best_dist = f32::MAX;

    for &(r_ts, r_freq) in reference {
        let dist = (ts - r_ts).abs();
        if dist < best_dist && dist < 0.02 { // Within 20ms tolerance
            best_dist = dist;
            best = Some((r_ts, r_freq));
        }
    }

    best
}
