//! Record 5 seconds from the default microphone and write a 16 kHz mono
//! PCM16 WAV file.
//!
//! `cargo run -p vuho-audio --example record_wav -- /tmp/out.wav`

use std::time::{Duration, Instant};

use vuho_audio::{start_capture, CaptureConfig};

fn main() {
    let out_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/tmp/vuho-record.wav".to_string());

    let (handle, chunks) = start_capture(&CaptureConfig::default()).expect("start capture");
    println!("Recording 5s to {out_path}...");

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: vuho_audio::OUTPUT_SAMPLE_RATE,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(&out_path, spec).expect("create wav writer");

    let deadline = Instant::now() + Duration::from_secs(5);
    while Instant::now() < deadline {
        match chunks.recv_timeout(Duration::from_millis(200)) {
            Ok(chunk) => {
                for s in chunk {
                    #[allow(clippy::cast_possible_truncation)] // clamped to [-1,1] * i16::MAX above
                    let v = (s.clamp(-1.0, 1.0) * f32::from(i16::MAX)) as i16;
                    writer.write_sample(v).expect("write sample");
                }
            }
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => {}
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
        }
    }

    handle.stop();
    writer.finalize().expect("finalize wav");
    println!("Wrote {out_path}");
}
