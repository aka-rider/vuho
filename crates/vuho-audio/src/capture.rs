//! The capture thread: owns the `!Send` `cpal::Stream` end to end (build,
//! play, pump, drop — all on `"vuho-audio-capture"`), per CONSTITUTION
//! rule 1 (one owner per resource) and rule 9 (the stopper owns the stop
//! signal: `CaptureHandle::stop` sets the flag *and* joins).

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, Stream, StreamConfig};
use crossbeam_channel::{bounded, Receiver, Sender};

use crate::resample::Resampler;
use crate::{AudioError, CaptureConfig};

/// Ring buffer capacity between the realtime audio callback and the pump
/// loop, in samples (interleaved). ~2s of stereo @ 48kHz.
const RING_CAPACITY: usize = 1 << 18;

/// How long the pump loop waits for new ring-buffer data before checking
/// the stop flag again.
const PUMP_POLL: Duration = Duration::from_millis(10);

/// A live capture session. Dropping stops and joins the capture thread;
/// the thread is the only sanctioned teardown path (CONSTITUTION rule 9:
/// the stopper owns the stop signal).
pub struct CaptureHandle {
    stop: Arc<AtomicBool>,
    join: Option<JoinHandle<()>>,
    level_bits: Arc<AtomicU32>,
    error: Arc<Mutex<Option<AudioError>>>,
}

impl CaptureHandle {
    /// Root-mean-square level of the most recently processed audio block,
    /// for a cosmetic waveform. Not calibrated to dBFS.
    #[must_use]
    pub fn level_rms(&self) -> f32 {
        f32::from_bits(self.level_bits.load(Ordering::Relaxed))
    }

    /// Take (clear) any error the capture thread recorded. A stream that
    /// dies mid-session (device unplugged, format renegotiation) reports
    /// here rather than panicking the audio thread.
    pub fn take_error(&self) -> Option<AudioError> {
        self.error
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take()
    }

    /// Stop capture and join the capture thread.
    pub fn stop(self) {
        drop(self);
    }

    /// Private shutdown logic: set stop flag and join the thread.
    fn shutdown(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(j) = self.join.take() {
            drop(j.join());
        }
    }
}

impl Drop for CaptureHandle {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Start capturing from `cfg.device_name` (or the system default), resampled
/// to [`crate::OUTPUT_SAMPLE_RATE`] mono.
///
/// **Blocking contract**: normally returns within ~2 s (the handshake
/// timeout below). But that timeout only bounds how long this function
/// *waits* for the capture thread's ready signal — if the thread is still
/// inside `cpal::Device::build_input_stream` when the 2 s elapses (e.g. the
/// very first launch, where `build_input_stream` blocks until the user
/// answers the OS's microphone permission dialog, which can take
/// arbitrarily long), the timeout arm still joins that thread before
/// returning, so this call can block well past 2 s in that specific case.
/// There is no way to abandon an in-flight `build_input_stream` call
/// without leaking the thread (see the timeout arm's comment for why a
/// join, not a detach, was chosen here).
///
/// # Errors
///
/// Returns [`AudioError::StreamBuild`] if the thread fails to spawn or the
/// stream fails to start within 2s; [`AudioError::DeviceUnavailable`] if no
/// input device can be resolved.
pub fn start_capture(
    cfg: &CaptureConfig,
) -> Result<(CaptureHandle, Receiver<Vec<f32>>), AudioError> {
    let stop = Arc::new(AtomicBool::new(false));
    let level_bits = Arc::new(AtomicU32::new(0f32.to_bits()));
    let error = Arc::new(Mutex::new(None));
    let (chunk_tx, chunk_rx) = bounded::<Vec<f32>>(64);
    let (ready_tx, ready_rx) = bounded::<Result<(), AudioError>>(1);

    let device_name = cfg.device_name.clone();
    let thread_stop = stop.clone();
    let thread_level = level_bits.clone();
    let thread_error = error.clone();

    let join = std::thread::Builder::new()
        .name("vuho-audio-capture".into())
        .spawn(move || {
            run_capture_thread(
                device_name.as_ref(),
                &thread_stop,
                &thread_level,
                &thread_error,
                &chunk_tx,
                &ready_tx,
            );
        })
        .map_err(|e| AudioError::StreamBuild(e.to_string()))?;

    match ready_rx.recv_timeout(Duration::from_secs(2)) {
        Ok(Ok(())) => Ok((
            CaptureHandle {
                stop,
                join: Some(join),
                level_bits,
                error,
            },
            chunk_rx,
        )),
        Ok(Err(e)) => {
            drop(join.join());
            Err(e)
        }
        Err(_) => {
            // `stop.store(true)` below is load-bearing, not incidental: if
            // the thread is already past the ready handshake and into
            // `pump_loop`, it polls `stop` every `PUMP_POLL` — without this
            // store, that thread runs forever (nothing else ever sets
            // `stop` once its own handshake `send` above already failed as
            // a no-op against a receiver we've stopped listening on), and
            // the `join()` immediately below would block this call forever
            // waiting for a thread we never actually asked to stop. If the
            // thread is instead still parked earlier (e.g. inside
            // `build_input_stream`, see below), the store has no effect
            // yet, but the thread observes it on its very first `stop`
            // check once that call returns.
            //
            // `join()` here (rather than detaching) can block this call
            // arbitrarily longer than the 2s we already waited — e.g. the
            // thread is parked inside `build_input_stream` waiting on the
            // first-run TCC microphone dialog. Detaching instead would
            // return promptly but leave that thread running unsupervised
            // (CONSTITUTION rule 20: own what you create) with no handle
            // left to ever stop it, orphaning a stream that starts playing
            // audio into a ring buffer nobody drains, and racing this
            // function's caller retrying `start_capture` against a second,
            // untracked capture thread. A slow, bounded-by-the-user's-own
            // click return is preferable to an unkillable background
            // capture thread — see this function's doc comment for the
            // resulting blocking contract.
            stop.store(true, Ordering::SeqCst);
            drop(join.join());
            Err(AudioError::StreamBuild(
                "timeout waiting for stream start".into(),
            ))
        }
    }
}

/// Body of the `"vuho-audio-capture"` thread: resolve device → build stream
/// → play → pump ring buffer → resample → forward chunks, until `stop`.
fn run_capture_thread(
    device_name: Option<&String>,
    stop: &Arc<AtomicBool>,
    level_bits: &Arc<AtomicU32>,
    error: &Arc<Mutex<Option<AudioError>>>,
    chunk_tx: &Sender<Vec<f32>>,
    ready_tx: &Sender<Result<(), AudioError>>,
) {
    let device = match resolve_device(device_name.map(String::as_str)) {
        Ok(d) => d,
        Err(e) => {
            drop(ready_tx.send(Err(e)));
            return;
        }
    };

    let supported = match device.default_input_config() {
        Ok(c) => c,
        Err(e) => {
            drop(ready_tx.send(Err(AudioError::StreamBuild(e.to_string()))));
            return;
        }
    };

    let sample_format = supported.sample_format();
    let config: StreamConfig = supported.config();
    let channels = usize::from(config.channels);
    let input_rate = config.sample_rate;

    let (producer, mut consumer) = rtrb::RingBuffer::<f32>::new(RING_CAPACITY);
    let dropped = Arc::new(AtomicUsize::new(0));

    let error_for_cb = error.clone();
    let stop_for_cb = stop.clone();
    let error_callback = move |err: cpal::Error| {
        log::warn!("vuho-audio: stream error: {err}");
        record_error(&error_for_cb, AudioError::StreamDied(err.to_string()));
        stop_for_cb.store(true, Ordering::SeqCst);
    };

    let stream = match build_stream(StreamParams {
        device: &device,
        config,
        sample_format,
        producer,
        level_bits: level_bits.clone(),
        dropped: dropped.clone(),
        error_callback,
    }) {
        Ok(s) => s,
        Err(e) => {
            drop(ready_tx.send(Err(e)));
            return;
        }
    };

    if let Err(e) = stream.play() {
        drop(ready_tx.send(Err(AudioError::StreamPlay(e.to_string()))));
        return;
    }

    if ready_tx.send(Ok(())).is_err() {
        // Caller already timed out and gave up; nothing left to do but tear
        // down cleanly.
        stop.store(true, Ordering::SeqCst);
    }

    pump_loop(
        &mut consumer,
        input_rate,
        channels,
        stop,
        chunk_tx,
        &dropped,
        error,
    );

    // `stream` drops here, on the thread that built it.
    drop(stream);
}

/// Record a capture-thread failure into the shared error slot (recovering
/// from a poisoned lock per rule 12 — the slot's contents are about to be
/// overwritten with the new failure either way) so
/// [`CaptureHandle::take_error`] can report the true cause instead of the
/// generic disconnect a caller sees when a chunk-channel sender is simply
/// dropped. One chokepoint for every capture-thread exit path that needs to
/// leave a reason behind: the stream error callback, a resampler
/// construction failure, and a persistent resample failure (see
/// [`pump_loop`]).
fn record_error(error: &Mutex<Option<AudioError>>, err: AudioError) {
    *error
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(err);
}

/// Resolve the named device, falling back to the default input device (with
/// a warning) if the name doesn't match anything currently connected.
fn resolve_device(wanted: Option<&str>) -> Result<cpal::Device, AudioError> {
    let host = cpal::default_host();
    if let Some(wanted) = wanted {
        if let Ok(devices) = host.input_devices() {
            for d in devices {
                if d.description().is_ok_and(|desc| desc.name() == wanted) {
                    return Ok(d);
                }
            }
        }
        log::warn!(
            "vuho-audio: configured input device {wanted:?} not found; using system default"
        );
    }
    host.default_input_device()
        .ok_or_else(|| AudioError::DeviceUnavailable("no default input device".into()))
}

/// Grouped arguments for [`build_stream`], replacing a long positional
/// parameter list (was `#[allow(clippy::too_many_arguments)]`) with a named
/// struct; `device` borrows the caller's `cpal::Device` for the duration of
/// the build call only.
struct StreamParams<'a, E>
where
    E: FnMut(cpal::Error) + Send + 'static,
{
    device: &'a cpal::Device,
    config: StreamConfig,
    sample_format: SampleFormat,
    producer: rtrb::Producer<f32>,
    level_bits: Arc<AtomicU32>,
    dropped: Arc<AtomicUsize>,
    error_callback: E,
}

/// Build the typed input stream for `sample_format`, converting to f32 in
/// the callback when the device isn't natively f32. The callback does only
/// format conversion + ring-buffer push + RMS — no allocation, no resample
/// (CONSTITUTION rule: keep the realtime audio callback minimal). The
/// producer push is a single bulk [`rtrb::Producer::push_partial_slice`]
/// copy per block rather than a per-sample loop, structurally ruling out a
/// per-element branch/alloc in the realtime path.
fn build_stream<E>(params: StreamParams<'_, E>) -> Result<Stream, AudioError>
where
    E: FnMut(cpal::Error) + Send + 'static,
{
    let StreamParams {
        device,
        config,
        sample_format,
        mut producer,
        level_bits,
        dropped,
        error_callback,
    } = params;

    let push_block = move |samples: &[f32]| {
        let (_, remainder) = producer.push_partial_slice(samples);
        if !remainder.is_empty() {
            dropped.fetch_add(remainder.len(), Ordering::Relaxed);
        }
        #[allow(clippy::cast_precision_loss)]
        let rms = if samples.is_empty() {
            0.0
        } else {
            let sum_sq: f32 = samples.iter().map(|s| s * s).sum();
            (sum_sq / samples.len() as f32).sqrt()
        };
        level_bits.store(rms.to_bits(), Ordering::Relaxed);
    };

    let timeout = Some(Duration::from_secs(2));
    match sample_format {
        SampleFormat::F32 => {
            let mut push_block = push_block;
            device
                .build_input_stream(
                    config,
                    move |data: &[f32], _| push_block(data),
                    error_callback,
                    timeout,
                )
                .map_err(|e| AudioError::StreamBuild(e.to_string()))
        }
        SampleFormat::I16 => {
            let mut push_block = push_block;
            let mut buf = Vec::new();
            device
                .build_input_stream(
                    config,
                    move |data: &[i16], _| {
                        buf.clear();
                        buf.extend(data.iter().map(|&s| f32::from(s) / f32::from(i16::MAX)));
                        push_block(&buf);
                    },
                    error_callback,
                    timeout,
                )
                .map_err(|e| AudioError::StreamBuild(e.to_string()))
        }
        SampleFormat::U16 => {
            let mut push_block = push_block;
            let mut buf = Vec::new();
            device
                .build_input_stream(
                    config,
                    move |data: &[u16], _| {
                        buf.clear();
                        buf.extend(data.iter().map(|&s| (f32::from(s) - 32768.0) / 32768.0));
                        push_block(&buf);
                    },
                    error_callback,
                    timeout,
                )
                .map_err(|e| AudioError::StreamBuild(e.to_string()))
        }
        other => Err(AudioError::StreamBuild(format!(
            "unsupported sample format: {other:?}"
        ))),
    }
}

/// Pop interleaved samples off the ring buffer, downmix to mono, resample,
/// and forward to `chunk_tx` until `stop` is set (then flush the resampler
/// tail and exit) or the resampler fails unrecoverably (see the loop body
/// below). Either way, this function only ever *borrows* `chunk_tx` — it
/// never drops it. `chunk_tx` is actually owned by the closure
/// `start_capture` spawns onto `"vuho-audio-capture"`, which moved it in;
/// that closure's `chunk_tx` is dropped implicitly when it returns, at
/// thread exit, after this function (and the stream teardown that follows
/// it in `run_capture_thread`) has already completed. That drop — not
/// anything explicit here — is what the receiving `Receiver<Vec<f32>>`
/// observes as the end-of-audio signal.
fn pump_loop(
    consumer: &mut rtrb::Consumer<f32>,
    input_rate: u32,
    channels: usize,
    stop: &Arc<AtomicBool>,
    chunk_tx: &Sender<Vec<f32>>,
    dropped: &Arc<AtomicUsize>,
    error: &Arc<Mutex<Option<AudioError>>>,
) {
    let mut blocks = match BlockProcessor::new(input_rate, channels, chunk_tx, dropped) {
        Ok(b) => b,
        Err(e) => {
            log::error!("vuho-audio: resampler construction failed: {e}");
            record_error(error, e);
            return;
        }
    };

    loop {
        if let Err(e) = blocks.pump_once(consumer) {
            // A resample failure is not transient — the same `rubato`
            // state that just failed will fail identically on the next
            // tick, so spinning at `PUMP_POLL` and re-logging forever would
            // just lose every subsequent block of audio silently. Record
            // the cause and stop, instead: the session sees the channel
            // disconnect and, via `take_error`, why.
            log::error!(
                "vuho-audio: resample error: {e} — stopping capture (resampler cannot recover)"
            );
            record_error(error, e);
            break;
        }

        if stop.load(Ordering::SeqCst) {
            break;
        }
        std::thread::sleep(PUMP_POLL);
    }

    // Drain anything left after the stop flag was observed (best-effort:
    // if the loop above exited on a resample failure, this repeats it
    // against the same broken resampler and is dropped — the more useful
    // failure was already recorded).
    drop(blocks.pump_once(consumer));
    drop(blocks.flush());

    let dropped_total = dropped.load(Ordering::Relaxed);
    if dropped_total > 0 {
        log::warn!("vuho-audio: {dropped_total} samples/chunks dropped during capture (ring overrun or backpressure)");
    }
}

/// The drain ring buffer → downmix → resample → forward stage of
/// [`pump_loop`], grouping the resampler with the channel count and output
/// sinks it feeds — one code path for the loop's steady-state body and its
/// post-stop tail drain.
///
/// Both scratch buffers are reused across every [`Self::pump_once`] call
/// (cleared, never reallocated once grown to steady-state capacity) — no
/// per-tick allocation on the pump path.
struct BlockProcessor<'a> {
    channels: usize,
    resampler: Resampler,
    interleaved: Vec<f32>,
    mono_buf: Vec<f32>,
    chunk_tx: &'a Sender<Vec<f32>>,
    dropped: &'a Arc<AtomicUsize>,
}

impl<'a> BlockProcessor<'a> {
    fn new(
        input_rate: u32,
        channels: usize,
        chunk_tx: &'a Sender<Vec<f32>>,
        dropped: &'a Arc<AtomicUsize>,
    ) -> Result<Self, AudioError> {
        Ok(Self {
            channels,
            resampler: Resampler::new(input_rate)?,
            interleaved: Vec::new(),
            mono_buf: Vec::new(),
            chunk_tx,
            dropped,
        })
    }

    /// Pop everything currently in the ring buffer and, if that yielded any
    /// samples, downmix → resample → forward one block.
    fn pump_once(&mut self, consumer: &mut rtrb::Consumer<f32>) -> Result<(), AudioError> {
        self.interleaved.clear();
        while let Ok(s) = consumer.pop() {
            self.interleaved.push(s);
        }
        if self.interleaved.is_empty() {
            return Ok(());
        }
        let mono = downmix(&self.interleaved, self.channels, &mut self.mono_buf);
        let chunk = self.resampler.process(mono)?;
        if !chunk.is_empty() {
            forward_chunk(self.chunk_tx, chunk, self.dropped);
        }
        Ok(())
    }

    /// Flush the resampler's buffered tail and forward it — call once at
    /// stream end.
    fn flush(&mut self) -> Result<(), AudioError> {
        let tail = self.resampler.flush()?;
        if !tail.is_empty() {
            forward_chunk(self.chunk_tx, tail, self.dropped);
        }
        Ok(())
    }
}

fn forward_chunk(chunk_tx: &Sender<Vec<f32>>, chunk: Vec<f32>, dropped: &Arc<AtomicUsize>) {
    let n = chunk.len();
    if chunk_tx
        .send_timeout(chunk, Duration::from_millis(200))
        .is_err()
    {
        log::warn!("vuho-audio: chunk send timed out; dropping {n} samples");
        dropped.fetch_add(1, Ordering::Relaxed);
    }
}

/// Average `channels` interleaved channels down to mono.
///
/// The mono case (the common one — most devices/`vuho` itself run mono)
/// passes `interleaved` straight through with no copy. Multi-channel input
/// writes into `scratch`, which the caller (the pump loop) owns and reuses
/// across ticks, so downmixing never allocates once `scratch` has grown to
/// its steady-state capacity.
#[allow(clippy::cast_precision_loss)]
fn downmix<'a>(interleaved: &'a [f32], channels: usize, scratch: &'a mut Vec<f32>) -> &'a [f32] {
    if channels <= 1 {
        return interleaved;
    }
    scratch.clear();
    scratch.extend(
        interleaved
            .chunks(channels)
            .map(|frame| frame.iter().sum::<f32>() / frame.len() as f32),
    );
    scratch.as_slice()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// D4 regression: when the resampler can't even be constructed (e.g. an
    /// invalid input rate — `0` here, which `rubato::FftFixedIn::new`
    /// rejects deterministically, no device needed), `pump_loop` must record
    /// the failure into the shared `error` slot before returning, not just
    /// log it — otherwise `CaptureHandle::take_error` has nothing to report
    /// and the session sees a generic disconnect instead of the resample
    /// cause.
    #[test]
    fn pump_loop_records_resampler_construction_failure_in_error_slot() {
        let (_producer, mut consumer) = rtrb::RingBuffer::<f32>::new(16);
        let stop = Arc::new(AtomicBool::new(true));
        let (chunk_tx, chunk_rx) = crossbeam_channel::unbounded::<Vec<f32>>();
        let dropped = Arc::new(AtomicUsize::new(0));
        let error = Arc::new(Mutex::new(None));

        pump_loop(&mut consumer, 0, 1, &stop, &chunk_tx, &dropped, &error);

        let recorded = error
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take();
        assert!(
            matches!(recorded, Some(AudioError::Resample(_))),
            "expected a Resample error recorded in the error slot, got: {recorded:?}"
        );
        assert!(
            chunk_rx.try_recv().is_err(),
            "no chunks should have been forwarded when construction itself failed"
        );
    }

    #[test]
    fn downmix_mono_is_passthrough() {
        let input = [0.1, 0.2, 0.3];
        let mut scratch = Vec::new();
        {
            let out = downmix(&input, 1, &mut scratch);
            assert_eq!(out, &[0.1, 0.2, 0.3]);
            assert_eq!(out.as_ptr(), input.as_ptr(), "mono path must not copy");
        }
        // Passthrough must not touch the scratch buffer at all.
        assert!(scratch.is_empty());
    }

    #[test]
    fn downmix_stereo_averages_pairs() {
        let input = [1.0, -1.0, 0.5, 0.5];
        let mut scratch = Vec::new();
        assert_eq!(downmix(&input, 2, &mut scratch), &[0.0, 0.5]);
    }

    #[test]
    fn downmix_reuses_scratch_capacity_across_calls() {
        let mut scratch = Vec::new();
        let first = [1.0, 1.0, 2.0, 2.0];
        assert_eq!(downmix(&first, 2, &mut scratch), &[1.0, 2.0]);
        let cap_after_first = scratch.capacity();
        assert!(cap_after_first >= 2);

        let second = [3.0, 3.0, 4.0, 4.0];
        assert_eq!(downmix(&second, 2, &mut scratch), &[3.0, 4.0]);
        // No reallocation needed: same input size, capacity unchanged.
        assert_eq!(scratch.capacity(), cap_after_first);
    }
}
