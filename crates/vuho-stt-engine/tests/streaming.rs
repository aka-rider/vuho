//! Real-microphone streaming smoke test — `#[ignore]`d by default (see
//! `streaming_smoke` below), since it needs an interactive microphone grant
//! and a signed binary declaring `NSMicrophoneUsageDescription`.

use vuho_stt_engine::{ParakeetEngine, TranscriptionEngine};

// The former `start_stream_without_init_returns_error` test is gone: an
// uninitialized engine is no longer representable. `ParakeetEngine::load` is
// the only constructor and it does not return until the models are resident
// (including ANE warmup), so there is no engine to call `start_stream` on
// before that finishes. The failure path it covered now lives in
// `engine_load_fails_for_a_missing_model_folder`.

/// # Streaming smoke test
///
/// Requires a microphone + a binary with `Info.plist` containing
/// `NSMicrophoneUsageDescription`. Run manually:
///
///   cargo test -p vuho-stt-engine `streaming_smoke` -- --ignored
///
/// Speak into the mic for ~3 seconds, then confirm.
#[ignore = "requires microphone + signed binary with NSMicrophoneUsageDescription; run manually with -- --ignored"]
#[test]
fn streaming_smoke() {
    let engine = ParakeetEngine::load(
        vuho_stt_engine::resolve_model_folder().expect("resolve model folder"),
    )
    .expect("engine load");

    let rx = engine.start_stream(Some("en"), None).expect("start_stream");

    let start = std::time::Instant::now();
    let mut got_partial = false;
    while start.elapsed() < std::time::Duration::from_secs(5) {
        if let Ok(vuho_domain::DictationEvent::PartialTranscript { .. }) =
            rx.recv_timeout(std::time::Duration::from_millis(200))
        {
            got_partial = true;
        }
    }

    let result = engine.stop_stream().expect("stop_stream");
    assert!(got_partial, "expected at least one PartialTranscript event");
    assert!(
        !result.full_text.is_empty(),
        "expected non-empty transcript"
    );
}

/// Calling `start_stream` while a session is already active must return
/// `Err(EngineError::StreamAlreadyActive)` (WP5's typed-error taxonomy) —
/// not silently replace the live session (which would leak its capture
/// thread), and the engine must remain stoppable afterward.
///
/// `#[ignore]`d for the same reason as `streaming_smoke`: the *first*
/// `start_stream` call must actually establish a real session, which needs
/// live microphone access. The double-start check itself fires before any
/// `CoreML`/capture work on the *second* call, but there is no way to reach
/// that code path without a genuinely active first stream.
#[ignore = "requires microphone + signed binary with NSMicrophoneUsageDescription; run manually with -- --ignored"]
#[test]
fn double_start_stream_returns_stream_already_active() {
    let engine = ParakeetEngine::load(
        vuho_stt_engine::resolve_model_folder().expect("resolve model folder"),
    )
    .expect("engine load");

    let _rx = engine
        .start_stream(Some("en"), None)
        .expect("first start_stream");

    let second = engine.start_stream(Some("en"), None);
    assert!(
        matches!(
            second,
            Err(vuho_stt_engine::EngineError::StreamAlreadyActive)
        ),
        "expected StreamAlreadyActive, got {second:?}"
    );

    // The engine must remain stoppable — the failed second call must not
    // have corrupted or replaced the first session's handle.
    let result = engine.stop_stream();
    assert!(
        result.is_ok(),
        "expected stop_stream to still succeed: {result:?}"
    );
}
