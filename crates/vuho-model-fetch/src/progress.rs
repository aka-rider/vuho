//! Forwarding hf-hub's download progress into [`vuho_domain::ModelStatus`].

use std::collections::HashMap;
use std::sync::Mutex;

use crossbeam_channel::Sender;
use hf_hub::progress::{DownloadEvent, ProgressEvent, ProgressHandler};
use vuho_domain::ModelStatus;

/// Implements `hf-hub`'s [`ProgressHandler`] and republishes every
/// `DownloadEvent::Progress` as a [`ModelStatus::Downloading`] on `tx`.
///
/// `hf-hub`'s `DownloadEvent::Progress { files }` is a **delta**: `files`
/// contains only the files whose status or byte count changed since the
/// previous `Progress` event (see `hf_hub::progress`'s module docs).
/// Summing each event's `files` directly would therefore double-count
/// bytes already reported by an earlier event and can overshoot
/// `total_bytes`. This handler instead keeps the *latest* `bytes_completed`
/// per filename in `received` and re-sums the whole map on every event —
/// monotonic by construction, because hf-hub's own per-file progress
/// within one download never regresses, and a map keyed by filename can
/// only grow or have an existing entry replaced by an equal-or-larger
/// value.
///
/// `DownloadEvent::AggregateProgress` is deliberately **not** used for the
/// running total: its `bytes_completed`/`total_bytes` describe only the
/// in-flight Xet batch, not the whole `snapshot_download` — treating it as
/// a whole-download total would under-count files transferred outside
/// that batch (plain-HTTPS files, or an earlier/later Xet batch).
pub(crate) struct ChannelProgress {
    tx: Sender<ModelStatus>,
    total_bytes: u64,
    /// Latest known `bytes_completed` per repository filename.
    received: Mutex<HashMap<String, u64>>,
}

impl ChannelProgress {
    pub(crate) fn new(tx: Sender<ModelStatus>, total_bytes: u64) -> Self {
        Self {
            tx,
            total_bytes,
            received: Mutex::new(HashMap::new()),
        }
    }
}

impl ProgressHandler for ChannelProgress {
    fn on_progress(&self, event: &ProgressEvent) {
        let ProgressEvent::Download(DownloadEvent::Progress { files }) = event else {
            return;
        };

        // Poison recovery = log + into_inner + continue (CONSTITUTION rule
        // 12) — a panic while another thread held this lock must not turn
        // every subsequent progress event into a silent no-op.
        let mut received = match self.received.lock() {
            Ok(guard) => guard,
            Err(poisoned) => {
                log::warn!("vuho-model-fetch: progress map lock was poisoned, recovering");
                poisoned.into_inner()
            }
        };
        for file in files {
            received.insert(file.filename.clone(), file.bytes_completed);
        }
        let received_bytes = received.values().sum::<u64>();
        drop(received);

        // A disconnected receiver just means nobody is listening anymore
        // (the caller dropped its end); the download itself is unaffected,
        // so dropping the update is the correct reaction, not an error.
        let _ = self.tx.send(ModelStatus::Downloading {
            received_bytes,
            total_bytes: self.total_bytes,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hf_hub::progress::{FileProgress, FileStatus};

    fn progress_event(files: Vec<FileProgress>) -> ProgressEvent {
        ProgressEvent::Download(DownloadEvent::Progress { files })
    }

    fn file(
        filename: &str,
        bytes_completed: u64,
        total_bytes: u64,
        status: FileStatus,
    ) -> FileProgress {
        FileProgress {
            filename: filename.to_owned(),
            bytes_completed,
            total_bytes,
            status,
        }
    }

    /// Replays a realistic interleaved multi-file delta sequence and
    /// asserts every observed `received_bytes` is monotonically
    /// non-decreasing and never exceeds `total_bytes` — the property an
    /// AggregateProgress-based or naive-summing implementation would
    /// violate.
    #[test]
    fn received_bytes_is_monotonic_and_bounded_across_a_delta_sequence() {
        let (tx, rx) = crossbeam_channel::unbounded();
        let total_bytes = 1_000;
        let handler = ChannelProgress::new(tx, total_bytes);

        let sequence = [
            progress_event(vec![file("a.bin", 0, 400, FileStatus::Started)]),
            progress_event(vec![file("b.bin", 0, 600, FileStatus::Started)]),
            progress_event(vec![file("a.bin", 200, 400, FileStatus::InProgress)]),
            progress_event(vec![file("b.bin", 300, 600, FileStatus::InProgress)]),
            progress_event(vec![file("a.bin", 400, 400, FileStatus::Complete)]),
            progress_event(vec![file("b.bin", 600, 600, FileStatus::Complete)]),
        ];

        for event in &sequence {
            handler.on_progress(event);
        }

        let mut last = 0u64;
        let mut observed = 0usize;
        while let Ok(status) = rx.try_recv() {
            let ModelStatus::Downloading {
                received_bytes,
                total_bytes: seen_total,
            } = status
            else {
                panic!("expected Downloading, got {status:?}");
            };
            assert_eq!(seen_total, total_bytes);
            assert!(
                received_bytes >= last,
                "received_bytes regressed: {received_bytes} < {last}"
            );
            assert!(
                received_bytes <= total_bytes,
                "received_bytes {received_bytes} exceeded total_bytes {total_bytes}"
            );
            last = received_bytes;
            observed += 1;
        }
        assert_eq!(observed, sequence.len());
        assert_eq!(
            last, total_bytes,
            "final event should report the full total"
        );
    }

    #[test]
    fn non_progress_events_are_ignored() {
        let (tx, rx) = crossbeam_channel::unbounded();
        let handler = ChannelProgress::new(tx, 100);

        handler.on_progress(&ProgressEvent::Download(DownloadEvent::Start {
            total_files: 1,
            total_bytes: 100,
        }));
        handler.on_progress(&ProgressEvent::Download(DownloadEvent::Complete));

        assert!(
            rx.try_recv().is_err(),
            "non-Progress events must not emit a status"
        );
    }

    #[test]
    fn aggregate_progress_is_not_used_for_the_running_total() {
        let (tx, rx) = crossbeam_channel::unbounded();
        let handler = ChannelProgress::new(tx, 1_000);

        // A large AggregateProgress event (an in-flight Xet batch) must
        // not leak into `received_bytes` — only Progress-family per-file
        // deltas do.
        handler.on_progress(&ProgressEvent::Download(DownloadEvent::AggregateProgress {
            bytes_completed: 900,
            total_bytes: 900,
            bytes_per_sec: None,
        }));
        handler.on_progress(&progress_event(vec![file(
            "a.bin",
            50,
            1_000,
            FileStatus::InProgress,
        )]));

        let status = rx
            .try_recv()
            .expect("the Progress event should emit a status");
        assert_eq!(
            status,
            ModelStatus::Downloading {
                received_bytes: 50,
                total_bytes: 1_000,
            }
        );
        assert!(
            rx.try_recv().is_err(),
            "AggregateProgress must not emit a status on its own"
        );
    }
}
