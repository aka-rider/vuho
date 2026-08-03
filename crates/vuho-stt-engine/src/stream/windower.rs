//! 15-second sliding window plan for Parakeet-TDT.
//!
//! Constants from the plan:
//!
//! - 1280 samples = 1 encoder frame = 80 ms
//! - 240 000 samples = 15 s = one model window
//! - 32 000 samples = 2 s = overlap (25 frames)
//! - 208 000 samples = 13 s = advance per window
//!
//! Note: `WINDOW_SAMPLES` (240 000) is not itself an exact multiple of
//! `SAMPLES_PER_FRAME` (240 000 / 1280 = 187.5) — the model's own encoder
//! rounds up to 188 frames internally. So only the very first window
//! (`start = 0`) and the explicitly floored end-aligned trailing window are
//! guaranteed frame-aligned; interior advance-window starts are not, and
//! `global_frame_offset` is a floor-divided estimate, not an exact frame
//! boundary, for those.

/// Samples per encoder frame (1280 = 80 ms @ 16 kHz).
pub(crate) const SAMPLES_PER_FRAME: usize = 1280;
/// Samples per model window (240 000 = 15 s).
pub(crate) const WINDOW_SAMPLES: usize = 240_000;
/// Overlap samples (32 000 = 2 s = 25 frames).
pub(crate) const OVERLAP_SAMPLES: usize = 32_000;
/// Advance per window (208 000 = 13 s).
pub(crate) const ADVANCE: usize = WINDOW_SAMPLES - OVERLAP_SAMPLES;
/// Overlap in encoder frames (25). `pub` (not `pub(crate)`) so
/// `benches/hot_paths.rs` — a separate crate — can use it as the single
/// source of truth for the overlap-frame count instead of a duplicated
/// literal (CONSTITUTION rule 26); re-exported via `crate::bench_support`.
pub const OVERLAP_FRAMES: usize = OVERLAP_SAMPLES / SAMPLES_PER_FRAME;

/// A planned window into the audio stream.
#[derive(Debug, Clone)]
pub struct Window {
    /// Start sample index in the input buffer.
    pub(crate) start: usize,
    /// Number of valid (non-padded) samples.
    pub(crate) len: usize,
    /// Global frame offset: `start / 1280`.
    pub(crate) global_frame_offset: usize,
}

/// Plan windows for `total_len` samples.
///
/// `pub` (not `pub(crate)`): re-exported by `bench_support` for
/// `benches/hot_paths.rs`.
///
/// Full windows every `ADVANCE` samples. When those don't already cover
/// `total_len`, one final window is appended, **end-aligned** (`start =
/// total_len.saturating_sub(WINDOW_SAMPLES)`, floored to a frame multiple)
/// — mirrors `FluidAudio`'s last-chunk warmup behavior. Flooring only ever
/// moves the start earlier, so the floored window still fits within
/// `WINDOW_SAMPLES` of `total_len`.
///
/// Returns at least one window (even for empty input, whose single window
/// has `len == 0`).
#[must_use]
pub fn plan(total_len: usize) -> Vec<Window> {
    if total_len == 0 {
        return vec![Window {
            start: 0,
            len: 0,
            global_frame_offset: 0,
        }];
    }

    let mut windows = Vec::new();
    let mut start = 0usize;
    while start + WINDOW_SAMPLES <= total_len {
        windows.push(Window {
            start,
            len: WINDOW_SAMPLES,
            global_frame_offset: start / SAMPLES_PER_FRAME,
        });
        start += ADVANCE;
    }

    let covered_end = windows.last().map_or(0, |w| w.start + w.len);
    if covered_end < total_len {
        let end_aligned_start = total_len.saturating_sub(WINDOW_SAMPLES);
        let frame_floored = (end_aligned_start / SAMPLES_PER_FRAME) * SAMPLES_PER_FRAME;
        let actual_len = WINDOW_SAMPLES.min(total_len - frame_floored);
        windows.push(Window {
            start: frame_floored,
            len: actual_len,
            global_frame_offset: frame_floored / SAMPLES_PER_FRAME,
        });
    }

    windows
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Single short input (less than one window): one end-aligned window
    /// covering everything from the start.
    #[test]
    fn single_short_window() {
        let windows = plan(100_000);
        assert_eq!(windows.len(), 1);
        assert_eq!(windows[0].len, 100_000);
        assert_eq!(windows[0].start, 0);
    }

    /// Exactly one window: no remainder, no duplicate trailing window.
    #[test]
    fn exact_one_window() {
        let windows = plan(WINDOW_SAMPLES);
        assert_eq!(windows.len(), 1);
        assert_eq!(windows[0].len, WINDOW_SAMPLES);
        assert_eq!(windows[0].start, 0);
    }

    /// One full window + a frame-aligned remainder: two windows, the second
    /// end-aligned exactly (remainder chosen as a multiple of the frame
    /// size so flooring is a no-op and the exact-position assertion holds).
    #[test]
    fn full_plus_remainder() {
        let total = WINDOW_SAMPLES + 50 * SAMPLES_PER_FRAME;
        let windows = plan(total);
        assert_eq!(windows.len(), 2);
        let last = &windows[windows.len() - 1];
        assert_eq!(last.start, total.saturating_sub(WINDOW_SAMPLES));
        assert_eq!(last.len, WINDOW_SAMPLES);
    }

    /// A remainder that is NOT frame-aligned still yields a window no
    /// longer than `WINDOW_SAMPLES` and covering up to `total_len`.
    #[test]
    fn unaligned_remainder_stays_within_window_bounds() {
        let total = WINDOW_SAMPLES + 50_000; // 50_000 is not a multiple of 1280
        let windows = plan(total);
        let last = windows.last().unwrap();
        assert!(last.len <= WINDOW_SAMPLES);
        assert!(last.start + last.len <= total);
        assert_eq!(
            last.start % SAMPLES_PER_FRAME,
            0,
            "end-aligned window must be frame-floored"
        );
    }

    /// Multiple full windows advance by exactly `ADVANCE` each.
    #[test]
    fn multiple_windows_advance_correctly() {
        let total = ADVANCE * 3 + WINDOW_SAMPLES; // 3 advances + 1 full window, no remainder
        let windows = plan(total);

        assert_eq!(windows.len(), 4);
        for (i, w) in windows.iter().enumerate() {
            assert_eq!(w.start, i * ADVANCE);
            assert_eq!(w.len, WINDOW_SAMPLES);
        }
    }

    /// The end-aligned trailing window is always frame-aligned, by
    /// construction, regardless of how unaligned `total_len` is.
    #[test]
    fn trailing_window_is_frame_aligned() {
        let windows = plan(175_000); // not frame-aligned
        let last = windows.last().unwrap();
        assert_eq!(
            last.start % SAMPLES_PER_FRAME,
            0,
            "trailing window start not frame-aligned"
        );
    }

    /// Empty input returns one zero-length window.
    #[test]
    fn empty_input() {
        let windows = plan(0);
        assert_eq!(windows.len(), 1);
        assert_eq!(windows[0].len, 0);
        assert_eq!(windows[0].start, 0);
    }

    /// Global frame offset is always `start / SAMPLES_PER_FRAME` (floor
    /// division), for every window `plan` produces.
    #[test]
    fn global_frame_offset_matches_floor_division() {
        let windows = plan(ADVANCE + WINDOW_SAMPLES);
        for w in &windows {
            assert_eq!(w.global_frame_offset, w.start / SAMPLES_PER_FRAME);
        }
    }
}
