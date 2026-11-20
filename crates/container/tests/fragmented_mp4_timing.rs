//! A fragmented MP4's timing comes from its fragments, not the moov.
//!
//! Chromium's `MediaRecorder` writes an `mdhd.duration` that is not the
//! recording's length. Read as the truth, it made a 2 s recording look like
//! 0.05 s at ~1000 fps, and a frame-rate cap downstream then decimated the
//! video to a handful of frames.

use container::streaming::demux_streaming;

const MEDIARECORDER: &[u8] = include_bytes!("fixtures/fragmented/chromium_mediarecorder_h264.mp4");

#[test]
fn mediarecorder_mp4_reports_the_length_of_its_fragments() {
    let mut d = demux_streaming(MEDIARECORDER).expect("opens");
    let info = d.header().info.clone();
    assert_eq!(info.codec, "h264");
    assert_eq!(info.total_frames, 56);
    // 59048 ticks at 30000: the sum of the trun sample durations.
    assert!(
        (info.duration - 59_048.0 / 30_000.0).abs() < 1e-9,
        "duration {} s, want 1.968 s (the mdhd says 0.05 s)",
        info.duration
    );
    assert!(
        (27.0..=29.0).contains(&info.frame_rate),
        "frame rate {}, want ~28 (56 frames over 2 s)",
        info.frame_rate
    );

    let mut n = 0;
    while d.next_video_sample().expect("pull").is_some() {
        n += 1;
    }
    assert_eq!(n, 56, "every fragment's samples are yielded");
}
