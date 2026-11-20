# Fragmented MP4 fixtures

`chromium_mediarecorder_h264.mp4` — 2.0 s, 56 frames, 160x96 H.264
Constrained Baseline, no audio. Recorded by Chromium 156's `MediaRecorder`
(`video/mp4;codecs=avc1.42E01E`, `start(1000)`) from a canvas
`captureStream(30)` drawing a moving square on a changing colour. Variable
frame durations, two fragments, an empty moov sample table, and an
`mdhd.duration` of 1535 at a 30000 timescale (0.05 s) that is not the
recording's length: what a browser upload looks like. The trun sample
durations add up to 59048 ticks (1.968 s); an independent demuxer reads
56 frames over 2.002 s.
