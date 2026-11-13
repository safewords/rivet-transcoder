//! Where a package's files go: the seam a caller uses to take over how
//! rivet's CMAF / HLS output is stored.
//!
//! Every file of an HLS package — each rendition's `init.mp4` and
//! `seg-NNNNN.m4s`, a subtitle rendition's `seg-NNNNN.vtt`, the playlists
//! (a live package's included), and the init rewrite
//! [`crate::cmaf::settle_video_sample_entry`] makes — is written whole,
//! through an [`OutputFs`], and read back through the same one. The default,
//! [`StdFs`], is the plain filesystem, each file appearing whole or not at
//! all ([`crate::atomic`]), and changes nothing. A service that
//! must not leave the package readable on its disk supplies its own — one that
//! encrypts each file with a key only it holds, say, or keeps them in memory —
//! and rivet never sees the difference: paths stay paths, byte counts in a
//! manifest are the bytes rivet produced, and whatever reads the package back
//! (the bandwidth measurement, the codec string, an uploader) goes through
//! the same `OutputFs`.
//!
//! Directories are still made with `std::fs`; they carry no content.

use std::fmt::Debug;
use std::io;
use std::path::Path;
use std::sync::Arc;

/// How the files of an output package are stored. Whole files only: every
/// writer in rivet assembles a file before writing it, so an implementation
/// never sees a partial one.
pub trait OutputFs: Send + Sync + Debug {
    /// Store `bytes` as the whole of `path`, replacing anything there. The
    /// parent directory exists. A live package's playlists are rewritten
    /// while a player polls them, so a reader of `path` must find the file
    /// that was there before or the new one, never part of either.
    fn write(&self, path: &Path, bytes: &[u8]) -> io::Result<()>;

    /// The bytes last stored at `path` by [`write`](Self::write).
    fn read(&self, path: &Path) -> io::Result<Vec<u8>>;
}

/// The plain filesystem. A file is written to a temporary file beside its
/// target and renamed over it ([`crate::atomic::write_atomic`]), so a job
/// that dies part way leaves no truncated file under a name a player trusts.
#[derive(Debug, Clone, Copy, Default)]
pub struct StdFs;

impl OutputFs for StdFs {
    fn write(&self, path: &Path, bytes: &[u8]) -> io::Result<()> {
        crate::atomic::write_atomic(path, bytes)
    }

    fn read(&self, path: &Path) -> io::Result<Vec<u8>> {
        std::fs::read(path)
    }
}

/// A shared [`OutputFs`], as the muxers and the engine hold it.
pub type SharedOutputFs = Arc<dyn OutputFs>;

/// The plain filesystem, shared.
pub fn std_fs() -> SharedOutputFs {
    Arc::new(StdFs)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::path::PathBuf;
    use std::sync::Mutex;

    /// An `OutputFs` that keeps every file in memory, XORed so a test can
    /// prove nothing reached the disk readable — and nothing at all.
    #[derive(Debug, Default)]
    pub struct MemFs {
        pub files: Mutex<HashMap<PathBuf, Vec<u8>>>,
    }

    impl OutputFs for MemFs {
        fn write(&self, path: &Path, bytes: &[u8]) -> io::Result<()> {
            self.files
                .lock()
                .unwrap()
                .insert(path.to_path_buf(), bytes.iter().map(|b| b ^ 0x5a).collect());
            Ok(())
        }

        fn read(&self, path: &Path) -> io::Result<Vec<u8>> {
            self.files
                .lock()
                .unwrap()
                .get(path)
                .map(|stored| stored.iter().map(|b| b ^ 0x5a).collect())
                .ok_or_else(|| io::Error::from(io::ErrorKind::NotFound))
        }
    }

    #[test]
    fn std_fs_round_trips() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("f");
        StdFs.write(&path, b"abc").unwrap();
        assert_eq!(StdFs.read(&path).unwrap(), b"abc");
        assert_eq!(std::fs::read(&path).unwrap(), b"abc", "plain on disk");
    }

    #[test]
    fn std_fs_writes_whole_or_not_at_all() {
        let dir = tempfile::tempdir().unwrap();
        // A hard link at the target is replaced, not written through: the
        // write went to a temporary file and was renamed over the target.
        let source = dir.path().join("source");
        std::fs::write(&source, b"source").unwrap();
        let path = dir.path().join("init.mp4");
        if std::fs::hard_link(&source, &path).is_ok() {
            StdFs.write(&path, b"new").unwrap();
            assert_eq!(std::fs::read(&source).unwrap(), b"source");
        }
        StdFs.write(&path, b"newer").unwrap();
        assert_eq!(StdFs.read(&path).unwrap(), b"newer");
        let names: Vec<String> = std::fs::read_dir(dir.path())
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert!(
            names.iter().all(|n| !crate::atomic::is_temporary_name(n)),
            "no temporary file left behind: {names:?}"
        );
        assert!(StdFs.write(&dir.path().join("missing/f"), b"x").is_err());
    }
}
