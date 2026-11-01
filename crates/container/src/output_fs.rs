//! Where a package's files go: the seam a caller uses to take over how
//! rivet's CMAF / HLS output is stored.
//!
//! Every file of an HLS package — each rendition's `init.mp4` and
//! `seg-NNNNN.m4s`, the playlists, and the init rewrite
//! [`crate::cmaf::settle_video_sample_entry`] makes — is written whole,
//! through an [`OutputFs`], and read back through the same one. The default,
//! [`StdFs`], is the plain filesystem and changes nothing. A service that
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
    /// parent directory exists.
    fn write(&self, path: &Path, bytes: &[u8]) -> io::Result<()>;

    /// The bytes last stored at `path` by [`write`](Self::write).
    fn read(&self, path: &Path) -> io::Result<Vec<u8>>;
}

/// The plain filesystem.
#[derive(Debug, Clone, Copy, Default)]
pub struct StdFs;

impl OutputFs for StdFs {
    fn write(&self, path: &Path, bytes: &[u8]) -> io::Result<()> {
        std::fs::write(path, bytes)
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
}
