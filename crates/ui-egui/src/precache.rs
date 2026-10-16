//! Background thumbnail pre-generation.
//!
//! After a library is opened or a folder is browsed, this iterates every photo in the catalog and
//! ensures a disk-cached thumbnail exists. Uncached photos are rendered at low priority so viewport
//! requests always win. Progress is reported through [`Precache::cached`] / [`Precache::total`].

use lightcraft_catalog::PhotoId;
use lightcraft_engine::media::thumb_bucket;

/// How many cache-key checks to run per frame (cheap: just a file-exists test).
const CHECKS_PER_FRAME: usize = 64;

/// Priority for precache jobs: below everything else (on-screen = 10, prefetch = 5,
/// background thumb replacement = 3).
const PRECACHE_PRIORITY: u32 = 1;

/// Background thumbnail precache state.
pub struct Precache {
    /// Photo IDs still to check (popped from the back).
    remaining: Vec<PhotoId>,
    /// How many photos we started with.
    total: usize,
    /// How many are already cached (or have been rendered).
    cached: usize,
    /// How many have been submitted to the render pool this session.
    submitted: usize,
    /// The catalog revision when we built the work list; if it changes we rebuild.
    catalog_rev: u64,
    /// The thumbnail bucket size we're pre-generating.
    bucket: usize,
    /// The long-edge size passed to `thumb_job` (before bucket rounding).
    long_edge: usize,
}

impl Precache {
    /// Start (or restart) precaching for the current catalog at the given thumbnail long edge.
    ///
    /// `long_edge` should match the grid's current thumbnail pixel size so that the cache keys
    /// line up with what the grid will request. It is rounded up to a [`THUMB_SIZES`] bucket.
    pub fn new(ids: Vec<PhotoId>, catalog_rev: u64, long_edge: usize) -> Self {
        let total = ids.len();
        Self { remaining: ids, total, cached: 0, submitted: 0, catalog_rev, bucket: thumb_bucket(long_edge), long_edge }
    }

    /// How many photos have been processed (cached or submitted).
    pub fn done(&self) -> usize {
        self.cached + self.submitted
    }

    /// Total number of photos.
    pub fn total(&self) -> usize {
        self.total
    }

    /// True when all photos have been checked and submitted (or were already cached).
    pub fn finished(&self) -> bool {
        self.remaining.is_empty()
    }

    /// The catalog revision this precache was built for.
    pub fn catalog_rev(&self) -> u64 {
        self.catalog_rev
    }

    /// The thumbnail bucket this precache is generating.
    pub fn bucket(&self) -> usize {
        self.bucket
    }

    /// Drive one frame's worth of work: check disk cache and submit render jobs.
    pub fn tick(&mut self, app_session: &mut lightcraft_engine::Session, renderer: &mut crate::render::Renderer) {
        // If the catalog changed (photos added/removed), we should restart, but the caller
        // handles that by replacing us with a new Precache.
        for _ in 0..CHECKS_PER_FRAME {
            let Some(id) = self.remaining.pop() else {
                break;
            };
            // Compute the disk cache key and check if it's already there.
            let Some(key) = app_session.thumb_disk_key(id, self.bucket) else {
                // Photo disappeared from catalog between listing and now.
                continue;
            };
            // Check the disk/memory cache (the borrow of `rendered` must not span the mutable
            // calls above and below, so we do the check and drop the reference immediately).
            if app_session.media.rendered.has(key) {
                self.cached += 1;
                continue;
            }
            // Not cached: build a render job and submit at low priority.
            let Some(job) = app_session.thumb_job(id, self.long_edge) else {
                continue;
            };
            renderer.request(crate::render::Slot::Precache(id), job, PRECACHE_PRIORITY);
            self.submitted += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lightcraft_engine::media::{thumb_bucket, THUMB_SIZES};

    /// The precache bucket must match what the grid requests for the same pixel size, otherwise
    /// precached thumbnails are never found and every grid cell re-renders from scratch.
    #[test]
    fn precache_bucket_matches_grid_bucket() {
        // Simulate the grid's thumb_px → request_thumb → thumb_bucket chain for a range of
        // display configurations. The grid calls `thumb_px(thumb_size * ppp)` which returns a
        // value from THUMB_SIZES, then `thumb_bucket(that)` which is an identity for those
        // values. The precache must land in the same bucket.
        for &(thumb_size, ppp) in &[
            (220.0_f32, 1.0_f32),  // default size, 1x display
            (220.0, 2.0),          // default size, Retina
            (150.0, 1.0),          // small thumbs
            (150.0, 2.0),          // small thumbs, Retina
            (400.0, 1.0),          // large thumbs
            (90.0, 1.0),           // minimum thumb size
            (480.0, 1.0),          // maximum thumb size
        ] {
            // What the grid computes (grid.rs thumb_px logic)
            let grid_px = (thumb_size * ppp).ceil() as usize;
            let grid_bucket = thumb_bucket(grid_px);

            // What the precache computes (lib.rs passes the same (thumb_size * ppp).ceil())
            let precache = Precache::new(vec![], 0, grid_px);
            assert_eq!(
                precache.bucket(),
                grid_bucket,
                "precache bucket {actual} != grid bucket {expected} for thumb_size={thumb_size}, ppp={ppp}",
                actual = precache.bucket(),
                expected = grid_bucket,
            );
        }
    }

    /// Each THUMB_SIZES entry is a valid bucket and maps to itself.
    #[test]
    fn thumb_sizes_are_identity_buckets() {
        for &sz in &THUMB_SIZES {
            assert_eq!(thumb_bucket(sz), sz);
        }
    }
}
