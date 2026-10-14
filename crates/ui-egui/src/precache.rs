//! Background thumbnail pre-generation.
//!
//! After a library is opened or a folder is browsed, this iterates every photo in the catalog and
//! ensures a disk-cached thumbnail exists. Uncached photos are rendered at low priority so viewport
//! requests always win. Progress is reported through [`Precache::cached`] / [`Precache::total`].

use lightcraft_catalog::PhotoId;
use lightcraft_engine::media::thumb_bucket;

/// Default long edge for pre-generated thumbnails: covers the most common grid cell sizes.
const PRECACHE_EDGE: usize = 384;

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
}

impl Precache {
    /// Start (or restart) precaching for the current catalog.
    pub fn new(ids: Vec<PhotoId>, catalog_rev: u64) -> Self {
        let total = ids.len();
        Self { remaining: ids, total, cached: 0, submitted: 0, catalog_rev, bucket: thumb_bucket(PRECACHE_EDGE) }
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
            let Some(job) = app_session.thumb_job(id, PRECACHE_EDGE) else {
                continue;
            };
            renderer.request(crate::render::Slot::Precache(id), job, PRECACHE_PRIORITY);
            self.submitted += 1;
        }
    }
}
