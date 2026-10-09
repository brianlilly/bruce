//! Persistent face embedding index and DBSCAN clustering.
//!
//! The index stores a 512-d L2-normalized embedding vector per face region. It is saved as a
//! binary file (`faces.bin`) in the library's file store, separate from the catalog JSON.
//!
//! Clustering runs DBSCAN over cosine distance on the stored embeddings. The session writes
//! cluster assignments as region names back into the catalog (via `SetMeta`), so the People
//! panel picks them up with no changes.

use std::collections::HashMap;

use lightcraft_catalog::PhotoId;
use lightcraft_meta::Rect;

/// The embedding dimension (AdaFace IResNet-18 produces 512-d vectors).
const EMBED_DIM: usize = 512;

/// File name in the library's file store.
pub const FACES_FILE: &str = "faces.bin";

/// Magic header bytes and format version for `faces.bin`.
const MAGIC: &[u8; 8] = b"BRUCEF01";

/// Bytes per stored face entry: 8 (photo id) + 4 (region index) + 512 * 4 (vector).
const ENTRY_BYTES: usize = 8 + 4 + EMBED_DIM * 4;

/// A face identified by its photo and region index within that photo's regions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FaceKey {
    pub photo: PhotoId,
    pub region_index: usize,
}

/// In-memory face embedding index.
#[derive(Default)]
pub struct FaceIndex {
    /// Embeddings keyed by (photo, region_index).
    entries: HashMap<FaceKey, Vec<f32>>,
    /// Whether the index has unsaved changes.
    dirty: bool,
}

/// A cluster assignment from DBSCAN.
#[derive(Clone, Debug)]
pub struct ClusterResult {
    /// Cluster id per face (same order as `keys`). `None` = noise (no cluster).
    pub labels: Vec<Option<usize>>,
    /// The face keys, in the same order as `labels`.
    pub keys: Vec<FaceKey>,
    /// Total number of clusters found.
    pub num_clusters: usize,
}

/// An unnamed face cluster for the People panel: a group of similar unnamed faces the user can
/// assign a name to. The representative face (largest area) is used for the card thumbnail.
#[derive(Clone, Debug)]
pub struct UnnamedCluster {
    /// The cluster id from DBSCAN (passed to `face.nameCluster`).
    pub cluster_id: usize,
    /// Number of unnamed faces in this cluster.
    pub count: usize,
    /// The photo with the largest unnamed face in this cluster (for the card thumbnail).
    pub photo: PhotoId,
    /// The face rect of the representative face (normalized, upright frame).
    pub face: Rect,
}

impl FaceIndex {
    /// Number of stored embeddings.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Whether there are unsaved changes.
    pub fn dirty(&self) -> bool {
        self.dirty
    }

    /// Insert or replace an embedding.
    pub fn insert(&mut self, key: FaceKey, vector: Vec<f32>) {
        self.entries.insert(key, vector);
        self.dirty = true;
    }

    /// Look up an embedding.
    pub fn get(&self, key: &FaceKey) -> Option<&[f32]> {
        self.entries.get(key).map(Vec::as_slice)
    }

    /// Whether a face has a stored embedding.
    pub fn contains(&self, key: &FaceKey) -> bool {
        self.entries.contains_key(key)
    }

    /// Remove all embeddings for a photo (e.g. when its regions change after re-detection).
    pub fn remove_photo(&mut self, photo: PhotoId) {
        let before = self.entries.len();
        self.entries.retain(|k, _| k.photo != photo);
        if self.entries.len() != before {
            self.dirty = true;
        }
    }

    /// All face keys that have stored embeddings.
    pub fn keys(&self) -> impl Iterator<Item = &FaceKey> {
        self.entries.keys()
    }

    // ---- persistence ----

    /// Load from a byte buffer (the content of `faces.bin`). Returns `Ok(index)` on success,
    /// `Err` if the data is corrupt or an unrecognised format version.
    pub fn load(data: &[u8]) -> Result<FaceIndex, String> {
        if data.len() < MAGIC.len() {
            return Err("faces.bin is too short".into());
        }
        if &data[..8] != MAGIC.as_slice() {
            return Err("faces.bin has an unrecognised header".into());
        }
        let body = &data[8..];
        if !body.len().is_multiple_of(ENTRY_BYTES) {
            return Err(format!("faces.bin body length {} is not a multiple of entry size {ENTRY_BYTES}", body.len()));
        }
        let count = body.len() / ENTRY_BYTES;
        let mut entries = HashMap::with_capacity(count);
        for i in 0..count {
            let off = i * ENTRY_BYTES;
            let photo_id = u64::from_le_bytes(body.get(off..off + 8).and_then(|s| s.try_into().ok()).ok_or("truncated photo id")?);
            let region_index =
                u32::from_le_bytes(body.get(off + 8..off + 12).and_then(|s| s.try_into().ok()).ok_or("truncated region index")?) as usize;
            let mut vector = Vec::with_capacity(EMBED_DIM);
            for j in 0..EMBED_DIM {
                let voff = off + 12 + j * 4;
                let f = f32::from_le_bytes(body.get(voff..voff + 4).and_then(|s| s.try_into().ok()).ok_or("truncated embedding vector")?);
                vector.push(f);
            }
            let key = FaceKey { photo: PhotoId(photo_id), region_index };
            entries.insert(key, vector);
        }
        Ok(FaceIndex { entries, dirty: false })
    }

    /// Serialize to bytes for writing to `faces.bin`.
    pub fn save(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(MAGIC.len() + self.entries.len() * ENTRY_BYTES);
        buf.extend_from_slice(MAGIC);
        // Sort by (photo id, region index) for deterministic output.
        let mut keys: Vec<&FaceKey> = self.entries.keys().collect();
        keys.sort_by_key(|k| (k.photo.0, k.region_index));
        for key in keys {
            let vector = &self.entries[key];
            buf.extend_from_slice(&key.photo.0.to_le_bytes());
            buf.extend_from_slice(&(key.region_index as u32).to_le_bytes());
            for &f in vector {
                buf.extend_from_slice(&f.to_le_bytes());
            }
        }
        buf
    }

    /// Mark as clean (after a successful save).
    pub fn mark_clean(&mut self) {
        self.dirty = false;
    }

    // ---- clustering ----

    /// Run DBSCAN clustering over all stored embeddings using cosine distance.
    ///
    /// - `eps`: maximum cosine distance (1 - similarity) for two faces to be neighbours.
    ///   Typical range: 0.3 (strict) to 0.6 (loose). Default recommendation: 0.45.
    /// - `min_samples`: minimum neighbours (including itself) for a core point. 2 is standard
    ///   for face clustering (a pair of the same person is enough to form a cluster).
    ///
    /// Returns cluster assignments for every stored face. Faces in no cluster get `None` (noise).
    pub fn cluster(&self, eps: f64, min_samples: usize) -> ClusterResult {
        let keys: Vec<FaceKey> = {
            let mut ks: Vec<FaceKey> = self.entries.keys().copied().collect();
            ks.sort_by_key(|k| (k.photo.0, k.region_index));
            ks
        };
        let n = keys.len();
        if n == 0 {
            return ClusterResult { labels: Vec::new(), keys, num_clusters: 0 };
        }

        // Pre-compute the distance matrix (cosine distance = 1 - cosine_similarity).
        // For N faces this is N*(N-1)/2 f64 values. At 10k faces that's ~400 MB, which is fine
        // for a desktop app. For truly huge libraries a spatial index would be needed, but
        // Picasa-scale libraries (< 100k faces) fit comfortably.
        let vectors: Vec<&[f32]> = keys.iter().map(|k| self.entries[k].as_slice()).collect();
        let dist = |i: usize, j: usize| -> f64 { 1.0 - cosine_similarity(vectors[i], vectors[j]) };

        // DBSCAN
        // Label: None = unvisited, Some(None) = noise, Some(Some(c)) = cluster c.
        let mut labels: Vec<Option<usize>> = vec![None; n];
        let mut visited = vec![false; n];
        let mut cluster_id: usize = 0;

        for i in 0..n {
            if visited[i] {
                continue;
            }
            visited[i] = true;

            let neighbours = range_query(&dist, n, i, eps);
            if neighbours.len() < min_samples {
                // noise (may be claimed by a cluster later)
                continue;
            }

            // Start a new cluster.
            let c = cluster_id;
            cluster_id += 1;
            labels[i] = Some(c);

            let mut seed_set: Vec<usize> = neighbours;
            let mut si = 0;
            while si < seed_set.len() {
                let q = seed_set[si];
                si += 1;

                if !visited[q] {
                    visited[q] = true;
                    let q_neighbours = range_query(&dist, n, q, eps);
                    if q_neighbours.len() >= min_samples {
                        // Merge q's neighbours into the seed set.
                        for nb in q_neighbours {
                            if !seed_set.contains(&nb) {
                                seed_set.push(nb);
                            }
                        }
                    }
                }
                if labels[q].is_none() {
                    labels[q] = Some(c);
                }
            }
        }

        ClusterResult { labels, keys, num_clusters: cluster_id }
    }
}

/// Cosine similarity between two L2-normalized vectors (a dot product).
fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b.iter()).map(|(&x, &y)| f64::from(x) * f64::from(y)).sum()
}

/// DBSCAN range query: all points within `eps` cosine distance of point `i`.
fn range_query(dist: &dyn Fn(usize, usize) -> f64, n: usize, i: usize, eps: f64) -> Vec<usize> {
    let mut result = Vec::new();
    for j in 0..n {
        if dist(i, j) <= eps {
            result.push(j);
        }
    }
    result
}

impl crate::Session {
    /// Compute unnamed face clusters for the People panel: run DBSCAN on the face index, then
    /// filter each cluster to only unnamed faces. Clusters that are entirely named (or empty after
    /// filtering) are excluded. The representative face in each cluster is the one with the largest
    /// area (best thumbnail).
    pub fn unnamed_clusters(&self) -> Vec<UnnamedCluster> {
        if self.face_index.is_empty() {
            return Vec::new();
        }
        let result = self.face_index.cluster(0.45, 2);
        let mut clusters = Vec::new();

        for cluster_id in 0..result.num_clusters {
            // Collect the unnamed members of this cluster.
            let mut best_photo = None;
            let mut best_face = None;
            let mut best_area: f64 = 0.0;
            let mut count: usize = 0;

            for (key, label) in result.keys.iter().zip(result.labels.iter()) {
                if *label != Some(cluster_id) {
                    continue;
                }
                // Check if this face already has a name.
                let Some(photo) = self.catalog.photo(key.photo) else { continue };
                let Some(region) = photo.meta.regions.get(key.region_index) else { continue };
                if region.name.as_deref().is_some_and(|n| !n.trim().is_empty()) {
                    continue; // already named — skip
                }
                count += 1;
                // Track the face with the largest area for the representative thumbnail.
                let area = (region.rect.x1 - region.rect.x0) * (region.rect.y1 - region.rect.y0) * f64::from(photo.width) * f64::from(photo.height);
                if area > best_area {
                    best_area = area;
                    best_photo = Some(key.photo);
                    best_face = Some(region.rect);
                }
            }

            if count >= 2
                && let (Some(photo), Some(face)) = (best_photo, best_face)
            {
                clusters.push(UnnamedCluster { cluster_id, count, photo, face });
            }
        }

        // Most faces first, then by cluster id for stability.
        clusters.sort_by(|a, b| b.count.cmp(&a.count).then_with(|| a.cluster_id.cmp(&b.cluster_id)));
        clusters
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_key(photo: u64, region: usize) -> FaceKey {
        FaceKey { photo: PhotoId(photo), region_index: region }
    }

    #[test]
    fn round_trip_empty() {
        let idx = FaceIndex::default();
        let data = idx.save();
        let loaded = FaceIndex::load(&data).expect("load");
        assert!(loaded.is_empty());
    }

    #[test]
    fn round_trip_with_entries() {
        let mut idx = FaceIndex::default();
        let v1: Vec<f32> = (0..512).map(|i| (i as f32) / 512.0).collect();
        let v2: Vec<f32> = (0..512).map(|i| 1.0 - (i as f32) / 512.0).collect();
        idx.insert(make_key(1, 0), v1.clone());
        idx.insert(make_key(2, 1), v2.clone());
        assert!(idx.dirty());

        let data = idx.save();
        let loaded = FaceIndex::load(&data).expect("load");
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded.get(&make_key(1, 0)).expect("v1"), v1.as_slice());
        assert_eq!(loaded.get(&make_key(2, 1)).expect("v2"), v2.as_slice());
    }

    #[test]
    fn remove_photo_clears_all_regions() {
        let mut idx = FaceIndex::default();
        let v: Vec<f32> = vec![0.0; 512];
        idx.insert(make_key(5, 0), v.clone());
        idx.insert(make_key(5, 1), v.clone());
        idx.insert(make_key(6, 0), v);
        assert_eq!(idx.len(), 3);
        idx.remove_photo(PhotoId(5));
        assert_eq!(idx.len(), 1);
        assert!(!idx.contains(&make_key(5, 0)));
        assert!(idx.contains(&make_key(6, 0)));
    }

    #[test]
    fn bad_magic_rejected() {
        let data = b"NOTVALID";
        assert!(FaceIndex::load(data).is_err());
    }

    #[test]
    fn truncated_data_rejected() {
        let mut idx = FaceIndex::default();
        idx.insert(make_key(1, 0), vec![0.0; 512]);
        let mut data = idx.save();
        data.truncate(data.len() - 10);
        assert!(FaceIndex::load(&data).is_err());
    }

    #[test]
    fn cluster_identical_vectors() {
        let mut idx = FaceIndex::default();
        // 3 identical vectors and 1 different one
        let same: Vec<f32> = {
            let mut v = vec![0.0; 512];
            v[0] = 1.0; // unit vector along dim 0
            v
        };
        let diff: Vec<f32> = {
            let mut v = vec![0.0; 512];
            v[1] = 1.0; // unit vector along dim 1 (cosine distance = 1.0 from `same`)
            v
        };
        idx.insert(make_key(1, 0), same.clone());
        idx.insert(make_key(2, 0), same.clone());
        idx.insert(make_key(3, 0), same);
        idx.insert(make_key(4, 0), diff);

        let result = idx.cluster(0.5, 2);
        assert_eq!(result.num_clusters, 1, "should find one cluster of the 3 identical faces");
        // The 3 identical faces should be in cluster 0; the outlier should be noise.
        let outlier_pos = result.keys.iter().position(|k| k.photo == PhotoId(4)).expect("outlier key");
        assert_eq!(result.labels[outlier_pos], None, "outlier should be noise");
        for (i, label) in result.labels.iter().enumerate() {
            if i != outlier_pos {
                assert_eq!(*label, Some(0), "identical faces should be in cluster 0");
            }
        }
    }

    #[test]
    fn cluster_two_groups() {
        let mut idx = FaceIndex::default();
        // Group A: vectors pointing in dim 0
        let a: Vec<f32> = {
            let mut v = vec![0.0; 512];
            v[0] = 1.0;
            v
        };
        // Group B: vectors pointing in dim 100
        let b: Vec<f32> = {
            let mut v = vec![0.0; 512];
            v[100] = 1.0;
            v
        };
        idx.insert(make_key(1, 0), a.clone());
        idx.insert(make_key(2, 0), a.clone());
        idx.insert(make_key(3, 0), a);
        idx.insert(make_key(10, 0), b.clone());
        idx.insert(make_key(11, 0), b.clone());
        idx.insert(make_key(12, 0), b);

        let result = idx.cluster(0.5, 2);
        assert_eq!(result.num_clusters, 2, "should find two clusters");
        // All faces should be assigned (no noise).
        for label in &result.labels {
            assert!(label.is_some(), "no face should be noise");
        }
        // Group A faces should share a cluster; group B faces should share a different cluster.
        let cluster_of = |photo: u64| -> usize {
            let pos = result.keys.iter().position(|k| k.photo == PhotoId(photo)).expect("key");
            result.labels[pos].expect("label")
        };
        assert_eq!(cluster_of(1), cluster_of(2));
        assert_eq!(cluster_of(1), cluster_of(3));
        assert_eq!(cluster_of(10), cluster_of(11));
        assert_eq!(cluster_of(10), cluster_of(12));
        assert_ne!(cluster_of(1), cluster_of(10));
    }

    #[test]
    fn cluster_empty_index() {
        let idx = FaceIndex::default();
        let result = idx.cluster(0.5, 2);
        assert_eq!(result.num_clusters, 0);
        assert!(result.labels.is_empty());
        assert!(result.keys.is_empty());
    }
}
