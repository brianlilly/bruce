//! Face pipeline commands: embed all faces, cluster by similarity, name clusters.

use lightcraft_catalog::{Op, PhotoId};
use lightcraft_meta::RegionKind;
use serde_json::{Value, json};

use super::{CommandSpec, always, bad, cmd, f64_or};
use crate::face_index::{FaceKey, ClusterResult};
use crate::Session;

/// Enablement: face embedding model must be installed.
fn embed_installed(s: &Session) -> std::result::Result<(), String> {
    s.face_embedder.model_dir().map(|_| ())
}

/// Collect all (photo, region_index) pairs for face regions in the catalog.
fn all_face_regions(s: &Session) -> Vec<(PhotoId, usize)> {
    let mut out = Vec::new();
    for photo in s.catalog.photos() {
        for (i, region) in photo.meta.regions.iter().enumerate() {
            if matches!(region.kind, RegionKind::Face) {
                out.push((photo.id, i));
            }
        }
    }
    out
}

/// Serialize a [`ClusterResult`] to JSON for the command response.
fn cluster_to_json(result: &ClusterResult) -> Value {
    let mut clusters: Vec<Value> = Vec::new();
    for c in 0..result.num_clusters {
        let members: Vec<Value> = result
            .keys
            .iter()
            .zip(result.labels.iter())
            .filter(|(_, label)| **label == Some(c))
            .map(|(key, _)| json!({"photo": key.photo.0, "region": key.region_index}))
            .collect();
        clusters.push(json!({"id": c, "count": members.len(), "members": members}));
    }
    let noise: Vec<Value> = result
        .keys
        .iter()
        .zip(result.labels.iter())
        .filter(|(_, label)| label.is_none())
        .map(|(key, _)| json!({"photo": key.photo.0, "region": key.region_index}))
        .collect();
    json!({
        "clusters": clusters,
        "noise": noise.len(),
        "total": result.keys.len(),
    })
}

pub fn specs() -> Vec<CommandSpec> {
    vec![
        cmd!(
            "face.embedAll",
            "Embed All Faces",
            [],
            None,
            "{} → {embedded, skipped, errors} — compute embeddings for every face region that does not already have one in the index",
            embed_installed,
            |s, _p| {
                let regions = all_face_regions(s);
                let mut embedded: u64 = 0;
                let mut skipped: u64 = 0;
                let mut errors: Vec<Value> = Vec::new();

                for (photo_id, region_index) in regions {
                    let key = FaceKey { photo: photo_id, region_index };
                    if s.face_index.contains(&key) {
                        skipped += 1;
                        continue;
                    }
                    match s.face_embed_sync(photo_id, region_index) {
                        Ok(result) => {
                            s.face_index.insert(key, result.vector);
                            embedded += 1;
                        }
                        Err(e) => {
                            errors.push(json!({
                                "photo": photo_id.0,
                                "region": region_index,
                                "error": e,
                            }));
                        }
                    }
                }

                Ok(json!({
                    "embedded": embedded,
                    "skipped": skipped,
                    "errors": errors.len(),
                    "errorDetails": errors,
                }))
            }
        ),
        cmd!(
            query "face.cluster",
            "Cluster Faces",
            [],
            None,
            "{eps?: f64 (default 0.45, range 0..1: max cosine distance), minSamples?: int (default 2)} → {clusters: [{id, count, members: [{photo, region}]}], noise, total}",
            always,
            |s, p| {
                let eps = f64_or(p, "eps", 0.45);
                let min_samples = p
                    .get("minSamples")
                    .and_then(Value::as_u64)
                    .and_then(|v| usize::try_from(v).ok())
                    .unwrap_or(2);

                if eps <= 0.0 || eps >= 1.0 {
                    return Err(bad("face.cluster", "`eps` must be between 0 and 1 (exclusive)"));
                }
                if min_samples == 0 {
                    return Err(bad("face.cluster", "`minSamples` must be at least 1"));
                }

                let result = s.face_index.cluster(eps, min_samples);
                Ok(cluster_to_json(&result))
            }
        ),
        cmd!(
            "face.nameCluster",
            "Name Face Cluster",
            [],
            None,
            "{cluster: int (cluster id from face.cluster), name: string, eps?: f64, minSamples?: int} — assigns the name to every face region in the cluster via SetMeta; undoable as one batch",
            always,
            |s, p| {
                let cluster_id = p
                    .get("cluster")
                    .and_then(Value::as_u64)
                    .and_then(|v| usize::try_from(v).ok())
                    .ok_or_else(|| bad("face.nameCluster", "missing `cluster` (integer)"))?;
                let name = p
                    .get("name")
                    .and_then(Value::as_str)
                    .filter(|n| !n.trim().is_empty())
                    .ok_or_else(|| bad("face.nameCluster", "missing or empty `name`"))?
                    .to_string();

                let eps = f64_or(p, "eps", 0.45);
                let min_samples = p
                    .get("minSamples")
                    .and_then(Value::as_u64)
                    .and_then(|v| usize::try_from(v).ok())
                    .unwrap_or(2);

                let result = s.face_index.cluster(eps, min_samples);
                if cluster_id >= result.num_clusters {
                    return Err(bad(
                        "face.nameCluster",
                        format!("cluster {cluster_id} does not exist (found {} clusters)", result.num_clusters),
                    ));
                }

                // Collect the face keys for the target cluster.
                let members: Vec<FaceKey> = result
                    .keys
                    .iter()
                    .zip(result.labels.iter())
                    .filter(|(_, label)| **label == Some(cluster_id))
                    .map(|(key, _)| *key)
                    .collect();

                if members.is_empty() {
                    return Ok(json!({"named": 0}));
                }

                // Build SetMeta ops: set the region's name for each face in the cluster.
                let mut ops = Vec::new();
                for key in &members {
                    let photo = s
                        .catalog
                        .photo(key.photo)
                        .ok_or_else(|| bad("face.nameCluster", format!("photo {} not found", key.photo.0)))?;
                    let mut meta = photo.meta.clone();
                    if let Some(region) = meta.regions.get_mut(key.region_index) {
                        region.name = Some(name.clone());
                        ops.push(Op::SetMeta { id: key.photo, meta: Box::new(meta) });
                    }
                }

                let named = ops.len();
                if named > 0 {
                    s.commit("Name Face Cluster", Op::Batch { ops })?;
                }

                Ok(json!({"named": named, "name": name}))
            }
        ),
    ]
}
