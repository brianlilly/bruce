//! Face pipeline commands: detect faces in all photos, embed all faces, cluster by similarity,
//! name clusters.

use lightcraft_catalog::{Op, PhotoId};
use lightcraft_meta::RegionKind;
use serde_json::{Value, json};

use super::{CommandSpec, always, bad, cmd, f64_or};
use crate::Session;
use crate::face_index::{ClusterResult, FaceKey};

/// Enablement: face detection model must be installed.
fn detect_installed(s: &Session) -> std::result::Result<(), String> {
    s.face_detector.model_dir().map(|_| ())
}

/// Enablement: face embedding model must be installed.
fn embed_installed(s: &Session) -> std::result::Result<(), String> {
    s.face_embedder.model_dir().map(|_| ())
}

/// Enablement: both face detection and embedding models must be installed.
fn detect_and_embed_installed(s: &Session) -> std::result::Result<(), String> {
    detect_installed(s)?;
    embed_installed(s)
}

/// Collect all photo ids that have no face regions yet.
fn photos_without_faces(s: &Session) -> Vec<PhotoId> {
    s.catalog
        .photos()
        .filter(|photo| !photo.meta.regions.iter().any(|r| matches!(r.kind, RegionKind::Face)))
        .map(|photo| photo.id)
        .collect()
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
            "face.detectAll",
            "Detect Faces in All Photos",
            ["Photo"],
            None,
            "{} → {detected, photos, skipped, errors} — run face detection on every photo that does not already have face regions",
            detect_installed,
            |s, _p| {
                let ids = photos_without_faces(s);
                let skipped = s.catalog.photos().count().saturating_sub(ids.len());
                let mut detected: u64 = 0;
                let mut photo_count: u64 = 0;
                let mut errors: Vec<Value> = Vec::new();
                let mut ops = Vec::new();

                for id in &ids {
                    match s.face_detect_sync(*id) {
                        Ok(result) => {
                            if result.faces.is_empty() {
                                continue;
                            }
                            let face_count = result.faces.len() as u64;
                            let photo = match s.catalog.photo(*id) {
                                Some(p) => p,
                                None => continue,
                            };
                            let mut meta = photo.meta.clone();
                            meta.regions.extend(result.faces);
                            ops.push(Op::SetMeta { id: *id, meta: Box::new(meta) });
                            detected = detected.saturating_add(face_count);
                            photo_count += 1;
                        }
                        Err(e) => {
                            errors.push(json!({
                                "photo": id.0,
                                "error": e,
                            }));
                        }
                    }
                }

                if !ops.is_empty() {
                    s.commit("Detect Faces", Op::Batch { ops })?;
                }

                Ok(json!({
                    "detected": detected,
                    "photos": photo_count,
                    "skipped": skipped,
                    "errors": errors.len(),
                    "errorDetails": errors,
                }))
            }
        ),
        cmd!(
            "face.scanAll",
            "Scan for Faces",
            ["Photo"],
            None,
            "{eps?: f64 (default 0.45), minSamples?: int (default 2)} → {detected, embedded, clusters, errors} — detect faces in all photos, embed them, and cluster",
            detect_and_embed_installed,
            |s, p| {
                // 1. Detect faces in all photos that don't have any.
                let detect_result = s.execute("face.detectAll", &json!({}))?;
                let detect_errors = detect_result.get("errors").and_then(Value::as_u64).unwrap_or(0);
                let detected = detect_result.get("detected").and_then(Value::as_u64).unwrap_or(0);

                // 2. Embed all face regions that aren't indexed yet.
                let embed_result = s.execute("face.embedAll", &json!({}))?;
                let embedded = embed_result.get("embedded").and_then(Value::as_u64).unwrap_or(0);
                let embed_errors = embed_result.get("errors").and_then(Value::as_u64).unwrap_or(0);

                // 3. Cluster.
                let eps = f64_or(p, "eps", 0.45);
                let min_samples = p.get("minSamples").and_then(Value::as_u64).unwrap_or(2);
                let cluster_result = s.execute("face.cluster", &json!({"eps": eps, "minSamples": min_samples}))?;
                let num_clusters = cluster_result
                    .get("clusters")
                    .and_then(Value::as_array)
                    .map(|a| a.len() as u64)
                    .unwrap_or(0);

                Ok(json!({
                    "detected": detected,
                    "embedded": embedded,
                    "clusters": num_clusters,
                    "errors": detect_errors.saturating_add(embed_errors),
                }))
            }
        ),
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
                let min_samples = p.get("minSamples").and_then(Value::as_u64).and_then(|v| usize::try_from(v).ok()).unwrap_or(2);

                let result = s.face_index.cluster(eps, min_samples);
                if cluster_id >= result.num_clusters {
                    return Err(bad("face.nameCluster", format!("cluster {cluster_id} does not exist (found {} clusters)", result.num_clusters)));
                }

                // Collect the face keys for the target cluster.
                let members: Vec<FaceKey> =
                    result.keys.iter().zip(result.labels.iter()).filter(|(_, label)| **label == Some(cluster_id)).map(|(key, _)| *key).collect();

                if members.is_empty() {
                    return Ok(json!({"named": 0}));
                }

                // Build SetMeta ops: set the region's name for each face in the cluster.
                let mut ops = Vec::new();
                for key in &members {
                    let photo = s.catalog.photo(key.photo).ok_or_else(|| bad("face.nameCluster", format!("photo {} not found", key.photo.0)))?;
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

#[cfg(test)]
mod tests {
    use lightcraft_catalog::{Photo, Source};
    use lightcraft_meta::{Region, RegionKind, Rect};

    use super::*;
    use crate::Session;

    /// Helper: create a session with a few photos (no face model, just catalog entries).
    fn session_with_photos() -> Session {
        let mut s = Session::new();

        // Photo 1: no face regions.
        let p1 = Photo::new(PhotoId(1), Source::File { path: "test/photo1.jpg".into() }, "photo1.jpg", "JPEG", 640, 480, "2026-01-01");
        s.catalog.apply(lightcraft_catalog::Op::AddPhoto { photo: Box::new(p1) }).expect("add photo 1");

        // Photo 2: already has a face region.
        let mut p2 = Photo::new(PhotoId(2), Source::File { path: "test/photo2.jpg".into() }, "photo2.jpg", "JPEG", 640, 480, "2026-01-01");
        p2.meta.regions.push(Region {
            rect: Rect::from_xywh(0.1, 0.2, 0.3, 0.4),
            kind: RegionKind::Face,
            name: None,
            description: None,
        });
        s.catalog.apply(lightcraft_catalog::Op::AddPhoto { photo: Box::new(p2) }).expect("add photo 2");

        // Photo 3: no face regions.
        let p3 = Photo::new(PhotoId(3), Source::File { path: "test/photo3.jpg".into() }, "photo3.jpg", "JPEG", 640, 480, "2026-01-01");
        s.catalog.apply(lightcraft_catalog::Op::AddPhoto { photo: Box::new(p3) }).expect("add photo 3");

        s
    }

    #[test]
    fn photos_without_faces_skips_photos_with_face_regions() {
        let s = session_with_photos();
        let ids = photos_without_faces(&s);
        // Photo 2 has a face region, so it should be excluded.
        assert!(!ids.contains(&PhotoId(2)), "photo 2 has a face region and should be excluded");
        assert!(ids.contains(&PhotoId(1)), "photo 1 has no face regions");
        assert!(ids.contains(&PhotoId(3)), "photo 3 has no face regions");
        assert_eq!(ids.len(), 2);
    }

    #[test]
    fn detect_installed_fails_without_model() {
        let s = Session::new();
        // No model directory configured → enablement should fail.
        let result = detect_installed(&s);
        assert!(result.is_err());
    }

    #[test]
    fn detect_and_embed_installed_fails_without_models() {
        let s = Session::new();
        let result = detect_and_embed_installed(&s);
        assert!(result.is_err());
    }

    #[test]
    fn detect_all_disabled_without_model() {
        let mut s = session_with_photos();
        // face.detectAll should be disabled when the model is not installed.
        let result = s.execute("face.detectAll", &json!({}));
        assert!(result.is_err(), "face.detectAll should fail without the face detection model");
    }

    #[test]
    fn scan_all_disabled_without_models() {
        let mut s = session_with_photos();
        // face.scanAll requires both detection and embedding models.
        let result = s.execute("face.scanAll", &json!({}));
        assert!(result.is_err(), "face.scanAll should fail without face models");
    }

    #[test]
    fn all_face_regions_includes_only_face_kind() {
        let s = session_with_photos();
        let regions = all_face_regions(&s);
        // Only photo 2 has a face region (index 0).
        assert_eq!(regions.len(), 1);
        assert_eq!(regions[0], (PhotoId(2), 0));
    }
}
