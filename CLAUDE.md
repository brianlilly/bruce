# Bruce — instructions for agents

Bruce is a desktop photo library manager with face detection/clustering — a modern Picasa replacement. It is a fork of LightCraft v0.4.0 (pure Rust, MIT OR Apache-2.0). Internal crate names remain `lightcraft-*`; the app identity (binary, window title, library path, env vars) is "Bruce."

## Project goals (in priority order)

1. **Photo gallery browsing** — fast, smooth navigation of 100k+ libraries on local/network storage
2. **Face detection and clustering** — automatic face detection, embedding, clustering with naming workflow
3. **Photo editing** — non-destructive adjustments (inherited from LightCraft, nice-to-have)

## Start every session here

1. Read this file.
2. Check `git status` and `git log --oneline -10` for context on recent work.
3. The face pipeline is the primary development focus. Gallery browsing works out of the box from LightCraft.

## Architecture

`geom`, `color`, `raster`, `tiff` (L0) → `raw`, `codecs`, `meta`, `develop` (L1) → `pipeline` → `catalog` → `engine`
→ `ui-egui`, `mcp` (L5) → apps `bruce` (desktop), `lightcraft-cli` (render/commands/MCP). Internal crate names are all `lightcraft-*`.

Key crates for face work:
- `crates/meta/src/lib.rs` — `Region` type with `RegionKind::Face`, normalized coordinates
- `crates/catalog/src/query.rs` — `Person` struct, `people_in()` aggregation
- `crates/engine/src/media.rs` — `face_job()` renders face crops from region coordinates
- `crates/ui-egui/src/panels/people.rs` — People view (cards per named person, virtualized)
- `crates/segment/` — SAM 3 inference on candle (template for adding face detection models)

## Never crash (outranks feature work)

- **Non-test code never panics:** no `unwrap()`, `expect()`, `panic!`, `unreachable!`, `todo!`, `unimplemented!`, and no `unsafe`. Return errors through `Result` and `?`.
- **`unsafe` lives only in `crates/sysmem`** (one FFI call).
- **Input-derived numbers are hostile:** `get()` instead of `[i]`, checked/saturating math.
- Every crash fix lands with a regression test.

## Non-negotiables

- **Clean-room.** Never read/disassemble Adobe app bundles. Never copy GPL/LGPL/AGPL code (darktable, RawTherapee, LibRaw, rawspeed, rawloader, dcraw-derived GPL code...).
- **Pure Rust** in the product. No C/C++ dependencies.
- **Layering** (enforced by `cargo xtask layers`): nothing below L5 depends on egui/eframe/winit/rfd.
- **Everything is a command** (`crates/engine`): id, label, menu path, shortcut, params.
- **Quality gates:** `cargo xtask ci` (fmt, clippy -D warnings, tests, layers, assets, wasm).

## Running

```sh
cargo run --release -p bruce                          # desktop app
cargo run --release -p bruce -- --control 7980        # with control server
cargo run --release -p bruce -- --memory              # in-memory demo session
cargo run --release -p lightcraft-cli -- render in.jpg -o out.jpg --set light.exposure=1
```

Environment: `BRUCE_LIBRARY`, `BRUCE_CONTROL_PORT`, `BRUCE_LOG`, `BRUCE_SAM3_DIR` for app-level settings. `LIGHTCRAFT_GPU_BACKEND`, `LIGHTCRAFT_GPU` for GPU settings (read by internal engine crates).

## Testing

- `cargo xtask ci` — full quality gate
- Unit/property tests next to the code
- New features need at least one test that would fail without them
- Shell gotcha: `mv`/`cp` are aliased interactive — use `/bin/mv -f` / `/bin/cp -f`

## Face pipeline (primary work)

The face pipeline needs building. Infrastructure to reuse:
- `Region` type with `Face` kind and normalized coordinates (crates/meta)
- People panel UI with virtualized cards (crates/ui-egui/panels/people.rs)
- PreviewCache + JobPool for background rendering (crates/preview)
- SAM 3 worker pattern: background thread, channel-based results (crates/segment)

Components to build:
1. **Face detection** — SCRFD or RetinaFace on candle, producing bounding boxes
2. **Face embeddings** — ArcFace or AdaFace on candle, 512-d vectors per face
3. **Face clustering** — DBSCAN or similar on embedding vectors
4. **Naming workflow** — UI for assigning names to clusters, merging, splitting
5. **Persistence** — face embeddings and cluster assignments in the catalog
