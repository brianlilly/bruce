# Bruce

**Your photos. Your faces. Your machine.**

Bruce is a lightweight desktop photo library manager with face detection and clustering — a modern replacement for Google Picasa. Built on the [LightCraft](https://github.com/storytold/lightcraft) engine (pure Rust), it focuses on fast browsing and organizing large photo collections with automatic face recognition.

## Goals

1. **Photo gallery browsing** — fast, smooth navigation of 100k+ photo libraries on local storage and network drives
2. **Face detection and clustering** — automatic face detection, embedding, and clustering with a simple naming workflow (the Picasa experience)
3. **Photo editing** — non-destructive adjustments inherited from LightCraft (nice-to-have, not the primary focus)

## Quick start

```sh
git clone https://github.com/brianlilly/bruce && cd bruce
cargo run --release -p bruce                        # opens ~/Pictures/Bruce Library
cargo run --release -p bruce -- ~/Pictures/trip     # import photos
cargo run --release -p bruce -- --memory            # throwaway demo session
cargo run --release -p bruce -- --control 7980      # with automation channel
```

## Environment variables

| Variable | Description |
|---|---|
| `BRUCE_LIBRARY` | Library directory (default: `~/Pictures/Bruce Library`) |
| `BRUCE_CONTROL_PORT` | JSON-lines control server port |
| `BRUCE_LOG=info\|debug` | Logging verbosity |
| `BRUCE_SAM3_DIR` | SAM 3 model directory for AI masks |
| `LIGHTCRAFT_GPU_BACKEND` | Graphics backend (`dx12`, `vulkan`, `metal`, `auto`, `off`) |
| `LIGHTCRAFT_GPU=0` | Disable GPU rendering |

## Architecture

Bruce is a fork of LightCraft v0.4.0. Internal crate names remain `lightcraft-*` for compatibility. The app binary, window title, default library path, and user-facing identity are all "Bruce."

Key crates: `engine` (session facade), `catalog` (photo database), `ui-egui` (native UI), `preview` (thumbnail cache), `meta` (XMP/face regions), `pipeline` (render pipeline).

## Status

- Gallery browsing: inherited from LightCraft, works today
- People panel: shows named face regions from XMP metadata
- Face detection pipeline: **in progress** (SCRFD/RetinaFace detection, ArcFace embeddings, DBSCAN clustering)

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option.
Based on [LightCraft](https://github.com/storytold/lightcraft) by the ArtCraft Team and contributors.
