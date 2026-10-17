//! Issue #137: the window appears immediately while the library loads on a background thread.
//! With >100 k photos the catalog deserialization can take tens of seconds; deferring it keeps
//! the window responsive.

use crate::headless::Headless;
use crate::panels::library_problem::LibraryProblem;
use crate::{LightcraftApp, LibraryLoadResult, Services};

fn has(h: &Headless, id: &str) -> bool {
    h.app.widgets.iter().any(|(w, _)| w == id)
}

/// When the background thread sends a successfully loaded session, the app swaps it in and the
/// loading overlay disappears.
#[test]
fn deferred_library_swaps_session_on_completion() {
    let dir = std::env::temp_dir().join(format!("lc-ui-deferred-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let lib = dir.join("lib");
    std::fs::create_dir_all(&lib).ok();

    // Start with an empty session + deferred loading receiver
    let (tx, rx) = std::sync::mpsc::channel();
    let app = LightcraftApp::new(lightcraft_engine::Session::new().with_fs(), Services::default());
    let app = app.with_deferred_library(rx);
    let mut h = Headless::new(app, [1300.0, 900.0], 1.0);

    // The library_loading is set
    assert!(h.app.library_loading.is_some());
    assert!(h.app.session.catalog.is_empty());
    h.step();

    // Simulate the background thread completing: send a session with a library
    let mut session = lightcraft_engine::Session::new().with_fs().with_system_clock();
    let _ = session.open_library(&lib, true);
    let result = LibraryLoadResult {
        session,
        problem: None,
        elapsed_ms: 42.0,
        pending_import: vec![],
    };
    tx.send(result).ok();
    h.step();
    h.step();

    // The loading is done: no more overlay, session has a library
    assert!(h.app.library_loading.is_none());
    assert!(h.app.session.library.is_some());
    assert!(h.app.library_problem.is_none());

    drop(h);
    let _ = std::fs::remove_dir_all(&dir);
}

/// When the background thread sends a failed load, the library problem dialog appears.
#[test]
fn deferred_library_shows_problem_on_failure() {
    let (tx, rx) = std::sync::mpsc::channel();
    let app = LightcraftApp::new(lightcraft_engine::Session::new().with_fs(), Services::default());
    let app = app.with_deferred_library(rx);
    let mut h = Headless::new(app, [1300.0, 900.0], 1.0);

    assert!(h.app.library_loading.is_some());
    h.step();

    // Send a failed load with pending imports
    let result = LibraryLoadResult {
        session: lightcraft_engine::Session::new().with_fs().with_system_clock(),
        problem: Some(LibraryProblem::new("/bad/path", "disk on fire")),
        elapsed_ms: 5.0,
        pending_import: vec!["photo.jpg".into()],
    };
    tx.send(result).ok();
    h.step();
    h.step();

    assert!(h.app.library_loading.is_none());
    let p = h.app.library_problem.as_ref().expect("library problem should be set");
    assert_eq!(p.error, "disk on fire");
    assert_eq!(p.pending_import, vec!["photo.jpg".to_string()]);
    assert!(has(&h, "button:libraryRetry"));
}

/// When the background thread disappears (e.g. panic), the app recovers gracefully.
#[test]
fn deferred_library_handles_thread_crash() {
    let (tx, rx) = std::sync::mpsc::channel();
    let app = LightcraftApp::new(lightcraft_engine::Session::new().with_fs(), Services::default());
    let app = app.with_deferred_library(rx);
    let mut h = Headless::new(app, [1300.0, 900.0], 1.0);

    assert!(h.app.library_loading.is_some());
    h.step();

    // Drop the sender without sending: simulates a thread panic
    drop(tx);
    h.step();
    h.step();

    // The loading is cleared, but no problem dialog (the thread just died)
    assert!(h.app.library_loading.is_none());
    assert!(h.app.library_problem.is_none());
}
