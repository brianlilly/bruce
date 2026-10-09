//! People: named people from face regions and unnamed face clusters that can be given a name.
//!
//! The top section shows named people (from XMP face regions): a close-up of their largest face,
//! the name and how many photos they are in. A click shows that person's photos in the grid.
//!
//! Below that, unnamed clusters from the face embedding index appear as cards the user can name:
//! clicking one opens a text input, and pressing Enter assigns the name to every face in the
//! cluster via `face.nameCluster`.
//!
//! Only the rows on screen ask for a face render (the engine caches them, memory and disk).

use egui::{Align2, Color32, Rect, Sense, Stroke, StrokeKind, pos2, vec2};
use lightcraft_catalog::Person;
use lightcraft_engine::face_index::UnnamedCluster;
use serde_json::json;

use crate::LightcraftApp;
use crate::theme::Tokens;
use crate::widgets::register;

/// A card's face picture (points); the name and count sit below it.
const CARD: f32 = 150.0;
const LABEL_H: f32 = 44.0;
const GAP: f32 = 16.0;
const PAD: f32 = 20.0;
const HEADER_H: f32 = 44.0;

pub fn show(app: &mut LightcraftApp, ui: &mut egui::Ui) {
    let t = Tokens::get(ui.ctx());
    let people = app.caches.people(&app.session.catalog, &app.session.filter);
    let clusters = app.caches.unnamed_clusters(&app.session);

    // --- Named People header ---
    let (head, _) = ui.allocate_exact_size(vec2(ui.available_width(), HEADER_H), Sense::hover());
    ui.painter().text(pos2(head.left() + PAD, head.center().y), Align2::LEFT_CENTER, "Named People", t.semibold(15.0), t.text);
    ui.painter().text(pos2(head.right() - PAD, head.center().y), Align2::RIGHT_CENTER, people.len().to_string(), t.font(13.0), t.text_dim);

    // the filters narrowing the list (a date, a keyword…), removable here
    let chips = lightcraft_engine::filter_chips(&app.session.filter, &app.session.catalog);
    super::chips::show(app, ui, &chips);

    if people.is_empty() && clusters.is_empty() {
        let (title, body) = if chips.is_empty() {
            ("No people yet", "Detect and embed faces, then come back to name them. Face names from XMP also show up here.")
        } else {
            ("No people in these photos", "Remove a filter above, or choose Clear all")
        };
        super::empty_message(ui, ui.available_rect_before_wrap(), title, body);
        return;
    }

    let ppp = ui.ctx().pixels_per_point();
    let active = app.session.filter.person.clone();

    egui::ScrollArea::vertical().auto_shrink(false).show_viewport(ui, |ui, viewport| {
        let width = ui.available_width();
        let cols = (((width - PAD * 2.0 + GAP) / (CARD + GAP)).floor() as usize).max(1);
        let row_h = CARD + LABEL_H + GAP;

        // --- Layout: named people rows, then a gap + header + unnamed cluster rows ---
        let named_rows = people.len().div_ceil(cols);
        let has_clusters = !clusters.is_empty();
        let cluster_header_h = if has_clusters { HEADER_H } else { 0.0 };
        let cluster_rows = clusters.len().div_ceil(cols);
        let total_h = PAD + named_rows as f32 * row_h + cluster_header_h + cluster_rows as f32 * row_h + PAD;

        let (area, _) = ui.allocate_exact_size(vec2(width, total_h), Sense::hover());

        // --- Named people cards (virtualized) ---
        let named_top = PAD;
        {
            let first = ((viewport.top() - area.top() - named_top) / row_h).floor().max(0.0) as usize;
            let last = (((viewport.bottom() - area.top() - named_top) / row_h).ceil().max(0.0) as usize).min(named_rows);
            for row in first..last {
                for col in 0..cols {
                    let Some(person) = people.get(row * cols + col) else { break };
                    let min = area.min + vec2(PAD + col as f32 * (CARD + GAP), named_top + row as f32 * row_h);
                    let selected = active.as_deref().is_some_and(|a| a.eq_ignore_ascii_case(&person.name));
                    named_card(app, ui, person, Rect::from_min_size(min, vec2(CARD, CARD + LABEL_H)), ppp, selected);
                }
            }
        }

        // --- Unnamed clusters section ---
        if has_clusters {
            let cluster_top = named_top + named_rows as f32 * row_h;

            // Section header
            let header_rect = Rect::from_min_size(area.min + vec2(0.0, cluster_top), vec2(width, HEADER_H));
            if viewport.intersects(header_rect) {
                ui.painter().text(
                    pos2(header_rect.left() + PAD, header_rect.center().y),
                    Align2::LEFT_CENTER,
                    "Unnamed Faces",
                    t.semibold(15.0),
                    t.text,
                );
                ui.painter().text(
                    pos2(header_rect.right() - PAD, header_rect.center().y),
                    Align2::RIGHT_CENTER,
                    clusters.len().to_string(),
                    t.font(13.0),
                    t.text_dim,
                );
            }

            let cards_top = cluster_top + cluster_header_h;
            let first = ((viewport.top() - area.top() - cards_top) / row_h).floor().max(0.0) as usize;
            let last = (((viewport.bottom() - area.top() - cards_top) / row_h).ceil().max(0.0) as usize).min(cluster_rows);
            for row in first..last {
                for col in 0..cols {
                    let Some(cluster) = clusters.get(row * cols + col) else { break };
                    let min = area.min + vec2(PAD + col as f32 * (CARD + GAP), cards_top + row as f32 * row_h);
                    cluster_card(app, ui, cluster, Rect::from_min_size(min, vec2(CARD, CARD + LABEL_H)), ppp);
                }
            }
        }
    });
}

/// A card for a named person.
fn named_card(app: &mut LightcraftApp, ui: &mut egui::Ui, person: &Person, r: Rect, ppp: f32, selected: bool) {
    let t = Tokens::get(ui.ctx());
    let face = Rect::from_min_size(r.min, vec2(CARD, CARD));
    let resp = ui.interact(r, egui::Id::new(("person-card", &person.name)), Sense::click());
    register(ui.ctx(), format!("person:{}", person.name), r);
    let p = ui.painter();
    p.rect_filled(face, 3.0, t.canvas);
    if let Some(job) = app.session.face_job(person.photo, person.face, (CARD * ppp).ceil() as usize)
        && let Some(tex) = app.renderer.variant(job)
    {
        p.image(tex.tex.id(), face, Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)), Color32::WHITE);
    }
    if selected {
        p.rect_stroke(face, 3.0, Stroke::new(2.0, Color32::WHITE), StrokeKind::Outside);
    } else if resp.hovered() {
        p.rect_stroke(face, 3.0, Stroke::new(1.0, t.text_dim), StrokeKind::Outside);
    }
    // a long name must not run past the card
    let name =
        if person.name.chars().count() > 19 { format!("{}…", person.name.chars().take(18).collect::<String>()) } else { person.name.clone() };
    p.text(pos2(r.left() + 2.0, face.bottom() + 14.0), Align2::LEFT_CENTER, name, t.semibold(13.0), t.text);
    let photos = if person.count == 1 { "1 photo".to_string() } else { format!("{} photos", person.count) };
    p.text(pos2(r.left() + 2.0, face.bottom() + 32.0), Align2::LEFT_CENTER, photos, t.font(12.0), t.text_dim);
    if resp.on_hover_text(format!("{} — show their photos", person.name)).clicked() {
        let _ = app.run("library.filter", json!({"person": person.name}));
        let _ = app.run("view.photoGrid", json!({}));
    }
}

/// A card for an unnamed face cluster. Clicking it starts the naming flow.
fn cluster_card(app: &mut LightcraftApp, ui: &mut egui::Ui, cluster: &UnnamedCluster, r: Rect, ppp: f32) {
    let t = Tokens::get(ui.ctx());
    let face_rect = Rect::from_min_size(r.min, vec2(CARD, CARD));
    let is_naming = app.ui.naming_cluster.as_ref().is_some_and(|(id, _)| *id == cluster.cluster_id);
    let card_id = egui::Id::new(("cluster-card", cluster.cluster_id));

    let resp = ui.interact(r, card_id, Sense::click());
    register(ui.ctx(), format!("cluster:{}", cluster.cluster_id), r);
    let p = ui.painter();

    // Face thumbnail
    p.rect_filled(face_rect, 3.0, t.canvas);
    if let Some(job) = app.session.face_job(cluster.photo, cluster.face, (CARD * ppp).ceil() as usize)
        && let Some(tex) = app.renderer.variant(job)
    {
        p.image(tex.tex.id(), face_rect, Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)), Color32::WHITE);
    }

    // Dashed border to signal "unnamed"
    let border_color = if is_naming {
        t.accent
    } else if resp.hovered() {
        t.text_dim
    } else {
        Color32::from_white_alpha(40)
    };
    p.rect_stroke(face_rect, 3.0, Stroke::new(1.0, border_color), StrokeKind::Outside);

    if is_naming {
        // --- Naming mode: show a text input ---
        let input_id = egui::Id::new(("cluster-name-input", cluster.cluster_id));
        let input_rect = Rect::from_min_size(pos2(r.left(), face_rect.bottom() + 4.0), vec2(CARD, LABEL_H - 8.0));
        // We need to extract the text, show the widget, then put it back
        let mut text = app.ui.naming_cluster.as_ref().map(|(_, t)| t.clone()).unwrap_or_default();
        let te = egui::TextEdit::singleline(&mut text).id(input_id).hint_text("Type a name…").desired_width(CARD - 4.0).font(t.font(13.0));
        let te_resp = ui.put(input_rect, te);
        register(ui.ctx(), format!("field:clusterName{}", cluster.cluster_id), te_resp.rect);

        // Focus on first frame
        if te_resp.gained_focus() || !te_resp.has_focus() {
            te_resp.request_focus();
        }

        // Update the stored text
        if let Some((_, ref mut stored_text)) = app.ui.naming_cluster {
            *stored_text = text.clone();
        }

        // Enter → name the cluster
        let enter = ui.input(|i| i.key_pressed(egui::Key::Enter));
        if enter && !text.trim().is_empty() {
            let name = text.trim().to_string();
            let cid = cluster.cluster_id;
            app.ui.naming_cluster = None;
            let _ = app.run("face.nameCluster", json!({"cluster": cid, "name": name}));
        }

        // Escape → cancel naming
        let esc = ui.input(|i| i.key_pressed(egui::Key::Escape));
        if esc {
            app.ui.naming_cluster = None;
        }
    } else {
        // --- Normal mode: show placeholder label ---
        let label = format!("Person {}", cluster.cluster_id.saturating_add(1));
        p.text(pos2(r.left() + 2.0, face_rect.bottom() + 14.0), Align2::LEFT_CENTER, label, t.font(13.0), t.text_dim);
        let faces = if cluster.count == 1 { "1 face".to_string() } else { format!("{} faces", cluster.count) };
        p.text(pos2(r.left() + 2.0, face_rect.bottom() + 32.0), Align2::LEFT_CENTER, faces, t.font(12.0), t.text_dim);

        if resp.on_hover_text("Click to name this person").clicked() {
            app.ui.naming_cluster = Some((cluster.cluster_id, String::new()));
        }
    }
}
