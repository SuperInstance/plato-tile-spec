//! Plato Tiling — Markdown → Semantic Tiles
//!
//! Splits Markdown by `##` headers into independent semantic nodes ("tiles").
//! Each tile has a slugified anchor, body, position, and extracted `[WordAnchors]`.
//!
//! Use: parse any Markdown doc into tiles, look up by anchor, inject context windows.
//! Zero dependencies. Pure data transform.

use std::collections::HashMap;

/// A single semantic tile — one `##`-delimited section of a Markdown document.
#[derive(Debug, Clone)]
pub struct KnowledgeTile {
    /// Unique anchor derived from the header text (e.g. `## Payment Flow` → `PaymentFlow`)
    pub anchor: String,
    /// Original header text
    pub header: String,
    /// Body content (everything between this header and the next)
    pub body: String,
    /// Zero-based position index in the original document
    pub position: usize,
    /// Tags extracted from the tile body (words inside `[brackets]`)
    pub word_anchors: Vec<String>,
}

impl KnowledgeTile {
    /// Build an anchor slug from a header string.
    pub fn slugify(header: &str) -> String {
        header
            .trim_start_matches('#')
            .trim()
            .split_whitespace()
            .map(|w| {
                let mut c = w.chars();
                match c.next() {
                    Some(f) => f.to_uppercase().to_string() + c.as_str(),
                    None => String::new(),
                }
            })
            .collect::<String>()
            .chars()
            .filter(|c| c.is_alphanumeric() || *c == '-')
            .collect()
    }

    /// Extract `[BracketedWord]` anchors from body text.
    pub fn extract_word_anchors(body: &str) -> Vec<String> {
        let mut anchors = Vec::new();
        let mut chars = body.chars().peekable();
        while let Some(ch) = chars.next() {
            if ch == '[' {
                let word: String = chars.by_ref().take_while(|&c| c != ']').collect();
                if !word.is_empty() && word.chars().all(|c| c.is_alphanumeric() || c == '-' || c == '_') {
                    anchors.push(word);
                }
            }
        }
        anchors
    }
}

/// The full tile registry for a parsed document.
#[derive(Debug, Default)]
pub struct TileRegistry {
    tiles: Vec<KnowledgeTile>,
    anchor_index: HashMap<String, usize>,
}

impl TileRegistry {
    /// Parse a Markdown string into tiles split on `##` headers.
    pub fn parse(content: &str) -> Self {
        let mut tiles = Vec::new();
        let mut current_header = String::from("Preamble");
        let mut current_body = String::new();
        let mut position = 0usize;

        for line in content.lines() {
            if line.starts_with("## ") {
                if !current_body.trim().is_empty() || !tiles.is_empty() || current_header != "Preamble" {
                    let anchor = KnowledgeTile::slugify(&current_header);
                    let word_anchors = KnowledgeTile::extract_word_anchors(&current_body);
                    tiles.push(KnowledgeTile { anchor, header: current_header.clone(), body: current_body.clone(), position, word_anchors });
                    position += 1;
                }
                current_header = line.to_string();
                current_body = String::new();
            } else {
                current_body.push_str(line);
                current_body.push('\n');
            }
        }

        if !current_body.trim().is_empty() {
            let anchor = KnowledgeTile::slugify(&current_header);
            let word_anchors = KnowledgeTile::extract_word_anchors(&current_body);
            tiles.push(KnowledgeTile { anchor, header: current_header, body: current_body, position, word_anchors });
        }

        let anchor_index = tiles.iter().enumerate().map(|(i, t)| (t.anchor.clone(), i)).collect();
        Self { tiles, anchor_index }
    }

    pub fn get_at(&self, position: usize) -> Option<&KnowledgeTile> { self.tiles.get(position) }

    /// Look up a tile by anchor slug (case-insensitive).
    pub fn get_by_anchor(&self, anchor: &str) -> Option<&KnowledgeTile> {
        if let Some(&idx) = self.anchor_index.get(anchor) { return self.tiles.get(idx); }
        let lower = anchor.to_lowercase();
        self.tiles.iter().find(|t| t.anchor.to_lowercase() == lower)
    }

    /// Return tiles in a window around current_position.
    pub fn inject_context(&self, current_position: usize, lookbehind: usize, lookahead: usize) -> Vec<&KnowledgeTile> {
        let start = current_position.saturating_sub(lookbehind);
        let end = (current_position + lookahead + 1).min(self.tiles.len());
        self.tiles[start..end].iter().collect()
    }

    pub fn all(&self) -> &[KnowledgeTile] { &self.tiles }
    pub fn len(&self) -> usize { self.tiles.len() }
    pub fn is_empty(&self) -> bool { self.tiles.is_empty() }
}

#[cfg(test)]
mod tests {
    use super::*;
    const SAMPLE: &str = "## Payment Flow\nHandles [PaymentFlow] requests.\n\n## Settlement\nClears funds.\n\n## Refund Policy\nRefunds the original [PaymentFlow].\n";

    #[test]
    fn parse_tiles() { let r = TileRegistry::parse(SAMPLE); assert_eq!(r.len(), 3); assert_eq!(r.get_at(0).unwrap().anchor, "PaymentFlow"); }
    #[test]
    fn anchor_lookup() { let r = TileRegistry::parse(SAMPLE); assert!(r.get_by_anchor("PaymentFlow").is_some()); assert!(r.get_by_anchor("paymentflow").is_some()); }
    #[test]
    fn word_anchors() { let r = TileRegistry::parse(SAMPLE); assert!(r.get_at(0).unwrap().word_anchors.contains(&"PaymentFlow".into())); }
    #[test]
    fn context_window() { let r = TileRegistry::parse(SAMPLE); assert_eq!(r.inject_context(1, 1, 1).len(), 3); }
}
