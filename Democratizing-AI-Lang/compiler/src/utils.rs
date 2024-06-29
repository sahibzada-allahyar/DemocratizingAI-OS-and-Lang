//! Utility functions and types for the Democratising compiler
//!
//! This module provides common functionality used across different parts of the compiler.

use std::fs;
use std::io;
use std::path::Path;

/// Read the contents of a source file
pub fn read_source_file(path: impl AsRef<Path>) -> io::Result<String> {
    fs::read_to_string(path)
}

/// A unique identifier generator
#[derive(Debug)]
pub struct IdGenerator {
    next_id: u32,
}

impl IdGenerator {
    /// Create a new ID generator starting from 0
    pub fn new() -> Self {
        Self { next_id: 0 }
    }

    /// Get the next unique ID
    pub fn next(&mut self) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }
}

/// A string interner for efficient string storage and comparison
#[derive(Debug, Default)]
pub struct StringInterner {
    strings: Vec<String>,
    map: std::collections::HashMap<String, usize>,
}

impl StringInterner {
    /// Create a new string interner
    pub fn new() -> Self {
        Self {
            strings: Vec::new(),
            map: std::collections::HashMap::new(),
        }
    }

    /// Intern a string, returning its index
    pub fn intern(&mut self, s: &str) -> usize {
        if let Some(&idx) = self.map.get(s) {
            return idx;
        }
        let idx = self.strings.len();
        self.strings.push(s.to_string());
        self.map.insert(s.to_string(), idx);
        idx
    }

    /// Get a string by its index
    pub fn get(&self, idx: usize) -> Option<&str> {
        self.strings.get(idx).map(|s| s.as_str())
    }
}

/// A type for tracking source code locations
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceLocation {
    /// The file path
    pub file: String,
    /// The line number (1-based)
    pub line: usize,
    /// The column number (1-based)
    pub column: usize,
    /// The byte offset in the source
    pub offset: usize,
}

impl SourceLocation {
    /// Create a new source location
    pub fn new(file: impl Into<String>, line: usize, column: usize, offset: usize) -> Self {
        Self {
            file: file.into(),
            line,
            column,
            offset,
        }
    }

    /// Create a dummy location for generated code
    pub fn generated() -> Self {
        Self {
            file: "<generated>".to_string(),
            line: 0,
            column: 0,
            offset: 0,
        }
    }
}

impl std::fmt::Display for SourceLocation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}:{}", self.file, self.line, self.column)
    }
}

/// A type for source code spans
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Span {
    /// Start location
    pub start: SourceLocation,
    /// End location
    pub end: SourceLocation,
}

impl Span {
    /// Create a new source span
    pub fn new(start: SourceLocation, end: SourceLocation) -> Self {
        Self { start, end }
    }

    /// Create a dummy span for generated code
    pub fn generated() -> Self {
        Self {
            start: SourceLocation::generated(),
            end: SourceLocation::generated(),
        }
    }
}

/// A type for diagnostic messages
#[derive(Debug)]
pub struct Diagnostic {
    /// The severity level
    pub level: DiagnosticLevel,
    /// The error message
    pub message: String,
    /// The source location
    pub location: Option<Span>,
    /// Additional notes
    pub notes: Vec<String>,
}

/// Diagnostic severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiagnosticLevel {
    /// An error that prevents compilation
    Error,
    /// A warning about potential issues
    Warning,
    /// An informational message
    Note,
}

impl Diagnostic {
    /// Create a new error diagnostic
    pub fn error(message: impl Into<String>) -> Self {
        Self {
            level: DiagnosticLevel::Error,
            message: message.into(),
            location: None,
            notes: Vec::new(),
        }
    }

    /// Create a new warning diagnostic
    pub fn warning(message: impl Into<String>) -> Self {
        Self {
            level: DiagnosticLevel::Warning,
            message: message.into(),
            location: None,
            notes: Vec::new(),
        }
    }

    /// Create a new note diagnostic
    pub fn note(message: impl Into<String>) -> Self {
        Self {
            level: DiagnosticLevel::Note,
            message: message.into(),
            location: None,
            notes: Vec::new(),
        }
    }

    /// Add a source location to the diagnostic
    pub fn with_location(mut self, location: Span) -> Self {
        self.location = Some(location);
        self
    }

    /// Add a note to the diagnostic
    pub fn with_note(mut self, note: impl Into<String>) -> Self {
        self.notes.push(note.into());
        self
    }
}

/// A collection of diagnostics
#[derive(Debug, Default)]
pub struct Diagnostics {
    messages: Vec<Diagnostic>,
}

impl Diagnostics {
    /// Create a new empty diagnostics collection
    pub fn new() -> Self {
        Self {
            messages: Vec::new(),
        }
    }

    /// Add a diagnostic message
    pub fn add(&mut self, diagnostic: Diagnostic) {
        self.messages.push(diagnostic);
    }

    /// Check if there are any error-level diagnostics
    pub fn has_errors(&self) -> bool {
        self.messages
            .iter()
            .any(|d| d.level == DiagnosticLevel::Error)
    }

    /// Get all diagnostic messages
    pub fn messages(&self) -> &[Diagnostic] {
        &self.messages
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_id_generator() {
        let mut gen = IdGenerator::new();
        assert_eq!(gen.next(), 0);
        assert_eq!(gen.next(), 1);
        assert_eq!(gen.next(), 2);
    }

    #[test]
    fn test_string_interner() {
        let mut interner = StringInterner::new();
        let idx1 = interner.intern("hello");
        let idx2 = interner.intern("world");
        let idx3 = interner.intern("hello");

        assert_eq!(idx1, idx3);
        assert_ne!(idx1, idx2);
        assert_eq!(interner.get(idx1), Some("hello"));
        assert_eq!(interner.get(idx2), Some("world"));
    }

    #[test]
    fn test_source_location() {
        let loc = SourceLocation::new("test.demo", 1, 1, 0);
        assert_eq!(loc.to_string(), "test.demo:1:1");

        let generated = SourceLocation::generated();
        assert_eq!(generated.file, "<generated>");
    }

    #[test]
    fn test_diagnostics() {
        let mut diagnostics = Diagnostics::new();

        diagnostics.add(
            Diagnostic::error("test error")
                .with_location(Span::generated())
                .with_note("additional info"),
        );

        diagnostics.add(Diagnostic::warning("test warning"));

        assert!(diagnostics.has_errors());
        assert_eq!(diagnostics.messages().len(), 2);
    }
}
