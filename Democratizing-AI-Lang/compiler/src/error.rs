//! Error handling for the Democratising compiler
//!
//! This module defines the error types and utilities for error handling
//! throughout the compiler.

use std::fmt;

/// A location in source code
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceLocation {
    /// The source file path
    pub file: String,
    /// The line number (1-based)
    pub line: usize,
    /// The column number (1-based)
    pub column: usize,
}

impl fmt::Display for SourceLocation {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}:{}:{}", self.file, self.line, self.column)
    }
}

/// The type of compiler error
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ErrorKind {
    /// Lexical analysis error
    Lexical,
    /// Syntax error
    Syntax,
    /// Type error
    Type,
    /// Name resolution error
    Name,
    /// IO error
    IO,
    /// Internal compiler error
    Internal,
}

impl fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ErrorKind::Lexical => write!(f, "lexical error"),
            ErrorKind::Syntax => write!(f, "syntax error"),
            ErrorKind::Type => write!(f, "type error"),
            ErrorKind::Name => write!(f, "name error"),
            ErrorKind::IO => write!(f, "I/O error"),
            ErrorKind::Internal => write!(f, "internal error"),
        }
    }
}

/// A compiler error
#[derive(Debug, Clone)]
pub struct CompilerError {
    /// The kind of error
    pub kind: ErrorKind,
    /// The location where the error occurred
    pub location: Option<SourceLocation>,
    /// The error message
    pub message: String,
    /// Optional notes providing additional context
    pub notes: Vec<String>,
}

impl CompilerError {
    /// Create a new compiler error
    pub fn new(
        kind: ErrorKind,
        location: Option<SourceLocation>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            kind,
            location,
            message: message.into(),
            notes: Vec::new(),
        }
    }

    /// Add a note to the error
    pub fn with_note(mut self, note: impl Into<String>) -> Self {
        self.notes.push(note.into());
        self
    }

    /// Create a lexical analysis error
    pub fn lex_error(file: &str, line: usize, column: usize, message: impl Into<String>) -> Self {
        Self::new(
            ErrorKind::Lexical,
            Some(SourceLocation {
                file: file.to_string(),
                line,
                column,
            }),
            message,
        )
    }

    /// Create a syntax error
    pub fn syntax_error(file: &str, line: usize, column: usize, message: impl Into<String>) -> Self {
        Self::new(
            ErrorKind::Syntax,
            Some(SourceLocation {
                file: file.to_string(),
                line,
                column,
            }),
            message,
        )
    }

    /// Create a type error
    pub fn type_error(file: &str, line: usize, column: usize, message: impl Into<String>) -> Self {
        Self::new(
            ErrorKind::Type,
            Some(SourceLocation {
                file: file.to_string(),
                line,
                column,
            }),
            message,
        )
    }

    /// Create a name resolution error
    pub fn name_error(file: &str, line: usize, column: usize, message: impl Into<String>) -> Self {
        Self::new(
            ErrorKind::Name,
            Some(SourceLocation {
                file: file.to_string(),
                line,
                column,
            }),
            message,
        )
    }

    /// Create an I/O error
    pub fn io_error(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::IO, None, message)
    }

    /// Create an internal compiler error
    pub fn internal_error(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Internal, None, message)
    }
}

impl std::error::Error for CompilerError {}

impl fmt::Display for CompilerError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if let Some(location) = &self.location {
            write!(f, "{}: {}: {}", location, self.kind, self.message)?;
        } else {
            write!(f, "{}: {}", self.kind, self.message)?;
        }

        for note in &self.notes {
            write!(f, "\nnote: {}", note)?;
        }

        Ok(())
    }
}

/// Result type for compiler operations
pub type Result<T> = std::result::Result<T, CompilerError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_source_location_display() {
        let loc = SourceLocation {
            file: "test.dem".to_string(),
            line: 42,
            column: 10,
        };
        assert_eq!(loc.to_string(), "test.dem:42:10");
    }

    #[test]
    fn test_error_construction() {
        let error = CompilerError::lex_error("test.dem", 1, 1, "unexpected character")
            .with_note("expected a digit");

        assert_eq!(error.kind, ErrorKind::Lexical);
        assert!(error.location.is_some());
        assert_eq!(error.notes.len(), 1);
        assert!(error.to_string().contains("unexpected character"));
        assert!(error.to_string().contains("expected a digit"));
    }

    #[test]
    fn test_error_without_location() {
        let error = CompilerError::io_error("failed to read file");
        assert_eq!(error.kind, ErrorKind::IO);
        assert!(error.location.is_none());
        assert!(error.to_string().contains("failed to read file"));
    }
}
