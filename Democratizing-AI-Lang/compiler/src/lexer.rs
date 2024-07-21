//! Lexical analyzer for the Democratising programming language
//!
//! This module implements the lexer (tokenizer) that converts source text into a stream of tokens.

use crate::error::{CompilerError, SourceLocation};
use std::iter::Peekable;
use std::str::Chars;

/// Token types for the Democratising language
#[derive(Debug, Clone, PartialEq)]
pub enum TokenKind {
    // Keywords
    Let,
    Fn,
    If,
    Else,
    While,
    For,
    In,
    Return,
    Break,
    Continue,
    Struct,
    Impl,
    Trait,
    Pub,
    Use,
    Mod,
    Type,
    Async,
    Await,

    // AI-specific keywords
    Tensor,
    Model,
    Train,
    Infer,
    Layer,
    Optimize,
    Gradient,

    // Literals
    Integer(i64),
    Float(f64),
    String(String),
    Boolean(bool),

    // Identifiers
    Identifier(String),

    // Operators
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    Equal,
    EqualEqual,
    Bang,
    BangEqual,
    Greater,
    GreaterEqual,
    Less,
    LessEqual,
    And,
    Or,

    // Delimiters
    LeftParen,
    RightParen,
    LeftBrace,
    RightBrace,
    LeftBracket,
    RightBracket,
    Comma,
    Dot,
    Semicolon,
    Colon,
    Arrow,

    // Special
    EOF,
}

/// A token with its location in the source
#[derive(Debug, Clone)]
pub struct Token {
    pub kind: TokenKind,
    pub lexeme: String,
    pub location: SourceLocation,
}

/// The lexer state
pub struct Lexer<'a> {
    source: &'a str,
    chars: Peekable<Chars<'a>>,
    file: String,
    line: usize,
    column: usize,
    start_column: usize,
}

impl<'a> Lexer<'a> {
    /// Create a new lexer for the given source
    pub fn new(source: &'a str, file: impl Into<String>) -> Self {
        Self {
            source,
            chars: source.chars().peekable(),
            file: file.into(),
            line: 1,
            column: 1,
            start_column: 1,
        }
    }

    /// Get the next token from the source
    pub fn next_token(&mut self) -> Result<Token, CompilerError> {
        self.skip_whitespace();

        self.start_column = self.column;

        let c = match self.advance() {
            Some(c) => c,
            None => return Ok(self.make_token(TokenKind::EOF)),
        };

        match c {
            // Single-character tokens
            '(' => Ok(self.make_token(TokenKind::LeftParen)),
            ')' => Ok(self.make_token(TokenKind::RightParen)),
            '{' => Ok(self.make_token(TokenKind::LeftBrace)),
            '}' => Ok(self.make_token(TokenKind::RightBrace)),
            '[' => Ok(self.make_token(TokenKind::LeftBracket)),
            ']' => Ok(self.make_token(TokenKind::RightBracket)),
            ',' => Ok(self.make_token(TokenKind::Comma)),
            '.' => Ok(self.make_token(TokenKind::Dot)),
            ';' => Ok(self.make_token(TokenKind::Semicolon)),
            ':' => Ok(self.make_token(TokenKind::Colon)),

            // One or two character tokens
            '=' => {
                let kind = if self.match_char('=') {
                    TokenKind::EqualEqual
                } else {
                    TokenKind::Equal
                };
                Ok(self.make_token(kind))
            }
            '!' => {
                let kind = if self.match_char('=') {
                    TokenKind::BangEqual
                } else {
                    TokenKind::Bang
                };
                Ok(self.make_token(kind))
            }
            '>' => {
                let kind = if self.match_char('=') {
                    TokenKind::GreaterEqual
                } else {
                    TokenKind::Greater
                };
                Ok(self.make_token(kind))
            }
            '<' => {
                let kind = if self.match_char('=') {
                    TokenKind::LessEqual
                } else {
                    TokenKind::Less
                };
                Ok(self.make_token(kind))
            }

            // Numbers
            '0'..='9' => self.number(),

            // Identifiers and keywords
            'a'..='z' | 'A'..='Z' | '_' => self.identifier(),

            // String literals
            '"' => self.string(),

            // Unknown character
            _ => Err(CompilerError::lex_error(
                &self.file,
                self.line,
                self.column,
                format!("unexpected character: {}", c),
            )),
        }
    }

    /// Advance the lexer by one character
    fn advance(&mut self) -> Option<char> {
        let c = self.chars.next();
        if let Some(c) = c {
            self.column += 1;
            if c == '\n' {
                self.line += 1;
                self.column = 1;
            }
        }
        c
    }

    /// Look at the next character without consuming it
    fn peek(&mut self) -> Option<char> {
        self.chars.peek().copied()
    }

    /// Check if the next character matches and consume it if it does
    fn match_char(&mut self, expected: char) -> bool {
        match self.peek() {
            Some(c) if c == expected => {
                self.advance();
                true
            }
            _ => false,
        }
    }

    /// Skip whitespace and comments
    fn skip_whitespace(&mut self) {
        while let Some(c) = self.peek() {
            match c {
                ' ' | '\r' | '\t' | '\n' => {
                    self.advance();
                }
                '/' if self.chars.clone().nth(1) == Some('/') => {
                    // Line comment
                    while let Some(c) = self.peek() {
                        if c == '\n' {
                            break;
                        }
                        self.advance();
                    }
                }
                _ => break,
            }
        }
    }

    /// Parse a number literal
    fn number(&mut self) -> Result<Token, CompilerError> {
        let mut is_float = false;

        while let Some(c) = self.peek() {
            match c {
                '0'..='9' => {
                    self.advance();
                }
                '.' if !is_float => {
                    is_float = true;
                    self.advance();
                }
                _ => break,
            }
        }

        let lexeme = self.current_lexeme();
        let kind = if is_float {
            match lexeme.parse::<f64>() {
                Ok(n) => TokenKind::Float(n),
                Err(_) => {
                    return Err(CompilerError::lex_error(
                        &self.file,
                        self.line,
                        self.start_column,
                        "invalid float literal",
                    ))
                }
            }
        } else {
            match lexeme.parse::<i64>() {
                Ok(n) => TokenKind::Integer(n),
                Err(_) => {
                    return Err(CompilerError::lex_error(
                        &self.file,
                        self.line,
                        self.start_column,
                        "invalid integer literal",
                    ))
                }
            }
        };

        Ok(self.make_token(kind))
    }

    /// Parse an identifier or keyword
    fn identifier(&mut self) -> Result<Token, CompilerError> {
        while let Some(c) = self.peek() {
            match c {
                'a'..='z' | 'A'..='Z' | '0'..='9' | '_' => {
                    self.advance();
                }
                _ => break,
            }
        }

        let lexeme = self.current_lexeme();
        let kind = match lexeme.as_str() {
            // Keywords
            "let" => TokenKind::Let,
            "fn" => TokenKind::Fn,
            "if" => TokenKind::If,
            "else" => TokenKind::Else,
            "while" => TokenKind::While,
            "for" => TokenKind::For,
            "in" => TokenKind::In,
            "return" => TokenKind::Return,
            "break" => TokenKind::Break,
            "continue" => TokenKind::Continue,
            "struct" => TokenKind::Struct,
            "impl" => TokenKind::Impl,
            "trait" => TokenKind::Trait,
            "pub" => TokenKind::Pub,
            "use" => TokenKind::Use,
            "mod" => TokenKind::Mod,
            "type" => TokenKind::Type,
            "async" => TokenKind::Async,
            "await" => TokenKind::Await,

            // AI-specific keywords
            "tensor" => TokenKind::Tensor,
            "model" => TokenKind::Model,
            "train" => TokenKind::Train,
            "infer" => TokenKind::Infer,
            "layer" => TokenKind::Layer,
            "optimize" => TokenKind::Optimize,
            "gradient" => TokenKind::Gradient,

            // Boolean literals
            "true" => TokenKind::Boolean(true),
            "false" => TokenKind::Boolean(false),

            // Identifier
            _ => TokenKind::Identifier(lexeme.to_string()),
        };

        Ok(self.make_token(kind))
    }

    /// Parse a string literal
    fn string(&mut self) -> Result<Token, CompilerError> {
        let mut value = String::new();

        while let Some(c) = self.peek() {
            match c {
                '"' => {
                    self.advance(); // Consume closing quote
                    return Ok(self.make_token(TokenKind::String(value)));
                }
                '\\' => {
                    self.advance(); // Consume backslash
                    match self.advance() {
                        Some('n') => value.push('\n'),
                        Some('r') => value.push('\r'),
                        Some('t') => value.push('\t'),
                        Some('"') => value.push('"'),
                        Some('\\') => value.push('\\'),
                        Some(c) => {
                            return Err(CompilerError::lex_error(
                                &self.file,
                                self.line,
                                self.column,
                                format!("invalid escape sequence: \\{}", c),
                            ))
                        }
                        None => {
                            return Err(CompilerError::lex_error(
                                &self.file,
                                self.line,
                                self.column,
                                "unterminated string literal",
                            ))
                        }
                    }
                }
                _ => {
                    value.push(c);
                    self.advance();
                }
            }
        }

        Err(CompilerError::lex_error(
            &self.file,
            self.line,
            self.start_column,
            "unterminated string literal",
        ))
    }

    /// Get the lexeme for the current token
    fn current_lexeme(&self) -> &str {
        &self.source[self.start_column - 1..self.column - 1]
    }

    /// Create a token with the current lexeme
    fn make_token(&self, kind: TokenKind) -> Token {
        Token {
            kind,
            lexeme: self.current_lexeme().to_string(),
            location: SourceLocation {
                file: self.file.clone(),
                line: self.line,
                column: self.start_column,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lex(source: &str) -> Result<Vec<TokenKind>, CompilerError> {
        let mut lexer = Lexer::new(source, "test");
        let mut tokens = Vec::new();

        loop {
            let token = lexer.next_token()?;
            let kind = token.kind.clone();
            tokens.push(kind);

            if matches!(kind, TokenKind::EOF) {
                break;
            }
        }

        Ok(tokens)
    }

    #[test]
    fn test_empty() {
        assert_eq!(lex("").unwrap(), vec![TokenKind::EOF]);
    }

    #[test]
    fn test_keywords() {
        assert_eq!(
            lex("let fn if else while").unwrap(),
            vec![
                TokenKind::Let,
                TokenKind::Fn,
                TokenKind::If,
                TokenKind::Else,
                TokenKind::While,
                TokenKind::EOF
            ]
        );
    }

    #[test]
    fn test_identifiers() {
        assert_eq!(
            lex("foo bar_baz").unwrap(),
            vec![
                TokenKind::Identifier("foo".to_string()),
                TokenKind::Identifier("bar_baz".to_string()),
                TokenKind::EOF
            ]
        );
    }

    #[test]
    fn test_numbers() {
        assert_eq!(
            lex("42 3.14").unwrap(),
            vec![
                TokenKind::Integer(42),
                TokenKind::Float(3.14),
                TokenKind::EOF
            ]
        );
    }

    #[test]
    fn test_strings() {
        assert_eq!(
            lex(r#""hello" "world\n""#).unwrap(),
            vec![
                TokenKind::String("hello".to_string()),
                TokenKind::String("world\n".to_string()),
                TokenKind::EOF
            ]
        );
    }

    #[test]
    fn test_operators() {
        assert_eq!(
            lex("+ - * / = == != > >= < <=").unwrap(),
            vec![
                TokenKind::Plus,
                TokenKind::Minus,
                TokenKind::Star,
                TokenKind::Slash,
                TokenKind::Equal,
                TokenKind::EqualEqual,
                TokenKind::BangEqual,
                TokenKind::Greater,
                TokenKind::GreaterEqual,
                TokenKind::Less,
                TokenKind::LessEqual,
                TokenKind::EOF
            ]
        );
    }

    #[test]
    fn test_delimiters() {
        assert_eq!(
            lex("( ) { } [ ] , . ; :").unwrap(),
            vec![
                TokenKind::LeftParen,
                TokenKind::RightParen,
                TokenKind::LeftBrace,
                TokenKind::RightBrace,
                TokenKind::LeftBracket,
                TokenKind::RightBracket,
                TokenKind::Comma,
                TokenKind::Dot,
                TokenKind::Semicolon,
                TokenKind::Colon,
                TokenKind::EOF
            ]
        );
    }

    #[test]
    fn test_ai_keywords() {
        assert_eq!(
            lex("tensor model train infer layer optimize gradient").unwrap(),
            vec![
                TokenKind::Tensor,
                TokenKind::Model,
                TokenKind::Train,
                TokenKind::Infer,
                TokenKind::Layer,
                TokenKind::Optimize,
                TokenKind::Gradient,
                TokenKind::EOF
            ]
        );
    }
}
