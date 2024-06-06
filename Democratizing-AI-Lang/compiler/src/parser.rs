//! Parser for the Democratising programming language
//!
//! This module implements a recursive descent parser that converts
//! a stream of tokens into an Abstract Syntax Tree (AST).

use crate::ast::*;
use crate::error::{CompilerError, Result, SourceLocation};
use crate::lexer::{Token, TokenKind};
use std::iter::Peekable;
use std::vec::IntoIter;

// ... [Previous code remains unchanged up to parse_identifier] ...

impl Parser {
    // ... [Previous methods remain unchanged] ...

    /// Parse a type
    fn parse_type(&mut self) -> Result<Type> {
        let mut base = match self.peek_kind() {
            Some(TokenKind::Identifier(_)) => {
                let mut path = vec![self.parse_identifier()?];
                while self.match_token(TokenKind::Colon) && self.match_token(TokenKind::Colon) {
                    path.push(self.parse_identifier()?);
                }
                Type::Named(path)
            }
            Some(TokenKind::LeftBracket) => {
                self.advance(); // consume [
                let element_type = Box::new(self.parse_type()?);
                let size = if self.match_token(TokenKind::Semicolon) {
                    Some(self.parse_expression()?)
                } else {
                    None
                };
                self.expect_token(TokenKind::RightBracket)?;
                Type::Array(element_type, size)
            }
            Some(TokenKind::LeftParen) => {
                self.advance(); // consume (
                let types = self.parse_separated(TokenKind::RightParen, |p| p.parse_type())?;
                Type::Tuple(types)
            }
            Some(TokenKind::Fn) => {
                self.advance(); // consume fn
                self.expect_token(TokenKind::LeftParen)?;
                let params = self.parse_separated(TokenKind::RightParen, |p| p.parse_type())?;
                self.expect_token(TokenKind::Arrow)?;
                let return_type = Box::new(self.parse_type()?);
                Type::Function {
                    params,
                    return_type,
                }
            }
            Some(TokenKind::Tensor) => {
                self.advance(); // consume tensor
                self.expect_token(TokenKind::Less)?;
                let element_type = Box::new(self.parse_type()?);
                self.expect_token(TokenKind::Greater)?;
                self.expect_token(TokenKind::LeftBracket)?;
                let dims = self.parse_separated(TokenKind::RightBracket, |p| p.parse_expression())?;
                Type::Tensor(element_type, dims)
            }
            Some(TokenKind::Model) => {
                self.advance(); // consume model
                self.expect_token(TokenKind::Less)?;
                let types = self.parse_separated(TokenKind::Greater, |p| p.parse_type())?;
                Type::Model(types)
            }
            Some(kind) => {
                return Err(self.error(format!("unexpected token in type position: {:?}", kind)))
            }
            None => {
                return Err(self.error("unexpected end of file"))
            }
        };

        // Parse generic type arguments if present
        if self.match_token(TokenKind::Less) {
            let name = match base {
                Type::Named(mut path) => {
                    if path.len() != 1 {
                        return Err(self.error("generic type arguments can only be applied to simple types"));
                    }
                    path.pop().unwrap()
                }
                _ => return Err(self.error("generic type arguments can only be applied to named types")),
            };
            let args = self.parse_separated(TokenKind::Greater, |p| p.parse_type())?;
            base = Type::Generic(name, args);
        }

        Ok(base)
    }

    /// Parse an expression
    fn parse_expression(&mut self) -> Result<Expression> {
        self.parse_assignment()
    }

    /// Parse an assignment expression
    fn parse_assignment(&mut self) -> Result<Expression> {
        let expr = self.parse_logical_or()?;

        if self.match_token(TokenKind::Equal) {
            let value = self.parse_assignment()?;
            match expr {
                Expression::Identifier(_) | Expression::Field(_) | Expression::Index(_) => {
                    Ok(Expression::Binary(Box::new(Binary {
                        left: expr,
                        operator: BinaryOperator::Eq,
                        right: value,
                        location: self.current_location(),
                    })))
                }
                _ => Err(self.error("invalid assignment target")),
            }
        } else {
            Ok(expr)
        }
    }

    /// Parse a logical OR expression
    fn parse_logical_or(&mut self) -> Result<Expression> {
        let mut expr = self.parse_logical_and()?;

        while self.match_token(TokenKind::Or) {
            let right = self.parse_logical_and()?;
            expr = Expression::Binary(Box::new(Binary {
                left: expr,
                operator: BinaryOperator::Or,
                right,
                location: self.current_location(),
            }));
        }

        Ok(expr)
    }

    /// Parse a logical AND expression
    fn parse_logical_and(&mut self) -> Result<Expression> {
        let mut expr = self.parse_equality()?;

        while self.match_token(TokenKind::And) {
            let right = self.parse_equality()?;
            expr = Expression::Binary(Box::new(Binary {
                left: expr,
                operator: BinaryOperator::And,
                right,
                location: self.current_location(),
            }));
        }

        Ok(expr)
    }

    /// Parse an equality expression
    fn parse_equality(&mut self) -> Result<Expression> {
        let mut expr = self.parse_comparison()?;

        while let Some(op) = self.match_equality_op() {
            let right = self.parse_comparison()?;
            expr = Expression::Binary(Box::new(Binary {
                left: expr,
                operator: op,
                right,
                location: self.current_location(),
            }));
        }

        Ok(expr)
    }

    /// Parse a comparison expression
    fn parse_comparison(&mut self) -> Result<Expression> {
        let mut expr = self.parse_term()?;

        while let Some(op) = self.match_comparison_op() {
            let right = self.parse_term()?;
            expr = Expression::Binary(Box::new(Binary {
                left: expr,
                operator: op,
                right,
                location: self.current_location(),
            }));
        }

        Ok(expr)
    }

    /// Parse a term (addition/subtraction)
    fn parse_term(&mut self) -> Result<Expression> {
        let mut expr = self.parse_factor()?;

        while let Some(op) = self.match_term_op() {
            let right = self.parse_factor()?;
            expr = Expression::Binary(Box::new(Binary {
                left: expr,
                operator: op,
                right,
                location: self.current_location(),
            }));
        }

        Ok(expr)
    }

    /// Parse a factor (multiplication/division)
    fn parse_factor(&mut self) -> Result<Expression> {
        let mut expr = self.parse_unary()?;

        while let Some(op) = self.match_factor_op() {
            let right = self.parse_unary()?;
            expr = Expression::Binary(Box::new(Binary {
                left: expr,
                operator: op,
                right,
                location: self.current_location(),
            }));
        }

        Ok(expr)
    }

    /// Parse a unary expression
    fn parse_unary(&mut self) -> Result<Expression> {
        if let Some(op) = self.match_unary_op() {
            let operand = self.parse_unary()?;
            Ok(Expression::Unary(Box::new(Unary {
                operator: op,
                operand,
                location: self.current_location(),
            })))
        } else {
            self.parse_call()
        }
    }

    /// Parse a call expression
    fn parse_call(&mut self) -> Result<Expression> {
        let mut expr = self.parse_primary()?;

        loop {
            expr = match self.peek_kind() {
                Some(TokenKind::LeftParen) => {
                    self.advance(); // consume (
                    let args = self.parse_separated(TokenKind::RightParen, |p| p.parse_expression())?;
                    Expression::Call(Box::new(Call {
                        function: expr,
                        arguments: args,
                        location: self.current_location(),
                    }))
                }
                Some(TokenKind::Dot) => {
                    self.advance(); // consume .
                    let method = self.parse_identifier()?;
                    if self.check(TokenKind::LeftParen) {
                        self.advance(); // consume (
                        let args = self.parse_separated(TokenKind::RightParen, |p| p.parse_expression())?;
                        Expression::MethodCall(Box::new(MethodCall {
                            receiver: expr,
                            method,
                            arguments: args,
                            location: self.current_location(),
                        }))
                    } else {
                        Expression::Field(Box::new(FieldAccess {
                            value: expr,
                            field: method,
                            location: self.current_location(),
                        }))
                    }
                }
                Some(TokenKind::LeftBracket) => {
                    self.advance(); // consume [
                    let index = self.parse_expression()?;
                    self.expect_token(TokenKind::RightBracket)?;
                    Expression::Index(Box::new(Index {
                        array: expr,
                        index,
                        location: self.current_location(),
                    }))
                }
                _ => break,
            };
        }

        Ok(expr)
    }

    /// Parse a primary expression
    fn parse_primary(&mut self) -> Result<Expression> {
        match self.peek_kind() {
            Some(TokenKind::Integer(n)) => {
                self.advance();
                Ok(Expression::Literal(Literal::Integer(n)))
            }
            Some(TokenKind::Float(n)) => {
                self.advance();
                Ok(Expression::Literal(Literal::Float(n)))
            }
            Some(TokenKind::String(s)) => {
                self.advance();
                Ok(Expression::Literal(Literal::String(s)))
            }
            Some(TokenKind::Boolean(b)) => {
                self.advance();
                Ok(Expression::Literal(Literal::Boolean(b)))
            }
            Some(TokenKind::Identifier(_)) => {
                Ok(Expression::Identifier(self.parse_identifier()?))
            }
            Some(TokenKind::LeftParen) => {
                self.advance(); // consume (
                let expr = self.parse_expression()?;
                self.expect_token(TokenKind::RightParen)?;
                Ok(expr)
            }
            Some(TokenKind::LeftBracket) => {
                self.advance(); // consume [
                let elements = self.parse_separated(TokenKind::RightBracket, |p| p.parse_expression())?;
                Ok(Expression::Array(Box::new(Array {
                    elements,
                    location: self.current_location(),
                })))
            }
            Some(kind) => {
                Err(self.error(format!("unexpected token in expression position: {:?}", kind)))
            }
            None => {
                Err(self.error("unexpected end of file"))
            }
        }
    }

    // Helper methods for parsing operators

    fn match_equality_op(&mut self) -> Option<BinaryOperator> {
        if self.match_token(TokenKind::EqualEqual) {
            Some(BinaryOperator::Eq)
        } else if self.match_token(TokenKind::BangEqual) {
            Some(BinaryOperator::Ne)
        } else {
            None
        }
    }

    fn match_comparison_op(&mut self) -> Option<BinaryOperator> {
        if self.match_token(TokenKind::Less) {
            Some(BinaryOperator::Lt)
        } else if self.match_token(TokenKind::LessEqual) {
            Some(BinaryOperator::Le)
        } else if self.match_token(TokenKind::Greater) {
            Some(BinaryOperator::Gt)
        } else if self.match_token(TokenKind::GreaterEqual) {
            Some(BinaryOperator::Ge)
        } else {
            None
        }
    }

    fn match_term_op(&mut self) -> Option<BinaryOperator> {
        if self.match_token(TokenKind::Plus) {
            Some(BinaryOperator::Add)
        } else if self.match_token(TokenKind::Minus) {
            Some(BinaryOperator::Sub)
        } else {
            None
        }
    }

    fn match_factor_op(&mut self) -> Option<BinaryOperator> {
        if self.match_token(TokenKind::Star) {
            Some(BinaryOperator::Mul)
        } else if self.match_token(TokenKind::Slash) {
            Some(BinaryOperator::Div)
        } else if self.match_token(TokenKind::Percent) {
            Some(BinaryOperator::Rem)
        } else {
            None
        }
    }

    fn match_unary_op(&mut self) -> Option<UnaryOperator> {
        if self.match_token(TokenKind::Minus) {
            Some(UnaryOperator::Neg)
        } else if self.match_token(TokenKind::Bang) {
            Some(UnaryOperator::Not)
        } else {
            None
        }
    }
}

// ... [Previous tests remain unchanged] ...

#[cfg(test)]
mod tests {
    use super::*;

    // ... [Previous test functions remain unchanged] ...

    #[test]
    fn test_expression_parsing() {
        let source = "x + y * z";
        let program = parse("let a = x + y * z;").unwrap();

        match &program.items[0] {
            Item::Let(l) => {
                match &l.value {
                    Expression::Binary(b) => {
                        assert_eq!(b.operator, BinaryOperator::Add);
                        match &b.right {
                            Expression::Binary(b2) => {
                                assert_eq!(b2.operator, BinaryOperator::Mul);
                            }
                            _ => panic!("Expected multiplication"),
                        }
                    }
                    _ => panic!("Expected binary expression"),
                }
            }
            _ => panic!("Expected let statement"),
        }
    }

    #[test]
    fn test_function_call() {
        let source = "foo(1, 2 + 3)";
        let expr = parse_expression(source).unwrap();

        match expr {
            Expression::Call(c) => {
                assert_eq!(c.arguments.len(), 2);
                match &c.arguments[1] {
                    Expression::Binary(b) => {
                        assert_eq!(b.operator, BinaryOperator::Add);
                    }
                    _ => panic!("Expected binary expression"),
                }
            }
            _ => panic!("Expected function call"),
        }
    }

    #[test]
    fn test_method_call() {
        let source = "obj.method(arg)";
        let expr = parse_expression(source).unwrap();

        match expr {
            Expression::MethodCall(m) => {
                assert_eq!(m.method.name, "method");
                assert_eq!(m.arguments.len(), 1);
            }
            _ => panic!("Expected method call"),
        }
    }
}
