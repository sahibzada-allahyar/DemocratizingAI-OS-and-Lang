//! Abstract Syntax Tree definitions for the Democratising programming language
//!
//! This module defines the AST node types that represent the structure of
//! programs written in the Democratising language.

use crate::error::SourceLocation;
use std::fmt;

/// A complete source file
#[derive(Debug, Clone)]
pub struct Program {
    pub items: Vec<Item>,
    pub location: SourceLocation,
}

/// Top-level items in a program
#[derive(Debug, Clone)]
pub enum Item {
    Function(Function),
    Struct(Struct),
    Impl(Impl),
    Trait(Trait),
    Module(Module),
    Use(Use),
    Type(TypeAlias),
    // AI-specific items
    Model(Model),
    Layer(Layer),
}

/// A function definition
#[derive(Debug, Clone)]
pub struct Function {
    pub name: Identifier,
    pub params: Vec<Parameter>,
    pub return_type: Option<Type>,
    pub body: Block,
    pub is_async: bool,
    pub visibility: Visibility,
    pub location: SourceLocation,
}

/// A struct definition
#[derive(Debug, Clone)]
pub struct Struct {
    pub name: Identifier,
    pub fields: Vec<Field>,
    pub visibility: Visibility,
    pub location: SourceLocation,
}

/// An implementation block
#[derive(Debug, Clone)]
pub struct Impl {
    pub type_name: Type,
    pub trait_name: Option<Type>,
    pub items: Vec<ImplItem>,
    pub location: SourceLocation,
}

/// A trait definition
#[derive(Debug, Clone)]
pub struct Trait {
    pub name: Identifier,
    pub items: Vec<TraitItem>,
    pub visibility: Visibility,
    pub location: SourceLocation,
}

/// A module definition
#[derive(Debug, Clone)]
pub struct Module {
    pub name: Identifier,
    pub items: Vec<Item>,
    pub visibility: Visibility,
    pub location: SourceLocation,
}

/// A use declaration
#[derive(Debug, Clone)]
pub struct Use {
    pub path: Vec<Identifier>,
    pub alias: Option<Identifier>,
    pub visibility: Visibility,
    pub location: SourceLocation,
}

/// A type alias
#[derive(Debug, Clone)]
pub struct TypeAlias {
    pub name: Identifier,
    pub type_params: Vec<TypeParameter>,
    pub value: Type,
    pub visibility: Visibility,
    pub location: SourceLocation,
}

/// An AI model definition
#[derive(Debug, Clone)]
pub struct Model {
    pub name: Identifier,
    pub layers: Vec<Layer>,
    pub config: Option<Block>,
    pub visibility: Visibility,
    pub location: SourceLocation,
}

/// A neural network layer
#[derive(Debug, Clone)]
pub struct Layer {
    pub name: Identifier,
    pub params: Vec<Parameter>,
    pub body: Block,
    pub visibility: Visibility,
    pub location: SourceLocation,
}

/// A parameter in a function or layer definition
#[derive(Debug, Clone)]
pub struct Parameter {
    pub name: Identifier,
    pub type_: Type,
    pub default_value: Option<Expression>,
    pub location: SourceLocation,
}

/// A field in a struct
#[derive(Debug, Clone)]
pub struct Field {
    pub name: Identifier,
    pub type_: Type,
    pub visibility: Visibility,
    pub location: SourceLocation,
}

/// An item in an impl block
#[derive(Debug, Clone)]
pub enum ImplItem {
    Method(Function),
    Constant(Constant),
    Type(TypeAlias),
}

/// An item in a trait definition
#[derive(Debug, Clone)]
pub enum TraitItem {
    Method(TraitMethod),
    Constant(TraitConstant),
    Type(TraitType),
}

/// A method signature in a trait
#[derive(Debug, Clone)]
pub struct TraitMethod {
    pub name: Identifier,
    pub params: Vec<Parameter>,
    pub return_type: Option<Type>,
    pub default_impl: Option<Block>,
    pub is_async: bool,
    pub location: SourceLocation,
}

/// A constant in a trait
#[derive(Debug, Clone)]
pub struct TraitConstant {
    pub name: Identifier,
    pub type_: Type,
    pub default_value: Option<Expression>,
    pub location: SourceLocation,
}

/// An associated type in a trait
#[derive(Debug, Clone)]
pub struct TraitType {
    pub name: Identifier,
    pub bounds: Vec<Type>,
    pub default: Option<Type>,
    pub location: SourceLocation,
}

/// A constant definition
#[derive(Debug, Clone)]
pub struct Constant {
    pub name: Identifier,
    pub type_: Type,
    pub value: Expression,
    pub visibility: Visibility,
    pub location: SourceLocation,
}

/// A block of statements
#[derive(Debug, Clone)]
pub struct Block {
    pub statements: Vec<Statement>,
    pub location: SourceLocation,
}

/// A statement
#[derive(Debug, Clone)]
pub enum Statement {
    Let(Let),
    Expression(Expression),
    Return(Return),
    Break(Break),
    Continue(Continue),
    // AI-specific statements
    Train(Train),
    Infer(Infer),
}

/// A let binding
#[derive(Debug, Clone)]
pub struct Let {
    pub pattern: Pattern,
    pub type_: Option<Type>,
    pub value: Expression,
    pub location: SourceLocation,
}

/// A return statement
#[derive(Debug, Clone)]
pub struct Return {
    pub value: Option<Expression>,
    pub location: SourceLocation,
}

/// A break statement
#[derive(Debug, Clone)]
pub struct Break {
    pub value: Option<Expression>,
    pub location: SourceLocation,
}

/// A continue statement
#[derive(Debug, Clone)]
pub struct Continue {
    pub location: SourceLocation,
}

/// A training statement
#[derive(Debug, Clone)]
pub struct Train {
    pub model: Expression,
    pub data: Expression,
    pub config: Option<Expression>,
    pub location: SourceLocation,
}

/// An inference statement
#[derive(Debug, Clone)]
pub struct Infer {
    pub model: Expression,
    pub input: Expression,
    pub location: SourceLocation,
}

/// An expression
#[derive(Debug, Clone)]
pub enum Expression {
    Literal(Literal),
    Identifier(Identifier),
    Binary(Box<Binary>),
    Unary(Box<Unary>),
    Call(Box<Call>),
    MethodCall(Box<MethodCall>),
    Field(Box<Field>),
    Index(Box<Index>),
    Array(Box<Array>),
    Tuple(Box<Tuple>),
    Block(Box<Block>),
    If(Box<If>),
    While(Box<While>),
    For(Box<For>),
    Match(Box<Match>),
    Lambda(Box<Lambda>),
    // AI-specific expressions
    Tensor(Box<Tensor>),
    Gradient(Box<Gradient>),
}

/// A literal value
#[derive(Debug, Clone)]
pub enum Literal {
    Integer(i64),
    Float(f64),
    String(String),
    Boolean(bool),
    Char(char),
}

/// A binary operation
#[derive(Debug, Clone)]
pub struct Binary {
    pub left: Expression,
    pub operator: BinaryOperator,
    pub right: Expression,
    pub location: SourceLocation,
}

/// A unary operation
#[derive(Debug, Clone)]
pub struct Unary {
    pub operator: UnaryOperator,
    pub operand: Expression,
    pub location: SourceLocation,
}

/// A function or method call
#[derive(Debug, Clone)]
pub struct Call {
    pub function: Expression,
    pub arguments: Vec<Expression>,
    pub location: SourceLocation,
}

/// A method call
#[derive(Debug, Clone)]
pub struct MethodCall {
    pub receiver: Expression,
    pub method: Identifier,
    pub arguments: Vec<Expression>,
    pub location: SourceLocation,
}

/// A field access
#[derive(Debug, Clone)]
pub struct FieldAccess {
    pub value: Expression,
    pub field: Identifier,
    pub location: SourceLocation,
}

/// An array indexing operation
#[derive(Debug, Clone)]
pub struct Index {
    pub array: Expression,
    pub index: Expression,
    pub location: SourceLocation,
}

/// An array literal
#[derive(Debug, Clone)]
pub struct Array {
    pub elements: Vec<Expression>,
    pub location: SourceLocation,
}

/// A tuple literal
#[derive(Debug, Clone)]
pub struct Tuple {
    pub elements: Vec<Expression>,
    pub location: SourceLocation,
}

/// An if expression
#[derive(Debug, Clone)]
pub struct If {
    pub condition: Expression,
    pub then_branch: Block,
    pub else_branch: Option<Block>,
    pub location: SourceLocation,
}

/// A while loop
#[derive(Debug, Clone)]
pub struct While {
    pub condition: Expression,
    pub body: Block,
    pub location: SourceLocation,
}

/// A for loop
#[derive(Debug, Clone)]
pub struct For {
    pub pattern: Pattern,
    pub iterator: Expression,
    pub body: Block,
    pub location: SourceLocation,
}

/// A match expression
#[derive(Debug, Clone)]
pub struct Match {
    pub value: Expression,
    pub arms: Vec<MatchArm>,
    pub location: SourceLocation,
}

/// A match arm
#[derive(Debug, Clone)]
pub struct MatchArm {
    pub pattern: Pattern,
    pub guard: Option<Expression>,
    pub body: Expression,
    pub location: SourceLocation,
}

/// A lambda expression
#[derive(Debug, Clone)]
pub struct Lambda {
    pub params: Vec<Parameter>,
    pub body: Expression,
    pub location: SourceLocation,
}

/// A tensor literal or operation
#[derive(Debug, Clone)]
pub struct Tensor {
    pub elements: Vec<Expression>,
    pub shape: Vec<Expression>,
    pub location: SourceLocation,
}

/// A gradient computation
#[derive(Debug, Clone)]
pub struct Gradient {
    pub value: Expression,
    pub variables: Vec<Expression>,
    pub location: SourceLocation,
}

/// A pattern in a match expression or let binding
#[derive(Debug, Clone)]
pub enum Pattern {
    Literal(Literal),
    Identifier(Identifier),
    Tuple(Vec<Pattern>),
    Struct {
        name: Type,
        fields: Vec<(Identifier, Pattern)>,
    },
    Array(Vec<Pattern>),
    Rest,
}

/// A type expression
#[derive(Debug, Clone)]
pub enum Type {
    Named(Vec<Identifier>),
    Array(Box<Type>, Option<Expression>),
    Tuple(Vec<Type>),
    Function {
        params: Vec<Type>,
        return_type: Box<Type>,
    },
    Generic(Identifier, Vec<Type>),
    // AI-specific types
    Tensor(Box<Type>, Vec<Expression>),
    Model(Vec<Type>),
}

/// A type parameter
#[derive(Debug, Clone)]
pub struct TypeParameter {
    pub name: Identifier,
    pub bounds: Vec<Type>,
    pub default: Option<Type>,
    pub location: SourceLocation,
}

/// Binary operators
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOperator {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    And,
    Or,
    BitAnd,
    BitOr,
    BitXor,
    Shl,
    Shr,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

/// Unary operators
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOperator {
    Neg,
    Not,
    BitNot,
}

/// Visibility level
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Visibility {
    Public,
    Private,
}

/// An identifier
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Identifier {
    pub name: String,
    pub location: SourceLocation,
}

impl fmt::Display for Identifier {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.name)
    }
}

/// Visitor trait for traversing the AST
pub trait Visitor {
    fn visit_program(&mut self, program: &Program);
    fn visit_item(&mut self, item: &Item);
    fn visit_function(&mut self, function: &Function);
    fn visit_struct(&mut self, struct_: &Struct);
    fn visit_impl(&mut self, impl_: &Impl);
    fn visit_trait(&mut self, trait_: &Trait);
    fn visit_module(&mut self, module: &Module);
    fn visit_use(&mut self, use_: &Use);
    fn visit_type_alias(&mut self, alias: &TypeAlias);
    fn visit_model(&mut self, model: &Model);
    fn visit_layer(&mut self, layer: &Layer);
    fn visit_statement(&mut self, statement: &Statement);
    fn visit_expression(&mut self, expression: &Expression);
    fn visit_pattern(&mut self, pattern: &Pattern);
    fn visit_type(&mut self, type_: &Type);
}

/// Default implementations for the visitor trait
pub trait DefaultVisitor: Visitor {
    fn default_visit_program(&mut self, program: &Program) {
        for item in &program.items {
            self.visit_item(item);
        }
    }

    fn default_visit_item(&mut self, item: &Item) {
        match item {
            Item::Function(f) => self.visit_function(f),
            Item::Struct(s) => self.visit_struct(s),
            Item::Impl(i) => self.visit_impl(i),
            Item::Trait(t) => self.visit_trait(t),
            Item::Module(m) => self.visit_module(m),
            Item::Use(u) => self.visit_use(u),
            Item::Type(t) => self.visit_type_alias(t),
            Item::Model(m) => self.visit_model(m),
            Item::Layer(l) => self.visit_layer(l),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ast_construction() {
        let loc = SourceLocation {
            file: "test".into(),
            line: 1,
            column: 1,
        };

        let program = Program {
            items: vec![Item::Function(Function {
                name: Identifier {
                    name: "main".into(),
                    location: loc.clone(),
                },
                params: vec![],
                return_type: None,
                body: Block {
                    statements: vec![],
                    location: loc.clone(),
                },
                is_async: false,
                visibility: Visibility::Public,
                location: loc,
            })],
            location: loc,
        };

        assert_eq!(program.items.len(), 1);
    }
}
