//! Semantic analysis for the Democratising compiler
//!
//! This module performs type checking and validation on the AST.
//! It ensures programs are well-typed before code generation.

use crate::ast::*;
use crate::error::{CompilerError, Result, SourceLocation};
use std::collections::HashMap;

/// A type environment for semantic analysis
#[derive(Debug, Clone)]
pub struct TypeEnv {
    /// Map from variable names to their types
    variables: HashMap<String, Type>,
    /// Map from function names to their signatures
    functions: HashMap<String, FunctionType>,
    /// Current return type (for checking return statements)
    return_type: Option<Type>,
    /// Whether we're in a loop (for break/continue)
    in_loop: bool,
}

/// A function type with parameter and return types
#[derive(Debug, Clone)]
pub struct FunctionType {
    /// Parameter types
    params: Vec<Type>,
    /// Return type
    return_type: Type,
    /// Whether the function is async
    is_async: bool,
}

impl TypeEnv {
    /// Create a new type environment
    pub fn new() -> Self {
        Self {
            variables: HashMap::new(),
            functions: HashMap::new(),
            return_type: None,
            in_loop: false,
        }
    }

    /// Add a variable to the environment
    pub fn add_variable(&mut self, name: String, ty: Type) -> Result<()> {
        if self.variables.insert(name.clone(), ty).is_some() {
            Err(CompilerError::name_error(
                "test",
                0,
                0,
                format!("Variable {} already declared", name),
            ))
        } else {
            Ok(())
        }
    }

    /// Look up a variable's type
    pub fn lookup_variable(&self, name: &str) -> Result<Type> {
        self.variables.get(name).cloned().ok_or_else(|| {
            CompilerError::name_error("test", 0, 0, format!("Undefined variable {}", name))
        })
    }

    /// Add a function to the environment
    pub fn add_function(&mut self, name: String, ty: FunctionType) -> Result<()> {
        if self.functions.insert(name.clone(), ty).is_some() {
            Err(CompilerError::name_error(
                "test",
                0,
                0,
                format!("Function {} already declared", name),
            ))
        } else {
            Ok(())
        }
    }

    /// Look up a function's type
    pub fn lookup_function(&self, name: &str) -> Result<FunctionType> {
        self.functions.get(name).cloned().ok_or_else(|| {
            CompilerError::name_error("test", 0, 0, format!("Undefined function {}", name))
        })
    }
}

impl Default for TypeEnv {
    fn default() -> Self {
        Self::new()
    }
}

/// The semantic analyzer
pub struct SemanticAnalyzer {
    /// Current type environment
    env: TypeEnv,
}

impl SemanticAnalyzer {
    /// Create a new semantic analyzer
    pub fn new() -> Self {
        Self {
            env: TypeEnv::new(),
        }
    }

    /// Check a complete program
    pub fn check_program(&mut self, program: &Program) -> Result<()> {
        // First pass: collect all function declarations
        for item in &program.items {
            if let Item::Function(f) = item {
                self.declare_function(f)?;
            }
        }

        // Second pass: check all items
        for item in &program.items {
            self.check_item(item)?;
        }

        Ok(())
    }

    /// Declare a function in the environment
    fn declare_function(&mut self, function: &Function) -> Result<()> {
        let param_types: Vec<_> = function
            .params
            .iter()
            .map(|p| self.check_type(&p.type_))
            .collect::<Result<_>>()?;

        let return_type = if let Some(ty) = &function.return_type {
            self.check_type(ty)?
        } else {
            Type::Void
        };

        self.env.add_function(
            function.name.name.clone(),
            FunctionType {
                params: param_types,
                return_type,
                is_async: function.is_async,
            },
        )
    }

    /// Check an item
    fn check_item(&mut self, item: &Item) -> Result<()> {
        match item {
            Item::Function(f) => self.check_function(f),
            Item::Struct(s) => self.check_struct(s),
            Item::Impl(i) => self.check_impl(i),
            Item::Trait(t) => self.check_trait(t),
            Item::Module(m) => self.check_module(m),
            Item::Use(u) => self.check_use(u),
            Item::Type(t) => self.check_type_alias(t),
            Item::Model(m) => self.check_model(m),
            Item::Layer(l) => self.check_layer(l),
        }
    }

    /// Check a function definition
    fn check_function(&mut self, function: &Function) -> Result<()> {
        // Create new environment for function body
        let mut body_env = self.env.clone();

        // Add parameters to environment
        for param in &function.params {
            let param_type = self.check_type(&param.type_)?;
            body_env.add_variable(param.name.name.clone(), param_type)?;
        }

        // Set return type for checking return statements
        body_env.return_type = function.return_type.clone();

        // Check function body
        let mut analyzer = SemanticAnalyzer { env: body_env };
        analyzer.check_block(&function.body)?;

        Ok(())
    }

    /// Check a block of statements
    fn check_block(&mut self, block: &Block) -> Result<()> {
        for stmt in &block.statements {
            self.check_statement(stmt)?;
        }
        Ok(())
    }

    /// Check a statement
    fn check_statement(&mut self, stmt: &Statement) -> Result<()> {
        match stmt {
            Statement::Let(l) => self.check_let(l),
            Statement::Expression(e) => {
                self.check_expression(e)?;
                Ok(())
            }
            Statement::Return(r) => self.check_return(r),
            Statement::Break(b) => self.check_break(b),
            Statement::Continue(c) => self.check_continue(c),
            Statement::Train(t) => self.check_train(t),
            Statement::Infer(i) => self.check_infer(i),
        }
    }

    /// Check a let binding
    fn check_let(&mut self, let_: &Let) -> Result<()> {
        // Check initializer
        let value_type = self.check_expression(&let_.value)?;

        // Check type annotation if present
        if let Some(type_) = &let_.type_ {
            let annotated_type = self.check_type(type_)?;
            if value_type != annotated_type {
                return Err(CompilerError::type_error(
                    "test",
                    0,
                    0,
                    format!(
                        "Type mismatch: expected {:?}, found {:?}",
                        annotated_type, value_type
                    ),
                ));
            }
        }

        // Add variable to environment
        match &let_.pattern {
            Pattern::Identifier(ident) => {
                self.env.add_variable(ident.name.clone(), value_type)?;
            }
            _ => unimplemented!("Pattern matching not yet implemented"),
        }

        Ok(())
    }

    /// Check a return statement
    fn check_return(&self, ret: &Return) -> Result<()> {
        let expected_type = self.env.return_type.clone().unwrap_or(Type::Void);

        if let Some(value) = &ret.value {
            let actual_type = self.check_expression(value)?;
            if actual_type != expected_type {
                return Err(CompilerError::type_error(
                    "test",
                    0,
                    0,
                    format!(
                        "Return type mismatch: expected {:?}, found {:?}",
                        expected_type, actual_type
                    ),
                ));
            }
        } else if expected_type != Type::Void {
            return Err(CompilerError::type_error(
                "test",
                0,
                0,
                format!("Expected return value of type {:?}", expected_type),
            ));
        }

        Ok(())
    }

    /// Check a break statement
    fn check_break(&self, break_: &Break) -> Result<()> {
        if !self.env.in_loop {
            return Err(CompilerError::syntax_error(
                "test",
                0,
                0,
                "Break statement outside of loop",
            ));
        }
        Ok(())
    }

    /// Check a continue statement
    fn check_continue(&self, continue_: &Continue) -> Result<()> {
        if !self.env.in_loop {
            return Err(CompilerError::syntax_error(
                "test",
                0,
                0,
                "Continue statement outside of loop",
            ));
        }
        Ok(())
    }

    /// Check a train statement
    fn check_train(&mut self, train: &Train) -> Result<()> {
        // Check model expression
        let model_type = self.check_expression(&train.model)?;
        match model_type {
            Type::Model(_) => {}
            _ => {
                return Err(CompilerError::type_error(
                    "test",
                    0,
                    0,
                    "Train statement requires a model",
                ))
            }
        }

        // Check data expression
        let data_type = self.check_expression(&train.data)?;
        // TODO: Check data type matches model input type

        // Check config if present
        if let Some(config) = &train.config {
            let config_type = self.check_expression(config)?;
            // TODO: Check config type
        }

        Ok(())
    }

    /// Check an infer statement
    fn check_infer(&mut self, infer: &Infer) -> Result<()> {
        // Check model expression
        let model_type = self.check_expression(&infer.model)?;
        match model_type {
            Type::Model(_) => {}
            _ => {
                return Err(CompilerError::type_error(
                    "test",
                    0,
                    0,
                    "Infer statement requires a model",
                ))
            }
        }

        // Check input expression
        let input_type = self.check_expression(&infer.input)?;
        // TODO: Check input type matches model input type

        Ok(())
    }

    /// Check an expression and return its type
    fn check_expression(&self, expr: &Expression) -> Result<Type> {
        match expr {
            Expression::Literal(lit) => Ok(match lit {
                Literal::Integer(_) => Type::Int64,
                Literal::Float(_) => Type::Float64,
                Literal::String(_) => unimplemented!("String type not yet implemented"),
                Literal::Boolean(_) => Type::Bool,
                Literal::Char(_) => unimplemented!("Char type not yet implemented"),
            }),
            Expression::Identifier(ident) => self.env.lookup_variable(&ident.name),
            Expression::Binary(binary) => self.check_binary_expression(binary),
            Expression::Unary(unary) => self.check_unary_expression(unary),
            Expression::Call(call) => self.check_call_expression(call),
            Expression::MethodCall(call) => self.check_method_call_expression(call),
            Expression::Field(field) => self.check_field_expression(field),
            Expression::Index(index) => self.check_index_expression(index),
            Expression::Array(array) => self.check_array_expression(array),
            Expression::Tuple(tuple) => self.check_tuple_expression(tuple),
            Expression::Block(block) => self.check_block_expression(block),
            Expression::If(if_) => self.check_if_expression(if_),
            Expression::While(while_) => self.check_while_expression(while_),
            Expression::For(for_) => self.check_for_expression(for_),
            Expression::Match(match_) => self.check_match_expression(match_),
            Expression::Lambda(lambda) => self.check_lambda_expression(lambda),
            Expression::Tensor(tensor) => self.check_tensor_expression(tensor),
            Expression::Gradient(gradient) => self.check_gradient_expression(gradient),
        }
    }

    // TODO: Implement remaining type checking methods

    /// Check a type and return its normalized form
    fn check_type(&self, ty: &Type) -> Result<Type> {
        // TODO: Implement type checking
        Ok(ty.clone())
    }
}

impl Default for SemanticAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_function_type_checking() {
        let mut analyzer = SemanticAnalyzer::new();

        // Create a simple function AST
        let function = Function {
            name: Identifier {
                name: "add".to_string(),
                location: SourceLocation {
                    file: "test".to_string(),
                    line: 1,
                    column: 1,
                },
            },
            params: vec![
                Parameter {
                    name: Identifier {
                        name: "x".to_string(),
                        location: SourceLocation {
                            file: "test".to_string(),
                            line: 1,
                            column: 1,
                        },
                    },
                    type_: Type::Float64,
                    default_value: None,
                    location: SourceLocation {
                        file: "test".to_string(),
                        line: 1,
                        column: 1,
                    },
                },
                Parameter {
                    name: Identifier {
                        name: "y".to_string(),
                        location: SourceLocation {
                            file: "test".to_string(),
                            line: 1,
                            column: 1,
                        },
                    },
                    type_: Type::Float64,
                    default_value: None,
                    location: SourceLocation {
                        file: "test".to_string(),
                        line: 1,
                        column: 1,
                    },
                },
            ],
            return_type: Some(Type::Float64),
            body: Block {
                statements: vec![Statement::Return(Return {
                    value: Some(Expression::Binary(Box::new(Binary {
                        left: Expression::Identifier(Identifier {
                            name: "x".to_string(),
                            location: SourceLocation {
                                file: "test".to_string(),
                                line: 1,
                                column: 1,
                            },
                        }),
                        operator: BinaryOperator::Add,
                        right: Expression::Identifier(Identifier {
                            name: "y".to_string(),
                            location: SourceLocation {
                                file: "test".to_string(),
                                line: 1,
                                column: 1,
                            },
                        }),
                        location: SourceLocation {
                            file: "test".to_string(),
                            line: 1,
                            column: 1,
                        },
                    }))),
                    location: SourceLocation {
                        file: "test".to_string(),
                        line: 1,
                        column: 1,
                    },
                })],
                location: SourceLocation {
                    file: "test".to_string(),
                    line: 1,
                    column: 1,
                },
            },
            is_async: false,
            visibility: Visibility::Public,
            location: SourceLocation {
                file: "test".to_string(),
                line: 1,
                column: 1,
            },
        };

        // Check the function
        assert!(analyzer.check_function(&function).is_ok());
    }
}
