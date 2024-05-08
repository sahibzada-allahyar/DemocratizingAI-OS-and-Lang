//! Code generation for the Democratising compiler
//!
//! This module handles converting the IR into LLVM IR for final compilation.
//! It uses the inkwell crate for LLVM bindings.

use crate::error::Result;
use crate::ir::{self, BasicBlock, BinaryOp, Function, Instruction, Terminator, Type, UnaryOp, ValueId};
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::module::{Linkage, Module};
use inkwell::types::{BasicType, BasicTypeEnum};
use inkwell::values::{BasicValue, BasicValueEnum, FunctionValue, PointerValue};
use inkwell::OptimizationLevel;
use std::collections::HashMap;

/// The code generator
pub struct CodeGenerator<'ctx> {
    /// LLVM context
    context: &'ctx Context,
    /// LLVM module
    module: Module<'ctx>,
    /// LLVM IR builder
    builder: Builder<'ctx>,
    /// Map from IR values to LLVM values
    value_map: HashMap<ValueId, BasicValueEnum<'ctx>>,
    /// Map from IR blocks to LLVM basic blocks
    block_map: HashMap<ir::BlockId, inkwell::basic_block::BasicBlock<'ctx>>,
    /// Current function being generated
    current_function: Option<FunctionValue<'ctx>>,
}

impl<'ctx> CodeGenerator<'ctx> {
    /// Create a new code generator
    pub fn new(context: &'ctx Context, module_name: &str) -> Self {
        Self {
            context,
            module: context.create_module(module_name),
            builder: context.create_builder(),
            value_map: HashMap::new(),
            block_map: HashMap::new(),
            current_function: None,
        }
    }

    /// Convert an IR type to an LLVM type
    fn convert_type(&self, ty: &Type) -> BasicTypeEnum<'ctx> {
        match ty {
            Type::Void => self.context.void_type().as_basic_type_enum(),
            Type::Bool => self.context.bool_type().as_basic_type_enum(),
            Type::Int8 => self.context.i8_type().as_basic_type_enum(),
            Type::Int16 => self.context.i16_type().as_basic_type_enum(),
            Type::Int32 => self.context.i32_type().as_basic_type_enum(),
            Type::Int64 => self.context.i64_type().as_basic_type_enum(),
            Type::Float32 => self.context.f32_type().as_basic_type_enum(),
            Type::Float64 => self.context.f64_type().as_basic_type_enum(),
            Type::Array(elem_ty, size) => {
                let elem_ty = self.convert_type(elem_ty);
                elem_ty.array_type(*size as u32).as_basic_type_enum()
            }
            Type::Tensor(elem_ty, shape) => {
                // For tensors, we use a struct containing:
                // - Data pointer
                // - Shape array
                // - Strides array
                let elem_ty = self.convert_type(elem_ty);
                let shape_len = shape.len();
                let struct_ty = self.context.struct_type(
                    &[
                        elem_ty.ptr_type(inkwell::AddressSpace::Generic).as_basic_type_enum(),
                        self.context.i64_type().array_type(shape_len as u32).as_basic_type_enum(),
                        self.context.i64_type().array_type(shape_len as u32).as_basic_type_enum(),
                    ],
                    false,
                );
                struct_ty.as_basic_type_enum()
            }
            Type::Function { params, return_type } => {
                let param_types: Vec<_> = params.iter().map(|t| self.convert_type(t)).collect();
                let return_type = self.convert_type(return_type);
                return_type
                    .fn_type(&param_types, false)
                    .ptr_type(inkwell::AddressSpace::Generic)
                    .as_basic_type_enum()
            }
            Type::Struct(fields) => {
                let field_types: Vec<_> = fields.iter().map(|t| self.convert_type(t)).collect();
                self.context
                    .struct_type(&field_types, false)
                    .as_basic_type_enum()
            }
            Type::Pointer(pointee) => self
                .convert_type(pointee)
                .ptr_type(inkwell::AddressSpace::Generic)
                .as_basic_type_enum(),
        }
    }

    /// Generate code for a function
    pub fn gen_function(&mut self, function: &Function) -> Result<FunctionValue<'ctx>> {
        // Create function type
        let return_type = self.convert_type(&function.return_type);
        let param_types: Vec<_> = function
            .params
            .iter()
            .map(|p| self.convert_type(&p.ty))
            .collect();
        let function_type = return_type.fn_type(&param_types, false);

        // Create function
        let function_value = self.module.add_function(
            &function.name,
            function_type,
            Some(Linkage::External),
        );

        // Store current function
        self.current_function = Some(function_value);

        // Create basic blocks
        for (id, _) in &function.blocks {
            let block = self.context.append_basic_block(function_value, "block");
            self.block_map.insert(*id, block);
        }

        // Generate code for each block
        for (id, block) in &function.blocks {
            let llvm_block = self.block_map[id];
            self.builder.position_at_end(llvm_block);
            self.gen_block(block)?;
        }

        Ok(function_value)
    }

    /// Generate code for a basic block
    fn gen_block(&mut self, block: &BasicBlock) -> Result<()> {
        // Generate instructions
        for instruction in &block.instructions {
            self.gen_instruction(instruction)?;
        }

        // Generate terminator
        self.gen_terminator(&block.terminator)?;

        Ok(())
    }

    /// Generate code for an instruction
    fn gen_instruction(&mut self, instruction: &Instruction) -> Result<()> {
        match instruction {
            Instruction::Binary { op, result, left, right } => {
                let lhs = self.value_map[left].into_float_value();
                let rhs = self.value_map[right].into_float_value();
                let value = match op {
                    BinaryOp::Add => self.builder.build_float_add(lhs, rhs, "add"),
                    BinaryOp::Sub => self.builder.build_float_sub(lhs, rhs, "sub"),
                    BinaryOp::Mul => self.builder.build_float_mul(lhs, rhs, "mul"),
                    BinaryOp::Div => self.builder.build_float_div(lhs, rhs, "div"),
                    _ => unimplemented!("Binary op {:?}", op),
                };
                self.value_map.insert(*result, value.as_basic_value_enum());
            }
            Instruction::Unary { op, result, operand } => {
                let value = self.value_map[operand].into_float_value();
                let result_value = match op {
                    UnaryOp::Neg => self.builder.build_float_neg(value, "neg"),
                    _ => unimplemented!("Unary op {:?}", op),
                };
                self.value_map.insert(*result, result_value.as_basic_value_enum());
            }
            Instruction::Call { result, function, arguments } => {
                let function = self.value_map[function]
                    .into_pointer_value()
                    .as_basic_value_enum();
                let args: Vec<_> = arguments
                    .iter()
                    .map(|arg| self.value_map[arg])
                    .collect();
                let call = self.builder.build_call(
                    function.into_function_value(),
                    &args,
                    "call",
                );
                if let Some(value) = call.try_as_basic_value().left() {
                    self.value_map.insert(*result, value);
                }
            }
            Instruction::Load { result, pointer } => {
                let ptr = self.value_map[pointer].into_pointer_value();
                let value = self.builder.build_load(ptr, "load");
                self.value_map.insert(*result, value);
            }
            Instruction::Store { value, pointer } => {
                let value = self.value_map[value];
                let ptr = self.value_map[pointer].into_pointer_value();
                self.builder.build_store(ptr, value);
            }
            Instruction::GetElementPtr { result, base, indices } => {
                let ptr = self.value_map[base].into_pointer_value();
                let indices: Vec<_> = indices
                    .iter()
                    .map(|i| self.value_map[i].into_int_value())
                    .collect();
                let gep = unsafe {
                    self.builder.build_gep(ptr, &indices, "gep")
                };
                self.value_map.insert(*result, gep.as_basic_value_enum());
            }
            Instruction::Alloca { result, ty } => {
                let ty = self.convert_type(ty);
                let alloca = self.builder.build_alloca(ty, "alloca");
                self.value_map.insert(*result, alloca.as_basic_value_enum());
            }
            Instruction::Cast { result, value, target_type } => {
                let value = self.value_map[value];
                let target = self.convert_type(target_type);
                let cast = match (value.get_type(), target) {
                    // Integer to float
                    (BasicTypeEnum::IntType(_), BasicTypeEnum::FloatType(_)) => {
                        self.builder
                            .build_signed_int_to_float(
                                value.into_int_value(),
                                target.into_float_type(),
                                "cast",
                            )
                            .as_basic_value_enum()
                    }
                    // Float to integer
                    (BasicTypeEnum::FloatType(_), BasicTypeEnum::IntType(_)) => {
                        self.builder
                            .build_float_to_signed_int(
                                value.into_float_value(),
                                target.into_int_type(),
                                "cast",
                            )
                            .as_basic_value_enum()
                    }
                    // Pointer cast
                    (BasicTypeEnum::PointerType(_), BasicTypeEnum::PointerType(_)) => {
                        self.builder
                            .build_pointer_cast(
                                value.into_pointer_value(),
                                target.into_pointer_type(),
                                "cast",
                            )
                            .as_basic_value_enum()
                    }
                    _ => unimplemented!("Cast between {:?} and {:?}", value.get_type(), target),
                };
                self.value_map.insert(*result, cast);
            }
            Instruction::TensorOp { .. } => {
                // TODO: Implement tensor operations
                unimplemented!("Tensor operations not yet implemented");
            }
        }
        Ok(())
    }

    /// Generate code for a terminator instruction
    fn gen_terminator(&mut self, terminator: &Terminator) -> Result<()> {
        match terminator {
            Terminator::Return(value) => {
                if let Some(value) = value {
                    let value = self.value_map[value];
                    self.builder.build_return(Some(&value));
                } else {
                    self.builder.build_return(None);
                }
            }
            Terminator::Branch(target) => {
                let block = self.block_map[target];
                self.builder.build_unconditional_branch(block);
            }
            Terminator::CondBranch {
                condition,
                true_block,
                false_block,
            } => {
                let cond = self.value_map[condition].into_int_value();
                let true_bb = self.block_map[true_block];
                let false_bb = self.block_map[false_block];
                self.builder.build_conditional_branch(cond, true_bb, false_bb);
            }
            Terminator::Switch {
                value,
                default,
                cases,
            } => {
                let value = self.value_map[value].into_int_value();
                let default_bb = self.block_map[default];
                let switch = self.builder.build_switch(value, default_bb, cases.len() as u32);
                for (value, block) in cases {
                    let case_value = self.context.i64_type().const_int(*value as u64, false);
                    let case_bb = self.block_map[block];
                    switch.add_case(case_value, case_bb);
                }
            }
        }
        Ok(())
    }

    /// Verify and optimize the module
    pub fn finalize(&self) -> Result<()> {
        // Verify module
        if let Err(err) = self.module.verify() {
            return Err(crate::error::CompilerError::internal_error(format!(
                "LLVM verification error: {}",
                err.to_string()
            )));
        }

        // Optimize module
        let pass_manager = inkwell::passes::PassManager::create(());
        pass_manager.add_instruction_combining_pass();
        pass_manager.add_reassociate_pass();
        pass_manager.add_gvn_pass();
        pass_manager.add_cfg_simplification_pass();
        pass_manager.add_basic_alias_analysis_pass();
        pass_manager.add_promote_memory_to_register_pass();
        pass_manager.add_instruction_combining_pass();
        pass_manager.add_reassociate_pass();

        pass_manager.run_on(&self.module);

        Ok(())
    }

    /// Get the generated LLVM module
    pub fn get_module(&self) -> &Module<'ctx> {
        &self.module
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{IrBuilder, Parameter};

    #[test]
    fn test_simple_function() {
        let context = Context::create();
        let mut codegen = CodeGenerator::new(&context, "test");

        // Create IR function that adds two numbers
        let mut builder = IrBuilder::new();
        builder.start_function(
            "add".to_string(),
            vec![
                Parameter {
                    name: "x".to_string(),
                    ty: Type::Float32,
                },
                Parameter {
                    name: "y".to_string(),
                    ty: Type::Float32,
                },
            ],
            Type::Float32,
        );

        let x = builder.new_value();
        let y = builder.new_value();
        let result = builder.new_value();

        builder
            .add_instruction(Instruction::Binary {
                op: BinaryOp::Add,
                result,
                left: x,
                right: y,
            })
            .unwrap();

        builder
            .set_terminator(Terminator::Return(Some(result)))
            .unwrap();

        let function = builder.finish_function().unwrap();

        // Generate LLVM IR
        codegen.gen_function(&function).unwrap();
        codegen.finalize().unwrap();

        // Verify the module is valid
        assert!(codegen.module.verify().is_ok());
    }
}
