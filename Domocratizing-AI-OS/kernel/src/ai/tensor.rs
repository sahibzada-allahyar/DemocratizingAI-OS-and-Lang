//! Tensor operations

use alloc::vec::Vec;
use core::ops::{Add, Mul, Sub};
use core::slice;
use spin::Mutex;

use crate::memory::{PhysAddr, VirtAddr};

/// Tensor data type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataType {
    /// 32-bit float
    Float32,
    /// 64-bit float
    Float64,
    /// 8-bit integer
    Int8,
    /// 16-bit integer
    Int16,
    /// 32-bit integer
    Int32,
    /// 64-bit integer
    Int64,
}

impl DataType {
    /// Get size in bytes
    pub fn size(&self) -> usize {
        match self {
            DataType::Float32 => 4,
            DataType::Float64 => 8,
            DataType::Int8 => 1,
            DataType::Int16 => 2,
            DataType::Int32 => 4,
            DataType::Int64 => 8,
        }
    }
}

/// Tensor shape
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shape(Vec<usize>);

impl Shape {
    /// Create new shape
    pub fn new(dims: &[usize]) -> Self {
        Shape(dims.to_vec())
    }

    /// Get dimensions
    pub fn dims(&self) -> &[usize] {
        &self.0
    }

    /// Get number of dimensions
    pub fn ndim(&self) -> usize {
        self.0.len()
    }

    /// Get total size
    pub fn size(&self) -> usize {
        self.0.iter().product()
    }
}

/// Tensor
pub struct Tensor {
    /// Data type
    dtype: DataType,
    /// Shape
    shape: Shape,
    /// Data pointer
    data: Mutex<VirtAddr>,
    /// Physical address
    phys: PhysAddr,
}

impl Tensor {
    /// Create new tensor
    pub fn new(dtype: DataType, shape: Shape, data: VirtAddr, phys: PhysAddr) -> Self {
        Tensor {
            dtype,
            shape,
            data: Mutex::new(data),
            phys,
        }
    }

    /// Get data type
    pub fn dtype(&self) -> DataType {
        self.dtype
    }

    /// Get shape
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Get data pointer
    pub fn data(&self) -> VirtAddr {
        *self.data.lock()
    }

    /// Get physical address
    pub fn phys(&self) -> PhysAddr {
        self.phys
    }

    /// Get total size in bytes
    pub fn size(&self) -> usize {
        self.shape.size() * self.dtype.size()
    }

    /// Get slice
    pub unsafe fn as_slice<T>(&self) -> &[T] {
        slice::from_raw_parts(self.data().as_ptr(), self.shape.size())
    }

    /// Get mutable slice
    pub unsafe fn as_slice_mut<T>(&self) -> &mut [T] {
        slice::from_raw_parts_mut(self.data().as_mut_ptr(), self.shape.size())
    }

    /// Copy from slice
    pub unsafe fn copy_from_slice<T>(&self, src: &[T]) {
        let dst = self.as_slice_mut();
        dst.copy_from_slice(src);
    }

    /// Copy to slice
    pub unsafe fn copy_to_slice<T>(&self, dst: &mut [T]) {
        let src = self.as_slice();
        dst.copy_from_slice(src);
    }
}

/// Tensor operations
impl Tensor {
    /// Add tensors
    pub fn add(&self, other: &Tensor) -> Result<Tensor, &'static str> {
        if self.dtype != other.dtype {
            return Err("mismatched data types");
        }
        if self.shape != other.shape {
            return Err("mismatched shapes");
        }

        match self.dtype {
            DataType::Float32 => unsafe {
                let a = self.as_slice::<f32>();
                let b = other.as_slice::<f32>();
                let mut c = Vec::with_capacity(self.shape.size());
                for i in 0..self.shape.size() {
                    c.push(a[i] + b[i]);
                }
                Ok(Tensor::new(
                    self.dtype,
                    self.shape.clone(),
                    VirtAddr::new(c.as_ptr() as usize),
                    PhysAddr::new(0),
                ))
            },
            DataType::Float64 => unsafe {
                let a = self.as_slice::<f64>();
                let b = other.as_slice::<f64>();
                let mut c = Vec::with_capacity(self.shape.size());
                for i in 0..self.shape.size() {
                    c.push(a[i] + b[i]);
                }
                Ok(Tensor::new(
                    self.dtype,
                    self.shape.clone(),
                    VirtAddr::new(c.as_ptr() as usize),
                    PhysAddr::new(0),
                ))
            },
            _ => Err("unsupported data type"),
        }
    }

    /// Subtract tensors
    pub fn sub(&self, other: &Tensor) -> Result<Tensor, &'static str> {
        if self.dtype != other.dtype {
            return Err("mismatched data types");
        }
        if self.shape != other.shape {
            return Err("mismatched shapes");
        }

        match self.dtype {
            DataType::Float32 => unsafe {
                let a = self.as_slice::<f32>();
                let b = other.as_slice::<f32>();
                let mut c = Vec::with_capacity(self.shape.size());
                for i in 0..self.shape.size() {
                    c.push(a[i] - b[i]);
                }
                Ok(Tensor::new(
                    self.dtype,
                    self.shape.clone(),
                    VirtAddr::new(c.as_ptr() as usize),
                    PhysAddr::new(0),
                ))
            },
            DataType::Float64 => unsafe {
                let a = self.as_slice::<f64>();
                let b = other.as_slice::<f64>();
                let mut c = Vec::with_capacity(self.shape.size());
                for i in 0..self.shape.size() {
                    c.push(a[i] - b[i]);
                }
                Ok(Tensor::new(
                    self.dtype,
                    self.shape.clone(),
                    VirtAddr::new(c.as_ptr() as usize),
                    PhysAddr::new(0),
                ))
            },
            _ => Err("unsupported data type"),
        }
    }

    /// Multiply tensors
    pub fn mul(&self, other: &Tensor) -> Result<Tensor, &'static str> {
        if self.dtype != other.dtype {
            return Err("mismatched data types");
        }
        if self.shape != other.shape {
            return Err("mismatched shapes");
        }

        match self.dtype {
            DataType::Float32 => unsafe {
                let a = self.as_slice::<f32>();
                let b = other.as_slice::<f32>();
                let mut c = Vec::with_capacity(self.shape.size());
                for i in 0..self.shape.size() {
                    c.push(a[i] * b[i]);
                }
                Ok(Tensor::new(
                    self.dtype,
                    self.shape.clone(),
                    VirtAddr::new(c.as_ptr() as usize),
                    PhysAddr::new(0),
                ))
            },
            DataType::Float64 => unsafe {
                let a = self.as_slice::<f64>();
                let b = other.as_slice::<f64>();
                let mut c = Vec::with_capacity(self.shape.size());
                for i in 0..self.shape.size() {
                    c.push(a[i] * b[i]);
                }
                Ok(Tensor::new(
                    self.dtype,
                    self.shape.clone(),
                    VirtAddr::new(c.as_ptr() as usize),
                    PhysAddr::new(0),
                ))
            },
            _ => Err("unsupported data type"),
        }
    }
}
