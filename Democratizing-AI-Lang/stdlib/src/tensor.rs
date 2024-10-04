use crate::core::{
    array::NdArray,
    error::{Error, Result},
    traits::{Broadcast, DeviceTransfer, Differentiable, ElementWise, MatrixOps, ShapeOps},
    types::{Device, GradientMode, Shape},
};
use std::sync::{Arc, RwLock};

/// A tensor with automatic differentiation support
#[derive(Debug, Clone)]
pub struct Tensor<T> {
    data: Arc<NdArray<T>>,
    grad: Arc<RwLock<Option<NdArray<T>>>>,
    requires_grad: bool,
    device: Device,
    shape: Shape,
}

impl<T> Tensor<T>
where
    T: Differentiable + Send + Sync + 'static,
{
    /// Create a new tensor from an array
    pub fn new(array: NdArray<T>) -> Self {
        let shape = array.shape().to_vec();
        Self {
            data: Arc::new(array),
            grad: Arc::new(RwLock::new(None)),
            requires_grad: false,
            device: Device::CPU,
            shape,
        }
    }

    /// Create a new tensor with zeros
    pub fn zeros(shape: &[usize]) -> Result<Self> {
        Ok(Self::new(NdArray::zeros(shape)?))
    }

    /// Create a new tensor with ones
    pub fn ones(shape: &[usize]) -> Result<Self> {
        Ok(Self::new(NdArray::ones(shape)?))
    }

    /// Create a new tensor with random values
    pub fn randn(shape: &[usize]) -> Result<Self> {
        Ok(Self::new(NdArray::randn(shape)?))
    }

    /// Get a reference to the underlying data
    pub fn data(&self) -> &NdArray<T> {
        &self.data
    }

    /// Get the gradient
    pub fn grad(&self) -> Option<Arc<NdArray<T>>> {
        self.grad.read().unwrap().as_ref().map(Arc::new)
    }

    /// Set requires gradient
    pub fn requires_grad(mut self) -> Self {
        self.requires_grad = true;
        self
    }

    /// Check if requires gradient
    pub fn requires_gradient(&self) -> bool {
        self.requires_grad
    }

    /// Zero gradient
    pub fn zero_grad(&mut self) {
        if self.requires_grad {
            *self.grad.write().unwrap() = None;
        }
    }

    /// Backward pass with gradient
    pub fn backward_with_grad(&self, grad: &NdArray<T>) -> Result<()> {
        if !self.requires_grad {
            return Ok(());
        }

        let mut current_grad = self.grad.write().unwrap();
        *current_grad = Some(match current_grad.as_ref() {
            Some(existing) => &(existing + grad)?,
            None => grad.clone(),
        });

        Ok(())
    }

    /// Backward pass
    pub fn backward(&self) -> Result<()> {
        let grad = NdArray::ones(&self.shape)?;
        self.backward_with_grad(&grad)
    }
}

impl<T> ShapeOps for Tensor<T>
where
    T: Differentiable + Send + Sync + 'static,
{
    fn shape(&self) -> &Shape {
        &self.shape
    }

    fn reshape(&self, shape: &Shape) -> Result<Self> {
        Ok(Self {
            data: Arc::new(self.data.reshape(shape)?),
            grad: Arc::new(RwLock::new(None)),
            requires_grad: self.requires_grad,
            device: self.device,
            shape: shape.to_vec(),
        })
    }

    fn is_shape_compatible(&self, other: &Shape) -> bool {
        &self.shape == other
    }
}

impl<T> DeviceTransfer for Tensor<T>
where
    T: Differentiable + Send + Sync + 'static,
{
    fn to_device(&self, device: Device) -> Result<Self> {
        if self.device == device {
            return Ok(self.clone());
        }

        Ok(Self {
            data: Arc::new(self.data.to_device(device)?),
            grad: Arc::new(RwLock::new(None)),
            requires_grad: self.requires_grad,
            device,
            shape: self.shape.clone(),
        })
    }

    fn device(&self) -> Device {
        self.device
    }
}

impl<T> MatrixOps for Tensor<T>
where
    T: Differentiable + Send + Sync + 'static,
{
    fn matmul(&self, other: &Self) -> Result<Self> {
        let result = self.data.matmul(&other.data)?;
        let shape = result.shape().to_vec();
        Ok(Self {
            data: Arc::new(result),
            grad: Arc::new(RwLock::new(None)),
            requires_grad: self.requires_grad || other.requires_grad,
            device: self.device,
            shape,
        })
    }

    fn transpose(&self) -> Result<Self> {
        let result = self.data.transpose()?;
        let mut shape = self.shape.clone();
        if shape.len() >= 2 {
            shape.swap(shape.len() - 2, shape.len() - 1);
        }
        Ok(Self {
            data: Arc::new(result),
            grad: Arc::new(RwLock::new(None)),
            requires_grad: self.requires_grad,
            device: self.device,
            shape,
        })
    }
}

impl<T> std::ops::Add for &Tensor<T>
where
    T: Differentiable + Send + Sync + 'static,
{
    type Output = Result<Tensor<T>>;

    fn add(self, other: &Tensor<T>) -> Self::Output {
        Ok(Tensor {
            data: Arc::new((&*self.data + &*other.data)?),
            grad: Arc::new(RwLock::new(None)),
            requires_grad: self.requires_grad || other.requires_grad,
            device: self.device,
            shape: self.shape.clone(),
        })
    }
}

impl<T> std::ops::Sub for &Tensor<T>
where
    T: Differentiable + Send + Sync + 'static,
{
    type Output = Result<Tensor<T>>;

    fn sub(self, other: &Tensor<T>) -> Self::Output {
        Ok(Tensor {
            data: Arc::new((&*self.data - &*other.data)?),
            grad: Arc::new(RwLock::new(None)),
            requires_grad: self.requires_grad || other.requires_grad,
            device: self.device,
            shape: self.shape.clone(),
        })
    }
}

impl<T> std::ops::Mul for &Tensor<T>
where
    T: Differentiable + Send + Sync + 'static,
{
    type Output = Result<Tensor<T>>;

    fn mul(self, other: &Tensor<T>) -> Self::Output {
        Ok(Tensor {
            data: Arc::new((&*self.data * &*other.data)?),
            grad: Arc::new(RwLock::new(None)),
            requires_grad: self.requires_grad || other.requires_grad,
            device: self.device,
            shape: self.shape.clone(),
        })
    }
}

impl<T> std::ops::Div for &Tensor<T>
where
    T: Differentiable + Send + Sync + 'static,
{
    type Output = Result<Tensor<T>>;

    fn div(self, other: &Tensor<T>) -> Self::Output {
        Ok(Tensor {
            data: Arc::new((&*self.data / &*other.data)?),
            grad: Arc::new(RwLock::new(None)),
            requires_grad: self.requires_grad || other.requires_grad,
            device: self.device,
            shape: self.shape.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::traits::TensorElement;

    impl Differentiable for f32 {
        fn grad(&self) -> Option<Self> {
            None
        }

        fn set_grad(&mut self, _grad: Option<Self>) {}

        fn zero_grad(&mut self) {}

        fn requires_grad(&self) -> bool {
            false
        }

        fn set_requires_grad(&mut self, _requires_grad: bool) {}
    }

    #[test]
    fn test_tensor_creation() {
        let t = Tensor::<f32>::zeros(&[2, 3]).unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert!(!t.requires_gradient());

        let t = Tensor::<f32>::ones(&[2, 3]).unwrap().requires_grad();
        assert_eq!(t.shape(), &[2, 3]);
        assert!(t.requires_gradient());
    }

    #[test]
    fn test_tensor_operations() {
        let a = Tensor::<f32>::ones(&[2, 2]).unwrap();
        let b = Tensor::<f32>::ones(&[2, 2]).unwrap();

        let c = (&a + &b).unwrap();
        assert_eq!(c.data().data().as_slice().unwrap(), &[2.0, 2.0, 2.0, 2.0]);

        let d = (&a * &b).unwrap();
        assert_eq!(d.data().data().as_slice().unwrap(), &[1.0, 1.0, 1.0, 1.0]);

        let e = a.matmul(&b).unwrap();
        assert_eq!(e.data().data().as_slice().unwrap(), &[2.0, 2.0, 2.0, 2.0]);
    }

    #[test]
    fn test_tensor_device() {
        let t = Tensor::<f32>::zeros(&[2, 3]).unwrap();
        assert_eq!(t.device(), Device::CPU);

        #[cfg(feature = "gpu")]
        {
            let gpu_t = t.to_device(Device::GPU(0)).unwrap();
            assert_eq!(gpu_t.device(), Device::GPU(0));
        }
    }

    #[test]
    fn test_tensor_backward() {
        let mut t = Tensor::<f32>::ones(&[2, 2]).unwrap().requires_grad();
        t.backward().unwrap();
        assert!(t.grad().is_some());

        t.zero_grad();
        assert!(t.grad().is_none());
    }
}
