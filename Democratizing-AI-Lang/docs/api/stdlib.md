# Democratising Standard Library API Reference

This document provides detailed documentation for the Democratising standard library.

## Core Module

The core module provides fundamental types and traits used throughout the library.

### Types

#### `Array<T>`
A multi-dimensional array type that serves as the foundation for tensors.

```rust
pub struct Array<T> {
    // ...
}

impl<T> Array<T> {
    pub fn new(shape: Vec<usize>, value: T) -> Self
    pub fn zeros(shape: Vec<usize>) -> Self where T: Number
    pub fn ones(shape: Vec<usize>) -> Self where T: Number
    pub fn shape(&self) -> &[usize]
    pub fn ndim(&self) -> usize
    pub fn size(&self) -> usize
}
```

#### `Result<T>`
A specialized result type for handling errors in the library.

```rust
pub type Result<T> = std::result::Result<T, Error>;
```

### Traits

#### `Number`
A trait for numeric types that can be used in mathematical operations.

```rust
pub trait Number:
    Add<Output = Self> +
    Sub<Output = Self> +
    Mul<Output = Self> +
    Div<Output = Self> +
    Clone +
    Copy +
    Send +
    Sync +
    'static
{
    fn zero() -> Self;
    fn one() -> Self;
    fn from_f64(value: f64) -> Self;
    fn to_f64(self) -> f64;
}
```

#### `Numeric`
A trait for numeric types that support additional mathematical operations.

```rust
pub trait Numeric: Number {
    fn abs(self) -> Self;
    fn sqrt(self) -> Self;
    fn exp(self) -> Self;
    fn ln(self) -> Self;
    fn powf(self, n: Self) -> Self;
    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn tan(self) -> Self;
}
```

## Tensor Module

The tensor module provides high-level tensor operations for machine learning.

### Types

#### `Tensor<T>`
A tensor type built on top of `Array<T>` with additional operations for machine learning.

```rust
pub struct Tensor<T> {
    // ...
}

impl<T: Number> Tensor<T> {
    pub fn new(data: Array<T>) -> Self
    pub fn zeros(shape: Vec<usize>) -> Self
    pub fn ones(shape: Vec<usize>) -> Self
    pub fn shape(&self) -> &[usize]
    pub fn ndim(&self) -> usize
    pub fn size(&self) -> usize
    pub fn matmul(&self, other: &Self) -> Result<Self>
    pub fn transpose(&self, dim0: usize, dim1: usize) -> Result<Self>
}
```

## Neural Network Module

The nn module provides building blocks for creating and training neural networks.

### Types

#### `Sequential`
A sequential model composed of layers.

```rust
pub struct Sequential {
    // ...
}

impl Sequential {
    pub fn new(layers: Vec<Box<dyn Layer>>) -> Self
    pub fn forward(&self, input: &Tensor<f64>) -> Result<Tensor<f64>>
    pub fn update(&mut self, learning_rate: f64)
    pub fn zero_grad(&mut self)
}
```

#### `Dense`
A fully connected (dense) layer.

```rust
pub struct Dense {
    // ...
}

impl Dense {
    pub fn new(
        input_size: usize,
        output_size: usize,
        activation: Box<dyn Fn(&Tensor<f64>) -> Result<Tensor<f64>>>,
    ) -> Self
}
```

### Traits

#### `Layer`
A trait for neural network layers.

```rust
pub trait Layer {
    type Input;
    type Output;

    fn forward(&self, input: &Self::Input) -> Result<Self::Output>;
    fn update(&mut self, learning_rate: f64);
    fn zero_grad(&mut self);
    fn parameters(&self) -> Vec<&Tensor<f64>>;
}
```

### Functions

#### Activation Functions
```rust
pub mod activations {
    pub fn relu(x: &Tensor<f64>) -> Result<Tensor<f64>>
    pub fn sigmoid(x: &Tensor<f64>) -> Result<Tensor<f64>>
    pub fn tanh(x: &Tensor<f64>) -> Result<Tensor<f64>>
}
```

#### Loss Functions
```rust
pub mod losses {
    pub fn mse(prediction: &Tensor<f64>, target: &Tensor<f64>) -> Result<Tensor<f64>>
    pub fn cross_entropy(prediction: &Tensor<f64>, target: &Tensor<f64>) -> Result<Tensor<f64>>
}
```

## Automatic Differentiation Module

The autodiff module provides automatic differentiation capabilities.

### Types

#### `Variable`
A variable in the computation graph that supports automatic differentiation.

```rust
pub struct Variable {
    // ...
}

impl Variable {
    pub fn new(value: Tensor<f64>) -> Self
    pub fn constant(value: Tensor<f64>) -> Self
    pub fn value(&self) -> Tensor<f64>
    pub fn gradient(&self) -> Option<Tensor<f64>>
    pub fn zero_grad(&mut self)
    pub fn backward(&mut self) -> Result<()>
}
```

## GPU Support

The CUDA module provides GPU acceleration capabilities when the "gpu" feature is enabled.

### Configuration

```rust
// Enable GPU support
let config = Config {
    use_gpu: true,
    ..Default::default()
};
configure(config)?;
```

### Memory Management

GPU memory is managed automatically when using tensors with GPU support enabled. The library handles:
- Memory allocation and deallocation
- Data transfer between CPU and GPU
- Automatic fallback to CPU when GPU is unavailable

## Error Handling

The library uses a custom `Error` type for error handling:

```rust
pub enum Error {
    InvalidArgument(String),
    InvalidOperation(String),
    IndexOutOfBounds { index: usize, len: usize },
    ShapeMismatch { expected: Vec<usize>, got: Vec<usize> },
    TypeMismatch { expected: String, got: String },
    // ...
}
```

Common error handling patterns:

```rust
// Using the ? operator
fn process_tensor(t: &Tensor<f64>) -> Result<Tensor<f64>> {
    let a = t.matmul(&other_tensor)?;
    let b = a.transpose(0, 1)?;
    Ok(b)
}

// Explicit error handling
match tensor.matmul(&other) {
    Ok(result) => // Use result,
    Err(e) => // Handle error,
}
```

## Examples

See the [examples directory](../../examples/) for complete examples of using the standard library.

## Best Practices

1. Initialize the library before use:
```rust
democratising_stdlib::init()?;
```

2. Configure GPU support when available:
```rust
let config = Config {
    use_gpu: true,
    ..Default::default()
};
configure(config)?;
```

3. Use error handling consistently:
```rust
fn train_model() -> Result<()> {
    let model = create_model()?;
    let data = load_data()?;
    model.train(&data)?;
    Ok(())
}
```

4. Clean up resources properly:
```rust
{
    let tensor = Tensor::ones(vec![1000, 1000]);
    // Tensor is automatically freed when it goes out of scope
}
```

## Performance Tips

1. Use GPU acceleration when available
2. Batch operations where possible
3. Reuse tensors instead of creating new ones
4. Use the appropriate numeric types for your needs
5. Profile your code to identify bottlenecks

## Further Reading

- [Language Reference](../language-reference.md)
- [Compiler Documentation](../compiler.md)
- [Examples](../../examples/README.md)
