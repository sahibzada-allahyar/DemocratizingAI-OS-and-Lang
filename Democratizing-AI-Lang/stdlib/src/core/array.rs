use std::{
    fmt,
    ops::{Deref, DerefMut, Index, IndexMut},
};

use super::error::{Error, Result};

/// A generic n-dimensional array
#[derive(Clone)]
pub struct Array<T> {
    data: Vec<T>,
    shape: Vec<usize>,
    strides: Vec<usize>,
}

impl<T: Clone> Array<T> {
    /// Create a new array with the given shape and data
    pub fn new(data: Vec<T>, shape: Vec<usize>) -> Result<Self> {
        let expected_len = shape.iter().product();
        ensure!(
            data.len() == expected_len,
            InvalidShape,
            "Data length {} does not match shape {:?} (expected {})",
            data.len(),
            shape,
            expected_len
        );

        let strides = Self::compute_strides(&shape);
        Ok(Self { data, shape, strides })
    }

    /// Create a new array filled with the given value
    pub fn full(shape: Vec<usize>, value: T) -> Self {
        let size = shape.iter().product();
        let data = vec![value; size];
        let strides = Self::compute_strides(&shape);
        Self { data, shape, strides }
    }

    /// Create a new array filled with zeros
    pub fn zeros(shape: Vec<usize>) -> Self
    where
        T: Default,
    {
        let size = shape.iter().product();
        let data = vec![T::default(); size];
        let strides = Self::compute_strides(&shape);
        Self { data, shape, strides }
    }

    /// Create a new array filled with ones
    pub fn ones(shape: Vec<usize>) -> Self
    where
        T: From<u8>,
    {
        let size = shape.iter().product();
        let data = vec![T::from(1); size];
        let strides = Self::compute_strides(&shape);
        Self { data, shape, strides }
    }

    /// Reshape the array
    pub fn reshape(&self, shape: Vec<usize>) -> Result<Self> {
        let expected_len = shape.iter().product();
        ensure!(
            self.data.len() == expected_len,
            InvalidShape,
            "Cannot reshape array of size {} into shape {:?} (expected {})",
            self.data.len(),
            shape,
            expected_len
        );

        let strides = Self::compute_strides(&shape);
        Ok(Self { data: self.data.clone(), shape, strides })
    }

    /// Get a slice of the array
    pub fn slice(&self, start: usize, end: usize) -> Result<Self> {
        ensure!(
            start <= end && end <= self.data.len(),
            InvalidIndex,
            "Invalid slice range {}..{} for array of length {}",
            start,
            end,
            self.data.len()
        );

        let data = self.data[start..end].to_vec();
        let mut shape = self.shape.clone();
        shape[0] = end - start;
        let strides = Self::compute_strides(&shape);

        Ok(Self { data, shape, strides })
    }

    /// Get the shape of the array
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get the strides of the array
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    /// Get the number of dimensions
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Get the total number of elements
    pub fn size(&self) -> usize {
        self.data.len()
    }

    /// Check if the array is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get a reference to the underlying data
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Get a mutable reference to the underlying data
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Convert the array into a vector
    pub fn into_vec(self) -> Vec<T> {
        self.data
    }

    /// Compute strides for the given shape
    fn compute_strides(shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![1; shape.len()];
        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }

    /// Get the linear index for the given indices
    fn get_index(&self, indices: &[usize]) -> Result<usize> {
        ensure!(
            indices.len() == self.shape.len(),
            InvalidIndex,
            "Wrong number of indices: expected {}, got {}",
            self.shape.len(),
            indices.len()
        );

        for (i, &idx) in indices.iter().enumerate() {
            ensure!(
                idx < self.shape[i],
                InvalidIndex,
                "Index {} out of bounds for axis {} with size {}",
                idx,
                i,
                self.shape[i]
            );
        }

        let index = indices
            .iter()
            .zip(self.strides.iter())
            .map(|(&i, &s)| i * s)
            .sum();

        Ok(index)
    }
}

impl<T> Index<&[usize]> for Array<T> {
    type Output = T;

    fn index(&self, indices: &[usize]) -> &Self::Output {
        let idx = self.get_index(indices).unwrap();
        &self.data[idx]
    }
}

impl<T> IndexMut<&[usize]> for Array<T> {
    fn index_mut(&mut self, indices: &[usize]) -> &mut Self::Output {
        let idx = self.get_index(indices).unwrap();
        &mut self.data[idx]
    }
}

impl<T> Deref for Array<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<T> DerefMut for Array<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

impl<T: fmt::Debug> fmt::Debug for Array<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Array({:?}, shape={:?})", self.data, self.shape)
    }
}

impl<T: PartialEq> PartialEq for Array<T> {
    fn eq(&self, other: &Self) -> bool {
        self.shape == other.shape && self.data == other.data
    }
}

impl<T: Eq> Eq for Array<T> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_array_creation() {
        let data = vec![1, 2, 3, 4, 5, 6];
        let shape = vec![2, 3];
        let arr = Array::new(data, shape).unwrap();
        assert_eq!(arr.shape(), &[2, 3]);
        assert_eq!(arr.strides(), &[3, 1]);
        assert_eq!(arr.size(), 6);
    }

    #[test]
    fn test_array_indexing() {
        let data = vec![1, 2, 3, 4, 5, 6];
        let shape = vec![2, 3];
        let arr = Array::new(data, shape).unwrap();
        assert_eq!(arr[&[0, 0]], 1);
        assert_eq!(arr[&[0, 1]], 2);
        assert_eq!(arr[&[1, 0]], 4);
        assert_eq!(arr[&[1, 2]], 6);
    }

    #[test]
    fn test_array_reshape() {
        let data = vec![1, 2, 3, 4, 5, 6];
        let shape = vec![2, 3];
        let arr = Array::new(data, shape).unwrap();
        let reshaped = arr.reshape(vec![3, 2]).unwrap();
        assert_eq!(reshaped.shape(), &[3, 2]);
        assert_eq!(reshaped.strides(), &[2, 1]);
    }

    #[test]
    fn test_array_slice() {
        let data = vec![1, 2, 3, 4, 5, 6];
        let shape = vec![2, 3];
        let arr = Array::new(data, shape).unwrap();
        let slice = arr.slice(0, 3).unwrap();
        assert_eq!(slice.shape(), &[1, 3]);
        assert_eq!(slice.as_slice(), &[1, 2, 3]);
    }
}
