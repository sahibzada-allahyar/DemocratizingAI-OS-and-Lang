use crate::{error::Result, tensor::Tensor};
use cuda_runtime_sys::*;
use std::{ffi::c_void, ptr};

// Include generated CUDA kernel bindings
include!(concat!(env!("OUT_DIR"), "/cuda_bindings.rs"));

/// CUDA device information
pub struct CudaDevice {
    id: i32,
    properties: cudaDeviceProp,
}

impl CudaDevice {
    /// Create a new CUDA device handle
    pub fn new(device_id: usize) -> Result<Self> {
        let mut properties = unsafe { std::mem::zeroed() };
        unsafe {
            check_cuda_error(cudaGetDeviceProperties(&mut properties, device_id as i32))?;
        }
        Ok(Self { id: device_id as i32, properties })
    }

    /// Get the number of available CUDA devices
    pub fn count() -> Result<usize> {
        let mut count = 0;
        unsafe {
            check_cuda_error(cudaGetDeviceCount(&mut count))?;
        }
        Ok(count as usize)
    }

    /// Get the device name
    pub fn name(&self) -> Result<String> {
        Ok(unsafe {
            std::ffi::CStr::from_ptr(self.properties.name.as_ptr())
                .to_string_lossy()
                .into_owned()
        })
    }

    /// Get the compute capability major version
    pub fn major(&self) -> Result<i32> {
        Ok(self.properties.major)
    }

    /// Get the compute capability minor version
    pub fn minor(&self) -> Result<i32> {
        Ok(self.properties.minor)
    }

    /// Get the total device memory in bytes
    pub fn total_memory(&self) -> Result<usize> {
        Ok(self.properties.totalGlobalMem as usize)
    }

    /// Get the available device memory in bytes
    pub fn free_memory(&self) -> Result<usize> {
        let mut free = 0;
        let mut total = 0;
        unsafe {
            check_cuda_error(cudaMemGetInfo(&mut free, &mut total))?;
        }
        Ok(free as usize)
    }

    /// Set this as the current CUDA device
    pub fn set_current(&self) -> Result<()> {
        unsafe {
            check_cuda_error(cudaSetDevice(self.id))?;
        }
        Ok(())
    }
}

/// CUDA memory allocation
pub struct CudaMemory {
    ptr: *mut c_void,
    size: usize,
}

impl CudaMemory {
    /// Allocate CUDA memory
    pub fn new(size: usize) -> Result<Self> {
        let mut ptr = ptr::null_mut();
        unsafe {
            check_cuda_error(cudaMalloc(&mut ptr, size))?;
        }
        Ok(Self { ptr, size })
    }

    /// Copy data from host to device
    pub fn copy_from_host(&mut self, data: &[u8]) -> Result<()> {
        assert!(data.len() <= self.size);
        unsafe {
            check_cuda_error(cudaMemcpy(
                self.ptr,
                data.as_ptr() as *const c_void,
                data.len(),
                cudaMemcpyKind::cudaMemcpyHostToDevice,
            ))?;
        }
        Ok(())
    }

    /// Copy data from device to host
    pub fn copy_to_host(&self, data: &mut [u8]) -> Result<()> {
        assert!(data.len() <= self.size);
        unsafe {
            check_cuda_error(cudaMemcpy(
                data.as_mut_ptr() as *mut c_void,
                self.ptr,
                data.len(),
                cudaMemcpyKind::cudaMemcpyDeviceToHost,
            ))?;
        }
        Ok(())
    }

    /// Get the raw pointer
    pub fn as_ptr(&self) -> *mut c_void {
        self.ptr
    }

    /// Get the allocation size
    pub fn size(&self) -> usize {
        self.size
    }
}

impl Drop for CudaMemory {
    fn drop(&mut self) {
        unsafe {
            cudaFree(self.ptr);
        }
    }
}

/// CUDA stream for asynchronous operations
pub struct CudaStream {
    stream: cudaStream_t,
}

impl CudaStream {
    /// Create a new CUDA stream
    pub fn new() -> Result<Self> {
        let mut stream = ptr::null_mut();
        unsafe {
            check_cuda_error(cudaStreamCreate(&mut stream))?;
        }
        Ok(Self { stream })
    }

    /// Synchronize the stream
    pub fn synchronize(&self) -> Result<()> {
        unsafe {
            check_cuda_error(cudaStreamSynchronize(self.stream))?;
        }
        Ok(())
    }

    /// Get the raw stream handle
    pub fn as_ptr(&self) -> cudaStream_t {
        self.stream
    }
}

impl Drop for CudaStream {
    fn drop(&mut self) {
        unsafe {
            cudaStreamDestroy(self.stream);
        }
    }
}

/// CUDA event for timing and synchronization
pub struct CudaEvent {
    event: cudaEvent_t,
}

impl CudaEvent {
    /// Create a new CUDA event
    pub fn new() -> Result<Self> {
        let mut event = ptr::null_mut();
        unsafe {
            check_cuda_error(cudaEventCreate(&mut event))?;
        }
        Ok(Self { event })
    }

    /// Record the event in a stream
    pub fn record(&self, stream: &CudaStream) -> Result<()> {
        unsafe {
            check_cuda_error(cudaEventRecord(self.event, stream.as_ptr()))?;
        }
        Ok(())
    }

    /// Synchronize on the event
    pub fn synchronize(&self) -> Result<()> {
        unsafe {
            check_cuda_error(cudaEventSynchronize(self.event))?;
        }
        Ok(())
    }

    /// Get elapsed time between two events in milliseconds
    pub fn elapsed_time(&self, end: &CudaEvent) -> Result<f32> {
        let mut ms = 0.0;
        unsafe {
            check_cuda_error(cudaEventElapsedTime(&mut ms, self.event, end.event))?;
        }
        Ok(ms)
    }
}

impl Drop for CudaEvent {
    fn drop(&mut self) {
        unsafe {
            cudaEventDestroy(self.event);
        }
    }
}

/// Launch configuration for CUDA kernels
#[derive(Debug, Clone, Copy)]
pub struct LaunchConfig {
    pub block_size: dim3,
    pub grid_size: dim3,
    pub shared_memory_bytes: usize,
    pub stream: Option<cudaStream_t>,
}

impl Default for LaunchConfig {
    fn default() -> Self {
        Self {
            block_size: dim3 { x: 256, y: 1, z: 1 },
            grid_size: dim3 { x: 1, y: 1, z: 1 },
            shared_memory_bytes: 0,
            stream: None,
        }
    }
}

impl LaunchConfig {
    /// Create a new launch configuration for the given number of elements
    pub fn for_num_elements(num_elements: usize) -> Self {
        let block_size = 256;
        let grid_size = (num_elements + block_size - 1) / block_size;
        Self {
            block_size: dim3 { x: block_size as u32, y: 1, z: 1 },
            grid_size: dim3 { x: grid_size as u32, y: 1, z: 1 },
            shared_memory_bytes: 0,
            stream: None,
        }
    }
}

/// Check CUDA error code
fn check_cuda_error(error: cudaError_t) -> Result<()> {
    if error != cudaError::cudaSuccess {
        let error_name = unsafe { cudaGetErrorName(error) };
        let error_string = unsafe { cudaGetErrorString(error) };
        let error_name = unsafe { std::ffi::CStr::from_ptr(error_name) }
            .to_string_lossy()
            .into_owned();
        let error_string = unsafe { std::ffi::CStr::from_ptr(error_string) }
            .to_string_lossy()
            .into_owned();
        Err(anyhow::anyhow!(
            "CUDA error: {} - {}",
            error_name,
            error_string
        ))
    } else {
        Ok(())
    }
}

/// Initialize CUDA
pub fn init() -> Result<()> {
    unsafe {
        check_cuda_error(cudaSetDevice(0))?;
    }
    Ok(())
}

/// Get the current CUDA device
pub fn current_device() -> Result<CudaDevice> {
    let mut device = 0;
    unsafe {
        check_cuda_error(cudaGetDevice(&mut device))?;
    }
    CudaDevice::new(device as usize)
}

/// Synchronize all CUDA operations
pub fn synchronize() -> Result<()> {
    unsafe {
        check_cuda_error(cudaDeviceSynchronize())?;
    }
    Ok(())
}

/// Reset the CUDA device
pub fn reset() -> Result<()> {
    unsafe {
        check_cuda_error(cudaDeviceReset())?;
    }
    Ok(())
}

/// Get the last CUDA error
pub fn get_last_error() -> Result<()> {
    unsafe {
        check_cuda_error(cudaGetLastError())?;
    }
    Ok(())
}

/// Clear the last CUDA error
pub fn clear_last_error() {
    unsafe {
        cudaGetLastError();
    }
}
