#include <cuda_runtime.h>

extern "C" {

// Element-wise addition kernel
__global__ void add_kernel(const float* a, const float* b, float* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Element-wise multiplication kernel
__global__ void mul_kernel(const float* a, const float* b, float* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] * b[idx];
    }
}

// Element-wise division kernel
__global__ void div_kernel(const float* a, const float* b, float* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] / b[idx];
    }
}

// Element-wise subtraction kernel
__global__ void sub_kernel(const float* a, const float* b, float* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] - b[idx];
    }
}

// Element-wise exponential kernel
__global__ void exp_kernel(const float* a, float* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = expf(a[idx]);
    }
}

// Element-wise natural logarithm kernel
__global__ void log_kernel(const float* a, float* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = logf(a[idx]);
    }
}

// Element-wise square root kernel
__global__ void sqrt_kernel(const float* a, float* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = sqrtf(a[idx]);
    }
}

// Element-wise power kernel
__global__ void pow_kernel(const float* a, float b, float* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = powf(a[idx], b);
    }
}

// Element-wise tanh kernel
__global__ void tanh_kernel(const float* a, float* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = tanhf(a[idx]);
    }
}

// Element-wise sigmoid kernel
__global__ void sigmoid_kernel(const float* a, float* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = 1.0f / (1.0f + expf(-a[idx]));
    }
}

// Element-wise ReLU kernel
__global__ void relu_kernel(const float* a, float* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = fmaxf(0.0f, a[idx]);
    }
}

// Element-wise leaky ReLU kernel
__global__ void leaky_relu_kernel(const float* a, float alpha, float* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = a[idx];
        c[idx] = x > 0.0f ? x : alpha * x;
    }
}

// Element-wise absolute value kernel
__global__ void abs_kernel(const float* a, float* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = fabsf(a[idx]);
    }
}

// Element-wise negative kernel
__global__ void neg_kernel(const float* a, float* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = -a[idx];
    }
}

// Element-wise reciprocal kernel
__global__ void reciprocal_kernel(const float* a, float* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = 1.0f / a[idx];
    }
}

// Element-wise sine kernel
__global__ void sin_kernel(const float* a, float* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = sinf(a[idx]);
    }
}

// Element-wise cosine kernel
__global__ void cos_kernel(const float* a, float* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = cosf(a[idx]);
    }
}

// Element-wise maximum kernel
__global__ void maximum_kernel(const float* a, const float* b, float* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = fmaxf(a[idx], b[idx]);
    }
}

// Element-wise minimum kernel
__global__ void minimum_kernel(const float* a, const float* b, float* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = fminf(a[idx], b[idx]);
    }
}

// Element-wise clip kernel
__global__ void clip_kernel(const float* a, float min_val, float max_val, float* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = fminf(fmaxf(a[idx], min_val), max_val);
    }
}

// Element-wise where kernel (ternary operation)
__global__ void where_kernel(const bool* cond, const float* a, const float* b, float* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = cond[idx] ? a[idx] : b[idx];
    }
}

// Element-wise equal kernel
__global__ void equal_kernel(const float* a, const float* b, bool* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] == b[idx];
    }
}

// Element-wise not equal kernel
__global__ void not_equal_kernel(const float* a, const float* b, bool* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] != b[idx];
    }
}

// Element-wise greater kernel
__global__ void greater_kernel(const float* a, const float* b, bool* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] > b[idx];
    }
}

// Element-wise greater equal kernel
__global__ void greater_equal_kernel(const float* a, const float* b, bool* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] >= b[idx];
    }
}

// Element-wise less kernel
__global__ void less_kernel(const float* a, const float* b, bool* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] < b[idx];
    }
}

// Element-wise less equal kernel
__global__ void less_equal_kernel(const float* a, const float* b, bool* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] <= b[idx];
    }
}

} // extern "C"
