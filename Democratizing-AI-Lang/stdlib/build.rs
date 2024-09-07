use std::env;
use std::path::PathBuf;

fn main() {
    // Only build CUDA kernels if the 'cuda' feature is enabled
    if cfg!(feature = "cuda") {
        println!("cargo:rerun-if-changed=src/kernels/elementwise.cu");

        // Get CUDA paths
        let cuda_path = env::var("CUDA_PATH").unwrap_or_else(|_| {
            if cfg!(target_os = "windows") {
                "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.1".to_string()
            } else {
                "/usr/local/cuda".to_string()
            }
        });

        // Set up CUDA compiler
        let nvcc = if cfg!(target_os = "windows") {
            format!("{}\\bin\\nvcc.exe", cuda_path)
        } else {
            format!("{}/bin/nvcc", cuda_path)
        };

        // Set up include paths
        let include_path = format!("{}/include", cuda_path);

        // Set up output paths
        let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
        let cuda_lib = out_dir.join("libcuda_kernels.a");

        // Build flags
        let mut nvcc_flags = vec![
            "-O3",                                        // Optimization level
            "--compiler-options",                         // C++ compiler options
            "-fPIC",                                      // Position Independent Code
            "-shared",                                    // Create shared library
            "--generate-code=arch=compute_60,code=sm_60", // Minimum supported architecture
            "--generate-code=arch=compute_70,code=sm_70",
            "--generate-code=arch=compute_75,code=sm_75",
            "--generate-code=arch=compute_80,code=sm_80",
            "--generate-code=arch=compute_86,code=sm_86",
            "--generate-code=arch=compute_89,code=sm_89",
            "--generate-code=arch=compute_90,code=sm_90",
        ];

        // Add debug symbols in debug mode
        if env::var("PROFILE").unwrap() == "debug" {
            nvcc_flags.push("-g");
            nvcc_flags.push("-G");
        }

        // Windows-specific flags
        if cfg!(target_os = "windows") {
            nvcc_flags.push("-Xcompiler");
            nvcc_flags.push("/MD"); // Multi-threaded DLL runtime
        }

        // Compile CUDA kernels
        let status = std::process::Command::new(&nvcc)
            .args(&nvcc_flags)
            .arg("-I")
            .arg(&include_path)
            .arg("src/kernels/elementwise.cu")
            .arg("-o")
            .arg(&cuda_lib)
            .status()
            .expect("Failed to execute nvcc");

        if !status.success() {
            panic!("Failed to compile CUDA kernels");
        }

        // Link the CUDA library
        println!("cargo:rustc-link-search=native={}", cuda_path);
        if cfg!(target_os = "windows") {
            println!("cargo:rustc-link-search=native={}\\lib\\x64", cuda_path);
        } else {
            println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
        }
        println!("cargo:rustc-link-lib=static=cuda_kernels");
        println!("cargo:rustc-link-lib=dylib=cudart");
        println!("cargo:rustc-link-lib=dylib=cuda");

        // Generate Rust bindings for CUDA kernels
        let bindings = bindgen::Builder::default()
            .header("src/kernels/elementwise.cu")
            .clang_arg(format!("-I{}", include_path))
            .parse_callbacks(Box::new(bindgen::CargoCallbacks))
            .generate()
            .expect("Unable to generate bindings");

        bindings
            .write_to_file(out_dir.join("cuda_bindings.rs"))
            .expect("Couldn't write bindings");
    }

    // Print feature status
    if cfg!(feature = "cuda") {
        println!("cargo:warning=Building with CUDA support");
    } else {
        println!("cargo:warning=Building without CUDA support");
    }

    // Rebuild if build.rs changes
    println!("cargo:rerun-if-changed=build.rs");
}
