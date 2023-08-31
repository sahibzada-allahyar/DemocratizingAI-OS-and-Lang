//! Kernel build script

use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    // Get environment variables
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let target = env::var("TARGET").unwrap();

    // Only proceed for aarch64 targets
    if !target.contains("aarch64") {
        panic!("Unsupported target architecture: {}", target);
    }

    // Set linker script path
    let linker_script = fs::canonicalize("src/arch/aarch64/linker.ld").unwrap();
    println!("cargo:rerun-if-changed={}", linker_script.display());
    println!("cargo:rustc-link-arg=-T{}", linker_script.display());

    // Set target CPU
    if target.contains("apple") {
        println!("cargo:rustc-cfg=target_cpu=\"apple-m1\"");
        println!("cargo:rustc-link-arg=-march=armv8.5-a");
    } else {
        println!("cargo:rustc-cfg=target_cpu=\"generic\"");
        println!("cargo:rustc-link-arg=-march=armv8-a");
    }

    // Link against compiler-rt for compiler builtins
    println!("cargo:rustc-link-lib=static=compiler-rt");

    // Check for required tools
    let required_tools = [
        "aarch64-none-elf-as",
        "aarch64-none-elf-ld",
        "aarch64-none-elf-objcopy",
        "aarch64-none-elf-objdump",
        "aarch64-none-elf-readelf",
        "aarch64-none-elf-size",
        "aarch64-none-elf-strip",
    ];

    for tool in &required_tools {
        if Command::new(tool).arg("--version").output().is_err() {
            panic!("Required tool not found: {}", tool);
        }
    }

    // Generate version information
    let version = env::var("CARGO_PKG_VERSION").unwrap();
    let git_hash = Command::new("git")
        .args(&["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .unwrap_or_else(|| "unknown".to_string());
    let build_date = chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string();

    let version_info = format!(
        r#"
        pub const VERSION: &str = "{}";
        pub const GIT_HASH: &str = "{}";
        pub const BUILD_DATE: &str = "{}";
        pub const TARGET: &str = "{}";
        "#,
        version.trim(),
        git_hash.trim(),
        build_date,
        target
    );

    fs::write(out_dir.join("version.rs"), version_info).unwrap();

    // Generate memory layout information
    let memory_layout = format!(
        r#"
        pub const KERNEL_BASE: usize = 0xFFFF_0000_0000_0000;
        pub const KERNEL_SIZE: usize = 0x1000_0000;  // 16 MB
        pub const KERNEL_STACK_SIZE: usize = 0x4000;  // 16 KB
        pub const KERNEL_HEAP_SIZE: usize = 0x1000_0000;  // 16 MB
        pub const USER_BASE: usize = 0x0000_0000_0000_0000;
        pub const USER_SIZE: usize = 0x0000_8000_0000_0000;  // 128 TB
        pub const DEVICE_BASE: usize = 0xFFFF_FF00_0000_0000;
        pub const DEVICE_SIZE: usize = 0x0000_0100_0000_0000;  // 1 TB
        "#
    );

    fs::write(out_dir.join("memory_layout.rs"), memory_layout).unwrap();

    // Generate CPU feature detection
    let cpu_features = format!(
        r#"
        pub const HAS_FP: bool = true;
        pub const HAS_NEON: bool = true;
        pub const HAS_SVE: bool = {};
        pub const HAS_SVE2: bool = {};
        pub const HAS_AES: bool = true;
        pub const HAS_SHA2: bool = true;
        pub const HAS_SHA3: bool = {};
        pub const HAS_CRC32: bool = true;
        pub const HAS_ATOMICS: bool = true;
        pub const HAS_LSE: bool = true;
        pub const HAS_RDM: bool = true;
        pub const HAS_DOTPROD: bool = {};
        pub const HAS_DIT: bool = {};
        "#,
        target.contains("apple"),   // SVE
        target.contains("apple"),   // SVE2
        target.contains("apple"),   // SHA3
        target.contains("apple"),   // DOTPROD
        target.contains("apple"),   // DIT
    );

    fs::write(out_dir.join("cpu_features.rs"), cpu_features).unwrap();

    // Print configuration information
    println!("cargo:warning=Building for target: {}", target);
    println!("cargo:warning=Git hash: {}", git_hash.trim());
    println!("cargo:warning=Build date: {}", build_date);
}
