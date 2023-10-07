//! AArch64 cache management

use core::arch::asm;
use core::ptr;

use crate::arch::aarch64::registers::*;

/// Cache line size
pub const CACHE_LINE_SIZE: usize = 64;

/// Cache type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheType {
    /// Data cache
    Data,
    /// Instruction cache
    Instruction,
    /// Unified cache
    Unified,
}

/// Cache level information
#[derive(Debug, Clone)]
pub struct CacheLevelInfo {
    /// Cache type
    pub cache_type: CacheType,
    /// Cache size in bytes
    pub size: usize,
    /// Cache ways
    pub ways: usize,
    /// Cache sets
    pub sets: usize,
    /// Cache line size
    pub line_size: usize,
}

/// Cache implementation
pub struct Cache {
    /// Number of cache levels
    levels: usize,
    /// Cache level information
    level_info: [Option<CacheLevelInfo>; 7],
}

impl Cache {
    /// Create new cache instance
    pub fn new() -> Self {
        let mut cache = Cache {
            levels: 0,
            level_info: [None; 7],
        };
        cache.detect_caches();
        cache
    }

    /// Detect cache configuration
    fn detect_caches(&mut self) {
        unsafe {
            let clidr = clidr_el1();
            self.levels = ((clidr >> 21) & 7) as usize;

            for level in 0..self.levels {
                let cache_type = ((clidr >> (level * 3)) & 7) as u8;
                if cache_type == 0 {
                    continue;
                }

                // Select cache level
                set_csselr_el1((level << 1) as u64);
                isb();

                let ccsidr = ccsidr_el1();
                let line_size = 1 << ((ccsidr & 7) + 4);
                let ways = ((ccsidr >> 3) & 0x3FF) + 1;
                let sets = ((ccsidr >> 13) & 0x7FFF) + 1;
                let size = line_size * ways * sets;

                let cache_type = match cache_type {
                    1 => CacheType::Instruction,
                    2 => CacheType::Data,
                    3 => CacheType::Unified,
                    _ => continue,
                };

                self.level_info[level] = Some(CacheLevelInfo {
                    cache_type,
                    size,
                    ways: ways as usize,
                    sets: sets as usize,
                    line_size,
                });
            }
        }
    }

    /// Get cache level information
    pub fn level_info(&self, level: usize) -> Option<&CacheLevelInfo> {
        self.level_info[level].as_ref()
    }

    /// Get number of cache levels
    pub fn levels(&self) -> usize {
        self.levels
    }

    /// Clean data cache by virtual address
    pub fn clean_dcache_by_va(&self, addr: *const u8, size: usize) {
        let mut ptr = addr as usize;
        let end = ptr + size;

        while ptr < end {
            unsafe {
                asm!("dc cvac, {}", in(reg) ptr);
            }
            ptr += CACHE_LINE_SIZE;
        }

        dsb_sy();
    }

    /// Clean data cache by set/way
    pub fn clean_dcache_by_set_way(&self) {
        unsafe {
            let clidr = clidr_el1();
            let loc = (clidr >> 24) & 7;

            for level in 0..loc {
                if ((clidr >> (level * 3)) & 7) == 0 {
                    continue;
                }

                set_csselr_el1((level << 1) as u64);
                isb();

                let ccsidr = ccsidr_el1();
                let ways = ((ccsidr >> 3) & 0x3FF) + 1;
                let sets = ((ccsidr >> 13) & 0x7FFF) + 1;

                for way in 0..ways {
                    for set in 0..sets {
                        let val = (level << 1) | (way << 28) | (set << 6);
                        asm!("dc csw, {}", in(reg) val);
                    }
                }
            }

            dsb_sy();
        }
    }

    /// Invalidate data cache by virtual address
    pub fn invalidate_dcache_by_va(&self, addr: *const u8, size: usize) {
        let mut ptr = addr as usize;
        let end = ptr + size;

        while ptr < end {
            unsafe {
                asm!("dc ivac, {}", in(reg) ptr);
            }
            ptr += CACHE_LINE_SIZE;
        }

        dsb_sy();
    }

    /// Invalidate data cache by set/way
    pub fn invalidate_dcache_by_set_way(&self) {
        unsafe {
            let clidr = clidr_el1();
            let loc = (clidr >> 24) & 7;

            for level in 0..loc {
                if ((clidr >> (level * 3)) & 7) == 0 {
                    continue;
                }

                set_csselr_el1((level << 1) as u64);
                isb();

                let ccsidr = ccsidr_el1();
                let ways = ((ccsidr >> 3) & 0x3FF) + 1;
                let sets = ((ccsidr >> 13) & 0x7FFF) + 1;

                for way in 0..ways {
                    for set in 0..sets {
                        let val = (level << 1) | (way << 28) | (set << 6);
                        asm!("dc isw, {}", in(reg) val);
                    }
                }
            }

            dsb_sy();
        }
    }

    /// Clean and invalidate data cache by virtual address
    pub fn clean_invalidate_dcache_by_va(&self, addr: *const u8, size: usize) {
        let mut ptr = addr as usize;
        let end = ptr + size;

        while ptr < end {
            unsafe {
                asm!("dc civac, {}", in(reg) ptr);
            }
            ptr += CACHE_LINE_SIZE;
        }

        dsb_sy();
    }

    /// Clean and invalidate data cache by set/way
    pub fn clean_invalidate_dcache_by_set_way(&self) {
        unsafe {
            let clidr = clidr_el1();
            let loc = (clidr >> 24) & 7;

            for level in 0..loc {
                if ((clidr >> (level * 3)) & 7) == 0 {
                    continue;
                }

                set_csselr_el1((level << 1) as u64);
                isb();

                let ccsidr = ccsidr_el1();
                let ways = ((ccsidr >> 3) & 0x3FF) + 1;
                let sets = ((ccsidr >> 13) & 0x7FFF) + 1;

                for way in 0..ways {
                    for set in 0..sets {
                        let val = (level << 1) | (way << 28) | (set << 6);
                        asm!("dc cisw, {}", in(reg) val);
                    }
                }
            }

            dsb_sy();
        }
    }

    /// Invalidate instruction cache
    pub fn invalidate_icache(&self) {
        unsafe {
            asm!("ic ialluis");
            dsb_sy();
            isb();
        }
    }

    /// Invalidate instruction cache by virtual address
    pub fn invalidate_icache_by_va(&self, addr: *const u8, size: usize) {
        let mut ptr = addr as usize;
        let end = ptr + size;

        while ptr < end {
            unsafe {
                asm!("ic ivau, {}", in(reg) ptr);
            }
            ptr += CACHE_LINE_SIZE;
        }

        dsb_sy();
        isb();
    }
}

/// Global cache instance
static mut CACHE: Option<Cache> = None;

/// Initialize cache
pub fn init() {
    unsafe {
        CACHE = Some(Cache::new());
    }
}

/// Get cache instance
pub fn cache() -> &'static Cache {
    unsafe { CACHE.as_ref().unwrap() }
}
