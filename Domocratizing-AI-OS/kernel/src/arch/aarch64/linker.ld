/* AArch64 linker script for Democratizing AI OS */

ENTRY(_start)

/* Memory layout for AArch64 */
MEMORY {
    /* QEMU virt machine memory layout */
    BOOT (rx)  : ORIGIN = 0x40080000, LENGTH = 64K    /* Boot code */
    RAM  (rwx) : ORIGIN = 0x40090000, LENGTH = 4096M  /* Main memory */
}

/* Section Definitions */
SECTIONS {
    /* Boot section */
    .text.boot : {
        KEEP(*(.text.boot))
    } > BOOT

    /* Kernel sections */
    .text : {
        . = ALIGN(4K);
        _text_start = .;
        *(.text .text.*)
        . = ALIGN(4K);
        _text_end = .;
    } > RAM

    .rodata : {
        . = ALIGN(4K);
        _rodata_start = .;
        *(.rodata .rodata.*)
        . = ALIGN(4K);
        _rodata_end = .;
    } > RAM

    .data : {
        . = ALIGN(4K);
        _data_start = .;
        *(.data .data.*)
        . = ALIGN(4K);
        _data_end = .;
    } > RAM

    .bss : {
        . = ALIGN(4K);
        _bss_start = .;
        *(.bss .bss.* .sbss .sbss.*)
        *(COMMON)
        . = ALIGN(4K);
        _bss_end = .;
    } > RAM

    /* Stack section */
    .stack : {
        . = ALIGN(16);
        _stack_start = .;
        . += 64K * 4; /* 64KB stack per core (4 cores) */
        . = ALIGN(16);
        _stack_end = .;
    } > RAM

    /* Heap section */
    .heap : {
        . = ALIGN(4K);
        _heap_start = .;
        /* Heap grows upward from here */
    } > RAM

    /* Discard unused sections */
    /DISCARD/ : {
        *(.comment)
        *(.note.*)
        *(.eh_frame)
        *(.eh_frame_hdr)
        *(.ARM.*)
        *(.ARM.exidx*)
    }

    /* Debugging sections */
    .debug_info     0 : { *(.debug_info) }
    .debug_abbrev   0 : { *(.debug_abbrev) }
    .debug_aranges  0 : { *(.debug_aranges) }
    .debug_ranges   0 : { *(.debug_ranges) }
    .debug_str      0 : { *(.debug_str) }
    .debug_pubnames 0 : { *(.debug_pubnames) }
    .debug_pubtypes 0 : { *(.debug_pubtypes) }
    .debug_frame    0 : { *(.debug_frame) }
    .debug_line     0 : { *(.debug_line) }
    .debug_loc      0 : { *(.debug_loc) }
    .debug_macinfo  0 : { *(.debug_macinfo) }

    /* Export symbols for kernel */
    PROVIDE(_kernel_start = ADDR(.text.boot));
    PROVIDE(_kernel_end = ADDR(.heap) + SIZEOF(.heap));
    PROVIDE(_text_size = _text_end - _text_start);
    PROVIDE(_rodata_size = _rodata_end - _rodata_start);
    PROVIDE(_data_size = _data_end - _data_start);
    PROVIDE(_bss_size = _bss_end - _bss_start);
    PROVIDE(_stack_size = _stack_end - _stack_start);
    PROVIDE(_heap_size = ORIGIN(RAM) + LENGTH(RAM) - ADDR(.heap));
}

/* Ensure the entry point is properly aligned */
ASSERT(_start % 4 == 0, "Entry point must be aligned to 4 bytes")

/* Ensure sections are properly aligned */
ASSERT(_text_start % 4096 == 0, "Text section must be page-aligned")
ASSERT(_rodata_start % 4096 == 0, "Rodata section must be page-aligned")
ASSERT(_data_start % 4096 == 0, "Data section must be page-aligned")
ASSERT(_bss_start % 4096 == 0, "BSS section must be page-aligned")
ASSERT(_stack_start % 16 == 0, "Stack must be 16-byte aligned")
ASSERT(_heap_start % 4096 == 0, "Heap must be page-aligned")
