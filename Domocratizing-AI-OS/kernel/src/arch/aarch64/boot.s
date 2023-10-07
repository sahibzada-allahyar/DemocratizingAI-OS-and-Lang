// AArch64 boot code for Democratizing AI OS

.section .text.boot
.global _start

_start:
    // Check processor ID is 0 (primary core)
    mrs     x0, mpidr_el1
    and     x0, x0, #0xFF
    cbz     x0, primary_core
    // Secondary cores wait for events
    wfe
    b       .

primary_core:
    // Set stack pointer for each exception level
    // EL3
    mrs     x0, CurrentEL
    cmp     x0, #12
    bne     2f
    ldr     x0, =_stack_end
    mov     sp, x0
    b       el3_setup

2:  // EL2
    cmp     x0, #8
    bne     1f
    ldr     x0, =_stack_end
    mov     sp, x0
    b       el2_setup

1:  // EL1
    cmp     x0, #4
    bne     0f
    ldr     x0, =_stack_end
    mov     sp, x0
    b       el1_setup

0:  // Something went wrong, hang
    b       .

el3_setup:
    // Configure EL3 to EL2 transition
    ldr     x0, =0x5b1    // RW=1, HCE=1, SMD=1, RES1=1, NS=1
    msr     scr_el3, x0
    mov     x0, #0x3c9    // DAIF=1, M[4]=1 (EL2h)
    msr     spsr_el3, x0
    adr     x0, el2_setup
    msr     elr_el3, x0
    eret

el2_setup:
    // Configure EL2 to EL1 transition
    mov     x0, #(1 << 31)    // Enable AArch64 for EL1
    orr     x0, x0, #(1 << 1) // Enable EL1 access to timer
    msr     hcr_el2, x0
    mov     x0, #0x3c5        // DAIF=1, M[4]=1 (EL1h)
    msr     spsr_el2, x0
    adr     x0, el1_setup
    msr     elr_el2, x0
    eret

el1_setup:
    // Enable floating point and NEON
    mov     x0, #(3 << 20)
    msr     cpacr_el1, x0

    // Enable MMU
    mov     x0, #0x1005      // SCTLR_EL1 flags
    movk    x0, #0x0800, lsl #16
    msr     sctlr_el1, x0

    // Clear BSS
    ldr     x1, =_bss_start
    ldr     x2, =_bss_end
clear_bss:
    cmp     x1, x2
    beq     setup_stack
    str     xzr, [x1], #8
    b       clear_bss

setup_stack:
    // Set up stack for primary core
    mrs     x0, mpidr_el1
    and     x0, x0, #0xFF
    mov     x1, #64*1024     // 64KB per core
    mul     x1, x1, x0       // Offset by core ID
    ldr     x2, =_stack_end
    sub     x2, x2, x1       // Stack grows down
    mov     sp, x2

    // Enable interrupts
    msr     daifclr, #0xf

    // Jump to kernel_main
    ldr     x0, =kernel_main
    blr     x0

    // Should never reach here
    b       .
