// AArch64 exception vectors for Democratizing AI OS

.macro VECTOR handler
    .align 7
    b       \handler
.endm

.macro SAVE_REGISTERS
    // Save general purpose registers
    stp     x0, x1, [sp, #-16]!
    stp     x2, x3, [sp, #-16]!
    stp     x4, x5, [sp, #-16]!
    stp     x6, x7, [sp, #-16]!
    stp     x8, x9, [sp, #-16]!
    stp     x10, x11, [sp, #-16]!
    stp     x12, x13, [sp, #-16]!
    stp     x14, x15, [sp, #-16]!
    stp     x16, x17, [sp, #-16]!
    stp     x18, x19, [sp, #-16]!
    stp     x20, x21, [sp, #-16]!
    stp     x22, x23, [sp, #-16]!
    stp     x24, x25, [sp, #-16]!
    stp     x26, x27, [sp, #-16]!
    stp     x28, x29, [sp, #-16]!
    str     x30, [sp, #-8]!

    // Save system registers
    mrs     x0, spsr_el1
    mrs     x1, elr_el1
    stp     x0, x1, [sp, #-16]!

    // Save NEON/FP registers if enabled
    mrs     x0, cpacr_el1
    and     x0, x0, #(3 << 20)
    cmp     x0, #(3 << 20)
    bne     1f
    stp     q0, q1, [sp, #-32]!
    stp     q2, q3, [sp, #-32]!
    stp     q4, q5, [sp, #-32]!
    stp     q6, q7, [sp, #-32]!
    stp     q8, q9, [sp, #-32]!
    stp     q10, q11, [sp, #-32]!
    stp     q12, q13, [sp, #-32]!
    stp     q14, q15, [sp, #-32]!
    stp     q16, q17, [sp, #-32]!
    stp     q18, q19, [sp, #-32]!
    stp     q20, q21, [sp, #-32]!
    stp     q22, q23, [sp, #-32]!
    stp     q24, q25, [sp, #-32]!
    stp     q26, q27, [sp, #-32]!
    stp     q28, q29, [sp, #-32]!
    stp     q30, q31, [sp, #-32]!
1:
.endm

.macro RESTORE_REGISTERS
    // Restore NEON/FP registers if enabled
    mrs     x0, cpacr_el1
    and     x0, x0, #(3 << 20)
    cmp     x0, #(3 << 20)
    bne     1f
    ldp     q30, q31, [sp], #32
    ldp     q28, q29, [sp], #32
    ldp     q26, q27, [sp], #32
    ldp     q24, q25, [sp], #32
    ldp     q22, q23, [sp], #32
    ldp     q20, q21, [sp], #32
    ldp     q18, q19, [sp], #32
    ldp     q16, q17, [sp], #32
    ldp     q14, q15, [sp], #32
    ldp     q12, q13, [sp], #32
    ldp     q10, q11, [sp], #32
    ldp     q8, q9, [sp], #32
    ldp     q6, q7, [sp], #32
    ldp     q4, q5, [sp], #32
    ldp     q2, q3, [sp], #32
    ldp     q0, q1, [sp], #32
1:
    // Restore system registers
    ldp     x0, x1, [sp], #16
    msr     spsr_el1, x0
    msr     elr_el1, x1

    // Restore general purpose registers
    ldr     x30, [sp], #8
    ldp     x28, x29, [sp], #16
    ldp     x26, x27, [sp], #16
    ldp     x24, x25, [sp], #16
    ldp     x22, x23, [sp], #16
    ldp     x20, x21, [sp], #16
    ldp     x18, x19, [sp], #16
    ldp     x16, x17, [sp], #16
    ldp     x14, x15, [sp], #16
    ldp     x12, x13, [sp], #16
    ldp     x10, x11, [sp], #16
    ldp     x8, x9, [sp], #16
    ldp     x6, x7, [sp], #16
    ldp     x4, x5, [sp], #16
    ldp     x2, x3, [sp], #16
    ldp     x0, x1, [sp], #16
.endm

.section .text
.align 11
.global exception_vector_table

exception_vector_table:
    // Current EL with SP0
    VECTOR sync_sp0        // Synchronous
    VECTOR irq_sp0         // IRQ
    VECTOR fiq_sp0         // FIQ
    VECTOR error_sp0       // SError

    // Current EL with SPx
    VECTOR sync_spx        // Synchronous
    VECTOR irq_spx         // IRQ
    VECTOR fiq_spx         // FIQ
    VECTOR error_spx       // SError

    // Lower EL using AArch64
    VECTOR sync_aarch64    // Synchronous
    VECTOR irq_aarch64     // IRQ
    VECTOR fiq_aarch64     // FIQ
    VECTOR error_aarch64   // SError

    // Lower EL using AArch32
    VECTOR sync_aarch32    // Synchronous
    VECTOR irq_aarch32     // IRQ
    VECTOR fiq_aarch32     // FIQ
    VECTOR error_aarch32   // SError

// Exception handlers
sync_sp0:
    SAVE_REGISTERS
    mov     x0, sp
    bl      handle_sync_sp0
    RESTORE_REGISTERS
    eret

irq_sp0:
    SAVE_REGISTERS
    mov     x0, sp
    bl      handle_irq_sp0
    RESTORE_REGISTERS
    eret

fiq_sp0:
    SAVE_REGISTERS
    mov     x0, sp
    bl      handle_fiq_sp0
    RESTORE_REGISTERS
    eret

error_sp0:
    SAVE_REGISTERS
    mov     x0, sp
    bl      handle_error_sp0
    RESTORE_REGISTERS
    eret

sync_spx:
    SAVE_REGISTERS
    mov     x0, sp
    bl      handle_sync_spx
    RESTORE_REGISTERS
    eret

irq_spx:
    SAVE_REGISTERS
    mov     x0, sp
    bl      handle_irq_spx
    RESTORE_REGISTERS
    eret

fiq_spx:
    SAVE_REGISTERS
    mov     x0, sp
    bl      handle_fiq_spx
    RESTORE_REGISTERS
    eret

error_spx:
    SAVE_REGISTERS
    mov     x0, sp
    bl      handle_error_spx
    RESTORE_REGISTERS
    eret

sync_aarch64:
    SAVE_REGISTERS
    mov     x0, sp
    bl      handle_sync_aarch64
    RESTORE_REGISTERS
    eret

irq_aarch64:
    SAVE_REGISTERS
    mov     x0, sp
    bl      handle_irq_aarch64
    RESTORE_REGISTERS
    eret

fiq_aarch64:
    SAVE_REGISTERS
    mov     x0, sp
    bl      handle_fiq_aarch64
    RESTORE_REGISTERS
    eret

error_aarch64:
    SAVE_REGISTERS
    mov     x0, sp
    bl      handle_error_aarch64
    RESTORE_REGISTERS
    eret

sync_aarch32:
    SAVE_REGISTERS
    mov     x0, sp
    bl      handle_sync_aarch32
    RESTORE_REGISTERS
    eret

irq_aarch32:
    SAVE_REGISTERS
    mov     x0, sp
    bl      handle_irq_aarch32
    RESTORE_REGISTERS
    eret

fiq_aarch32:
    SAVE_REGISTERS
    mov     x0, sp
    bl      handle_fiq_aarch32
    RESTORE_REGISTERS
    eret

error_aarch32:
    SAVE_REGISTERS
    mov     x0, sp
    bl      handle_error_aarch32
    RESTORE_REGISTERS
    eret
