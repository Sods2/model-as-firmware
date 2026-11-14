// The .atome model blob, linked in as read-only data (flash, or SRAM in a copy_to_ram build).
// The blob starts with the 7-byte magic "ATOME01", and atome_load() points the LayerNorm and SSM float arrays
// straight into it. On the Cortex-M33 an unaligned float load (VLDR) faults, so the blob starts at 4k+1: everything
// after the magic is then word-aligned (all the packed ternary arrays are whole words).
    .section .rodata.atome_blob, "a"
    .balign 4
    .space 1
    .global atome_blob
atome_blob:
    .incbin ATOME_BIN_PATH
    .global atome_blob_end
atome_blob_end:
