// Shared declarations for the wide-conv benchmark (see main.c and harness/bench_conv.py).
#pragma once
#include <stdint.h>

#if defined(PICO_ON_DEVICE) && PICO_ON_DEVICE
#include "pico.h"
// The generated code kernels are always placed in SRAM, as mafc does for fc1 (FINDINGS F16).
#define BENCH_IN_SRAM __attribute__((section(".time_critical.bench"), noinline))
#include <arm_acle.h>
#define BENCH_USAT(x, bits) __usat((x), (bits))
#else
#define BENCH_IN_SRAM __attribute__((noinline))
static inline int32_t BENCH_USAT(int32_t x, int bits) {
    const int32_t hi = (int32_t)((1u << bits) - 1);
    return x < 0 ? 0 : x > hi ? hi : x;
}
#endif
