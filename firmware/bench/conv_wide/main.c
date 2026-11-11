// Wide-conv fairness benchmark on the Pico 2 (FINDINGS F17). One 3x3 conv, 10x10x32 -> 8x8x32, ternary
// weights at 50/70/90% zeros. Kernels:
//   cmsis_nn   arm_convolve_wrapper_s8 (im2col + SMLAD matrix kernel), the library's path for this shape
//   mafc_code  ternary weights compiled into code (harness/bench_conv.py gen), placed in SRAM
// Every kernel must produce the same bytes (same requantization); the checksum is checked on the board.
//
// Protocol (PC -> Pico):  '?' -> "INFO ..." ;  'B' -> one "B <kernel> <case> <nnz> <min> <mean> <chk> <ok>"
// line per kernel and case, then "END". Driven by harness/bench_conv.py run.
#include <stdio.h>
#include <string.h>
#include "pico/stdlib.h"
#include "hardware/clocks.h"
#include "arm_nnfunctions.h"
#include "gen/bench_data.h"

#ifndef BUILD_NAME
#define BUILD_NAME "bench_conv"
#endif
#define REPS 20

#define DEMCR      (*(volatile uint32_t *)0xE000EDFCu)
#define DWT_CTRL   (*(volatile uint32_t *)0xE0001000u)
#define DWT_CYCCNT (*(volatile uint32_t *)0xE0001004u)

static int8_t out[BOH * BOW * BCOUT];
static int8_t scratch[8192] __attribute__((aligned(4)));
static int32_t q_mult[BCOUT], q_shift[BCOUT];

static uint32_t checksum(const int8_t *o) {
    uint32_t c = 0;
    for (int i = 0; i < BOH * BOW * BCOUT; i++) c += (uint32_t)o[i] * (uint32_t)(i % 251 + 1);
    return c;
}

static void run_cmsis(const bench_case_t *bc) {
    const cmsis_nn_context ctx = {scratch, sizeof scratch};
    cmsis_nn_conv_params cp = {0};
    cp.input_offset = 0;
    cp.output_offset = 0;
    cp.stride.w = cp.stride.h = 1;
    cp.padding.w = cp.padding.h = 0;
    cp.dilation.w = cp.dilation.h = 1;
    cp.activation.min = 0;
    cp.activation.max = 127;
    for (int i = 0; i < BCOUT; i++) { q_mult[i] = 1 << 30; q_shift[i] = 1 - bc->shift; }
    const cmsis_nn_per_channel_quant_params q = {q_mult, q_shift};
    const cmsis_nn_dims idim = {1, BH, BW, BCIN}, fdim = {BCOUT, 3, 3, BCIN}, odim = {1, BOH, BOW, BCOUT};
    const cmsis_nn_dims bdim = {0, 0, 0, 0};
    arm_cmsis_nn_status st = arm_convolve_wrapper_s8(&ctx, &cp, &q, &idim, b_act, &fdim, bc->w, &bdim, NULL, &odim, out);
    if (st != ARM_CMSIS_NN_SUCCESS) panic("cmsis status %d", st);
}

static void report(const char *kernel, const bench_case_t *bc, uint32_t mn, uint32_t sum) {
    uint32_t chk = checksum(out);
    printf("B %s %s %d %lu %lu %lu %d\n", kernel, bc->name, bc->nnz, (unsigned long)mn,
           (unsigned long)(sum / REPS), (unsigned long)chk, chk == bc->chk);
}

static void bench(void) {
    for (int k = 0; k < BN_SPARSITY; k++) {
        const bench_case_t *bc = &b_cases[k];
        uint32_t mn = UINT32_MAX, sum = 0;
        for (int r = 0; r < REPS; r++) {
            memset(out, 0x55, sizeof out);
            uint32_t t0 = DWT_CYCCNT;
            run_cmsis(bc);
            uint32_t c = DWT_CYCCNT - t0;
            sum += c;
            if (c < mn) mn = c;
        }
        report("cmsis_nn", bc, mn, sum);

        mn = UINT32_MAX, sum = 0;
        for (int r = 0; r < REPS; r++) {
            memset(out, 0x55, sizeof out);
            uint32_t t0 = DWT_CYCCNT;
            bc->code(b_act, out);
            uint32_t c = DWT_CYCCNT - t0;
            sum += c;
            if (c < mn) mn = c;
        }
        report("mafc_code", bc, mn, sum);
    }
    printf("END\n");
}

static int read_byte(void) {
    int c;
    do { c = getchar_timeout_us(1000000); } while (c == PICO_ERROR_TIMEOUT);
    return c;
}

int main(void) {
    stdio_init_all();
    DEMCR |= 1u << 24;
    DWT_CYCCNT = 0;
    DWT_CTRL |= 1u;
    cmsis_nn_conv_params cp = {0};
    cp.stride.w = cp.stride.h = 1;
    cp.dilation.w = cp.dilation.h = 1;
    const cmsis_nn_dims idim = {1, BH, BW, BCIN}, fdim = {BCOUT, 3, 3, BCIN}, odim = {1, BOH, BOW, BCOUT};
    while (true) {
        int cmd = read_byte();
        if (cmd == '?') {
            int32_t need = arm_convolve_wrapper_s8_get_buffer_size(&cp, &idim, &fdim, &odim);
            printf("INFO bench=conv_wide sys_hz=%lu build=%s scratch_need=%ld\n",
                   (unsigned long)clock_get_hz(clk_sys), BUILD_NAME, (long)need);
        } else if (cmd == 'B') {
            bench();
        }
    }
}
