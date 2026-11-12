// Idea C benchmark on the Pico 2 (FINDINGS F21): one dense int8 linear layer, 256 -> 256, whose 64 KB of
// weights live in flash (4x the 16 KB XIP cache). The same SXTB16/SMLAD kernel runs over the weights read in
// different ways:
//   sram         weights copied to SRAM at boot (upper bound: compute only)
//   xip_cached   read in place through the XIP cache (0x10000000), as every flash build does today
//   xip_nocache  read in place through the uncached alias (0x14000000)
//   dma_seq      DMA a 2 KB block from the uncached alias into SRAM, wait, compute it (no overlap)
//   dma_alias    the same, double-buffered: DMA block k+1 while computing block k
//   dma_stream   double-buffered, but the DMA reads the XIP streaming FIFO (XIP_AUX), fed by a background
//                stream read of the whole weight array (DREQ_XIP_STREAM)
//   stream_seq   the stream FIFO without overlap: DMA a block, wait, compute it (separates transfer from overlap)
// All kernels and the benchmark loop run from SRAM, so instruction fetch never touches flash.
// Every variant's output is checked against the checksum from harness/bench_stream.py gen.
//
// Protocol (PC -> Pico): '?' -> "INFO ..."; 'B' -> "B <variant> <min> <mean> <chk> <ok>" lines, then "END".
#include <stdio.h>
#include <string.h>
#include "pico/stdlib.h"
#include "hardware/clocks.h"
#include "hardware/dma.h"
#include "hardware/structs/xip.h"
#include "hardware/structs/xip_aux.h"
#include "hardware/regs/addressmap.h"
#include <arm_acle.h>
#include "gen/stream_data.h"

#ifndef BUILD_NAME
#define BUILD_NAME "bench_stream"
#endif
#define REPS 10
#define ROWS_PER_BLOCK 8
#define BLOCK_BYTES (ROWS_PER_BLOCK * S_IN)
#define N_BLOCKS (S_OUT / ROWS_PER_BLOCK)
#define RAM __not_in_flash_func

#define DEMCR      (*(volatile uint32_t *)0xE000EDFCu)
#define DWT_CTRL   (*(volatile uint32_t *)0xE0001000u)
#define DWT_CYCCNT (*(volatile uint32_t *)0xE0001004u)

static int8_t w_sram[S_OUT * S_IN] __attribute__((aligned(4)));
static int16_t act[S_IN] __attribute__((aligned(4)));   // the input, copied to SRAM: only the weights stay in flash
static uint32_t buf[2][BLOCK_BYTES / 4];
static int32_t out[S_OUT];
static int dma_ch;

// One block of rows: out[r] = sum_j w[r][j] * act[j]; w in the w0 w2 w1 w3 byte order (see bench_stream.py)
static void RAM(rows)(const int8_t *w, int32_t *o, int n_rows) {
    for (int r = 0; r < n_rows; r++) {
        const uint32_t *ww = (const uint32_t *)(w + r * S_IN);
        const int16_t *a = act;
        int32_t s = 0;
        for (int j = 0; j < S_IN / 4; j++) {
            uint32_t x = ww[j], a01, a23;
            memcpy(&a01, a + 4 * j, 4);
            memcpy(&a23, a + 4 * j + 2, 4);
            s = __smlad(__sxtb16(x), a01, s);
            s = __smlad(__sxtb16(__ror(x, 8)), a23, s);
        }
        o[r] = s;
    }
}

static void RAM(run_inplace)(const int8_t *w) { rows(w, out, S_OUT); }

static void RAM(dma_start)(void *dst, const void *src, bool from_stream) {
    dma_channel_config c = dma_channel_get_default_config(dma_ch);
    channel_config_set_transfer_data_size(&c, DMA_SIZE_32);
    channel_config_set_read_increment(&c, !from_stream);
    channel_config_set_write_increment(&c, true);
    if (from_stream) channel_config_set_dreq(&c, DREQ_XIP_STREAM);
    dma_channel_configure(dma_ch, &c, dst, src, BLOCK_BYTES / 4, true);
}

static void RAM(run_dma)(int mode) {   // 0 = sequential alias, 1 = double-buffered alias, 2 = double-buffered stream, 3 = sequential stream
    const bool stream = mode >= 2;
    const int8_t *alias = (const int8_t *)((uintptr_t)s_w - XIP_BASE + XIP_NOCACHE_NOALLOC_BASE);
    if (stream) {
        xip_ctrl_hw->stream_ctr = 0;
        while (!(xip_ctrl_hw->stat & XIP_STAT_FIFO_EMPTY_BITS)) (void)xip_aux_hw->stream;
        xip_ctrl_hw->stream_addr = (uint32_t)(uintptr_t)s_w;
        xip_ctrl_hw->stream_ctr = S_OUT * S_IN / 4;
    }
    const void *src = stream ? (const void *)&xip_aux_hw->stream : (const void *)alias;
    dma_start(buf[0], src, stream);
    for (int k = 0; k < N_BLOCKS; k++) {
        dma_channel_wait_for_finish_blocking(dma_ch);
        if (mode == 0 || mode == 3) {
            rows((const int8_t *)buf[0], out + k * ROWS_PER_BLOCK, ROWS_PER_BLOCK);
            if (k + 1 < N_BLOCKS) dma_start(buf[0], stream ? src : alias + (k + 1) * BLOCK_BYTES, stream);
        } else {
            if (k + 1 < N_BLOCKS)
                dma_start(buf[(k + 1) & 1], stream ? src : alias + (k + 1) * BLOCK_BYTES, stream);
            rows((const int8_t *)buf[k & 1], out + k * ROWS_PER_BLOCK, ROWS_PER_BLOCK);
        }
    }
}

static uint32_t checksum(void) {
    uint32_t c = 0;
    for (int i = 0; i < S_OUT; i++) c += (uint32_t)out[i] * (uint32_t)(i + 1);
    return c;
}

static void RAM(measure)(const char *name, int variant) {
    uint32_t mn = UINT32_MAX, sum = 0;
    for (int r = 0; r < REPS; r++) {
        memset(out, 0, sizeof out);
        uint32_t t0 = DWT_CYCCNT;
        switch (variant) {
            case 0: run_inplace(w_sram); break;
            case 1: run_inplace(s_w); break;
            case 2: run_inplace((const int8_t *)((uintptr_t)s_w - XIP_BASE + XIP_NOCACHE_NOALLOC_BASE)); break;
            case 3: run_dma(0); break;
            case 4: run_dma(1); break;
            case 5: run_dma(2); break;
            case 6: run_dma(3); break;
        }
        uint32_t c = DWT_CYCCNT - t0;
        sum += c;
        if (c < mn) mn = c;
    }
    uint32_t chk = checksum();
    printf("B %s %lu %lu %lu %d\n", name, (unsigned long)mn, (unsigned long)(sum / REPS), (unsigned long)chk,
           chk == S_CHK);
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
    memcpy(w_sram, s_w, sizeof w_sram);
    memcpy(act, s_act, sizeof act);
    dma_ch = dma_claim_unused_channel(true);
    while (true) {
        int cmd = read_byte();
        if (cmd == '?') {
            printf("INFO bench=stream sys_hz=%lu build=%s weights_at=%p kb=%d\n", (unsigned long)clock_get_hz(clk_sys),
                   BUILD_NAME, (const void *)s_w, (int)(sizeof s_w / 1024));
        } else if (cmd == 'B') {
            measure("sram", 0);
            measure("xip_cached", 1);
            measure("xip_nocache", 2);
            measure("dma_seq", 3);
            measure("dma_alias", 4);
            measure("dma_stream", 5);
            measure("stream_seq", 6);
            printf("END\n");
        }
    }
}
