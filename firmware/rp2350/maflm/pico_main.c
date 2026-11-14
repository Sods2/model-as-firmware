// Pico 2 harness for the MafLM runtime, driven by harness/lm_run.py over USB serial.
//
// Protocol (PC -> Pico):
//   '?'                         -> "INFO model=<name> build=<build> window=<W> tree=<0|1> sys_hz=<hz>"
//   'R'                         -> empties the KV cache; "OK"
//   'S' + token                 -> runs one token: "S <argmax> <cycles> <checksum hex> <xip_acc> <xip_hit>"
//                                  checksum = sum_i logits[i] * (2i + 1) mod 2^64, over the int64 logits
//   'G' + plen + prompt + n     -> fresh KV cache, runs the prompt ("P <cycles>" for all of it), then generates
//                                  n tokens greedily: "T <token> <cycles> <xip_acc> <xip_hit>" each, then "E <n>"
//   'Q'                         -> "Q <qkv> <attn> <wo> <ffn1> <ffn2> <unembed> <tables>": stage cycles since the last 'R'/'G'
//                                  (zeros unless built with LM_PROFILE)
//   'L' + 2 bytes (seconds, LE) -> generates back to back from "Once" (restarting at the window end) for that long,
//                                  for energy measurement: "L <tokens> <microseconds>"
// Cycles come from the DWT cycle counter and cover lm_step() only.
#include <stdio.h>
#include <stdint.h>
#include "pico/stdlib.h"
#include "hardware/clocks.h"
#include "hardware/structs/xip.h"
#include "lm_rt.h"

#ifndef BUILD_NAME
#define BUILD_NAME "maflm"
#endif

#define DEMCR      (*(volatile uint32_t *)0xE000EDFCu)
#define DWT_CTRL   (*(volatile uint32_t *)0xE0001000u)
#define DWT_CYCCNT (*(volatile uint32_t *)0xE0001004u)

static int read_byte(void) {
    int c;
    do {
        c = getchar_timeout_us(1000000);
    } while (c == PICO_ERROR_TIMEOUT);
    return c;
}

static uint32_t timed_step(int tok, int *next, uint32_t *acc, uint32_t *hit) {
    xip_ctrl_hw->ctr_acc = 0;
    xip_ctrl_hw->ctr_hit = 0;
    uint32_t t0 = DWT_CYCCNT;
    *next = lm_step(tok);
    uint32_t c = DWT_CYCCNT - t0;
    *acc = xip_ctrl_hw->ctr_acc;
    *hit = xip_ctrl_hw->ctr_hit;
    return c;
}

int main(void) {
    stdio_init_all();
    gpio_init(PICO_DEFAULT_LED_PIN);
    gpio_set_dir(PICO_DEFAULT_LED_PIN, GPIO_OUT);
    gpio_put(PICO_DEFAULT_LED_PIN, 1);
    DEMCR |= 1u << 24;
    DWT_CYCCNT = 0;
    DWT_CTRL |= 1u;
    lm_init();

    while (true) {
        int cmd = read_byte(), next;
        uint32_t acc, hit;
        if (cmd == '?') {
            printf("INFO model=%s build=%s window=%d tree=%d sys_hz=%lu\n", LM_MODEL_NAME, BUILD_NAME, LM_WINDOW,
                   LM_TREE, (unsigned long)clock_get_hz(clk_sys));
        } else if (cmd == 'R') {
            lm_reset();
            printf("OK\n");
        } else if (cmd == 'S') {
            int tok = read_byte();
            uint32_t c = timed_step(tok, &next, &acc, &hit);
            uint64_t cs = 0;
            for (int i = 0; i < LM_VOCAB; i++) cs += (uint64_t)lm_logits[i] * (uint64_t)(2 * i + 1);
            printf("S %d %lu %016llx %lu %lu\n", next, (unsigned long)c, (unsigned long long)cs, (unsigned long)acc,
                   (unsigned long)hit);
        } else if (cmd == 'G') {
            int plen = read_byte();
            static int prompt[256];
            for (int i = 0; i < plen; i++) prompt[i] = read_byte();
            int n = read_byte();
            lm_reset();
            uint32_t pc = 0;
            next = 0;
            for (int i = 0; i < plen; i++) pc += timed_step(prompt[i], &next, &acc, &hit);
            printf("P %lu\n", (unsigned long)pc);
            int g = 0;
            for (; g < n && lm_pos < LM_WINDOW; g++) {
                int tok = next;
                uint32_t c = timed_step(tok, &next, &acc, &hit);
                printf("T %d %lu %lu %lu\n", tok, (unsigned long)c, (unsigned long)acc, (unsigned long)hit);
            }
            printf("E %d\n", g);
        } else if (cmd == 'Q') {
            printf("Q");
            for (int i = 0; i <= LM_ST_N; i++) printf(" %lu", (unsigned long)lm_stage_cycles[i]);
            printf("\n");
        } else if (cmd == 'L') {
            int lo = read_byte(), hi = read_byte();
            uint64_t us = (uint64_t)(lo | (hi << 8)) * 1000000u, t0 = time_us_64();
            uint32_t count = 0;
            while (time_us_64() - t0 < us) {
                lm_reset();
                int tok = 'O';
                for (const char *p = "nce"; *p; p++) lm_step(tok), tok = *p;
                while (lm_pos < LM_WINDOW && time_us_64() - t0 < us) { tok = lm_step(tok); count++; }
            }
            printf("L %lu %llu\n", (unsigned long)count, (unsigned long long)(time_us_64() - t0));
        }
    }
}
