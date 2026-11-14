// Pico 2 harness for the Atome LM C engine, driven by harness/atome_run.py over USB serial.
//
// Protocol (PC -> Pico):
//   '?'                                  -> "INFO model=<blob> d=<d> layers=<n> max_seq=<s> sys_hz=<hz> build=<name>"
//   'G' + plen + prompt bytes + n_new    -> one line per generated token, "T <token> <cycles> <xip_acc> <xip_hit>",
//                                           then "E <generated>". Greedy, the engine's own atome_predict_next, which
//                                           re-runs the whole prefix every token (as on the ESP32 demo).
//   'L' + 2 bytes (seconds, LE)          -> "L <tokens> <microseconds>": generates back to back from a fixed
//                                           prompt for that long, for energy measurement. No USB traffic meanwhile.
// Cycles come from the DWT cycle counter and cover atome_predict_next() only. The XIP counters count cache
// accesses and hits during the same call (misses x 8 bytes ~ bytes read from flash).
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "pico/stdlib.h"
#include "hardware/clocks.h"
#include "hardware/structs/xip.h"
#include "atome.h"

#ifndef BUILD_NAME
#define BUILD_NAME "atome"
#endif

extern const uint8_t atome_blob[], atome_blob_end[];

#define DEMCR      (*(volatile uint32_t *)0xE000EDFCu)
#define DWT_CTRL   (*(volatile uint32_t *)0xE0001000u)
#define DWT_CYCCNT (*(volatile uint32_t *)0xE0001004u)

static atome_model_t g_model;
static atome_state_t g_state;

static int read_byte(void) {
    int c;
    do {
        c = getchar_timeout_us(1000000);
    } while (c == PICO_ERROR_TIMEOUT);
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

    const size_t blob_len = (size_t)(atome_blob_end - atome_blob);
    const int load_rc = atome_load(&g_model, atome_blob, blob_len);
    atome_init(&g_state);

    static int toks[ATOME_MAX_SEQ];
    while (true) {
        int cmd = read_byte();
        if (cmd == '?') {
            printf("INFO model=%s d=%d layers=%d d_head=%d max_seq=%d blob=%u load=%d state=%u sys_hz=%lu build=%s\n",
                   BUILD_NAME, ATOME_D_MODEL, ATOME_N_LAYERS, ATOME_D_HEAD, ATOME_MAX_SEQ, (unsigned)blob_len,
                   load_rc, (unsigned)sizeof(atome_state_t), (unsigned long)clock_get_hz(clk_sys), BUILD_NAME);
        } else if (cmd == 'G') {
            int n = read_byte();
            for (int i = 0; i < n; i++) {
                int b = read_byte();
                if (i < ATOME_MAX_SEQ) toks[i] = b;
            }
            if (n > ATOME_MAX_SEQ) n = ATOME_MAX_SEQ;
            int n_new = read_byte(), g = 0;
            while (g < n_new && n < ATOME_MAX_SEQ && n > 0 && load_rc == 0) {
                xip_ctrl_hw->ctr_acc = 0;
                xip_ctrl_hw->ctr_hit = 0;
                uint32_t t0 = DWT_CYCCNT;
                int next = atome_predict_next(&g_model, &g_state, toks, n);
                uint32_t cyc = DWT_CYCCNT - t0;
                uint32_t acc = xip_ctrl_hw->ctr_acc, hit = xip_ctrl_hw->ctr_hit;
                toks[n++] = next;
                g++;
                printf("T %d %lu %lu %lu\n", next, (unsigned long)cyc, (unsigned long)acc, (unsigned long)hit);
            }
            printf("E %d\n", g);
        } else if (cmd == 'L') {
            int lo = read_byte(), hi = read_byte();
            uint64_t us = (uint64_t)(lo | (hi << 8)) * 1000000u;
            uint64_t t0 = time_us_64();
            uint32_t count = 0;
            while (time_us_64() - t0 < us) {
                static const char p[] = "Once";
                int n = 4;
                for (int i = 0; i < n; i++) toks[i] = p[i];
                while (n < ATOME_MAX_SEQ && time_us_64() - t0 < us) {
                    toks[n] = atome_predict_next(&g_model, &g_state, toks, n);
                    n++;
                    count++;
                }
            }
            printf("L %lu %llu\n", (unsigned long)count, (unsigned long long)(time_us_64() - t0));
        }
    }
}
