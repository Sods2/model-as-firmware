// Pico 2 firmware for the plain-C reference, driven by harness/run.py over USB serial.
//
// Protocol (PC -> Pico):
//   '?'               -> "INFO model=<header> sys_hz=<hz> build=<name>"
//   'I' + 256 bytes   -> "R <pred> <cycles> <logit0> ... <logit9>"
//   'T'               -> "T <stage>=<cycles> ..." for the last image
// Cycles come from the DWT cycle counter and cover ref_infer() only.
#include <stdio.h>
#include <stdint.h>
#include "pico/stdlib.h"
#include "hardware/clocks.h"
#include "ref_infer.h"

#ifndef BUILD_NAME
#define BUILD_NAME "ref_c"
#endif

#define DEMCR      (*(volatile uint32_t *)0xE000EDFCu)
#define DWT_CTRL   (*(volatile uint32_t *)0xE0001000u)
#define DWT_CYCCNT (*(volatile uint32_t *)0xE0001004u)

static void cyccnt_init(void) {
    DEMCR |= 1u << 24;     // TRCENA
    DWT_CYCCNT = 0;
    DWT_CTRL |= 1u << 0;   // CYCCNTENA
}

__attribute__((weak)) void ref_init(void) {}

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
    cyccnt_init();
    ref_init();

    static int8_t img[REF_IMG_PIXELS];
    int32_t logits[REF_N_CLASSES];
    uint32_t n = 0;

    while (true) {
        int cmd = read_byte();
        if (cmd == '?') {
            printf("INFO model=%s sys_hz=%lu build=%s\n", ref_model_name,
                   (unsigned long)clock_get_hz(clk_sys), BUILD_NAME);
        } else if (cmd == 'I') {
            for (int i = 0; i < REF_IMG_PIXELS; i++) img[i] = (int8_t)read_byte();
            uint32_t t0 = DWT_CYCCNT;
            uint32_t pred = ref_infer(img, logits);
            uint32_t cycles = DWT_CYCCNT - t0;
            printf("R %lu %lu", (unsigned long)pred, (unsigned long)cycles);
            for (int i = 0; i < REF_N_CLASSES; i++) printf(" %ld", (long)logits[i]);
            printf("\n");
            gpio_put(PICO_DEFAULT_LED_PIN, (++n >> 6) & 1);  // slow blink while busy
        } else if (cmd == 'T') {
            printf("T");
            for (int i = 0; i < ref_n_stages; i++) printf(" %s=%lu", ref_stage_names[i], (unsigned long)ref_stage_cycles[i]);
            printf("\n");
        }
        // anything else (stray newlines etc.) is ignored
    }
}
