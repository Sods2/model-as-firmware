// Bring-up firmware for the Pico 2 (RP2350, Cortex-M33).
// - Blinks the onboard LED (proof of life)
// - Prints over USB serial (stdio_usb -> COM port)
// - Times a fixed workload with the DWT cycle counter, the same mechanism
//   the inference benchmarks will use.
#include <stdio.h>
#include <stdint.h>
#include "pico/stdlib.h"
#include "hardware/clocks.h"

// Cortex-M33 debug/trace registers (ARMv8-M architecture addresses)
#define DEMCR      (*(volatile uint32_t *)0xE000EDFCu)
#define DWT_CTRL   (*(volatile uint32_t *)0xE0001000u)
#define DWT_CYCCNT (*(volatile uint32_t *)0xE0001004u)
#define DEMCR_TRCENA       (1u << 24)
#define DWT_CTRL_CYCCNTENA (1u << 0)

static void cyccnt_init(void) {
    DEMCR |= DEMCR_TRCENA;
    DWT_CYCCNT = 0;
    DWT_CTRL |= DWT_CTRL_CYCCNTENA;
}

static inline uint32_t cyccnt(void) { return DWT_CYCCNT; }

// Fixed workload: N multiply-accumulates. volatile stops the compiler
// from folding the loop away.
#define LOOP_N 100000u
static uint32_t __attribute__((noinline)) workload(void) {
    volatile uint32_t acc = 0;
    for (uint32_t i = 0; i < LOOP_N; i++) acc += i * 3u;
    return acc;
}

int main(void) {
    stdio_init_all();

    gpio_init(PICO_DEFAULT_LED_PIN);
    gpio_set_dir(PICO_DEFAULT_LED_PIN, GPIO_OUT);

    cyccnt_init();
    const uint32_t sys_hz = clock_get_hz(clk_sys);

    // Sanity check: is the counter running at all?
    uint32_t a = cyccnt();
    busy_wait_us(1000);
    uint32_t b = cyccnt();

    uint32_t run = 0;
    while (true) {
        gpio_put(PICO_DEFAULT_LED_PIN, run & 1);

        uint32_t t0 = cyccnt();
        uint32_t r = workload();
        uint32_t cycles = cyccnt() - t0;

        printf("hello run=%lu sys_hz=%lu cal_1ms=%lu loop_n=%lu cycles=%lu cyc_per_iter=%.2f us=%.1f r=%lu\n",
               (unsigned long)run, (unsigned long)sys_hz, (unsigned long)(b - a),
               (unsigned long)LOOP_N, (unsigned long)cycles,
               (double)cycles / LOOP_N, cycles * 1e6 / sys_hz, (unsigned long)r);
        run++;
        sleep_ms(500);
    }
}
