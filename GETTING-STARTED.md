# Model as Firmware — getting started (wiring first, then Phase 0 → Gate 0)

## Context
The project is `the repo folder`, which has a GitHub remote, the docs (HANDOFF.md, FEASIBILITY-STUDY.md, SHOPPING-LIST.md) and no code yet. Hardware on hand:
- **1× Pico 2 (RP2350) with headers**
- Micro-USB data cable
- Inline USB-A power meter
- Raspberry Pi 4

You do **not** have the Debug Probe, breadboard or jumper wires. The PC has VS Code, Python 3.12 and an RTX 3060 Ti, which is enough to train the model. None of the Pico toolchain is installed. You asked first for *what connects to what and how*, then a plan up to Gate 0 (HANDOFF §6, steps 1–6).

Main change from the handoff: without a Debug Probe, **flashing and the test link both go over the Pico's own USB**. Flashing uses BOOTSEL drag-and-drop or `picotool`, and the test link is USB serial (CDC). That covers everything up to Gate 0. SWD step-through debugging is optional. The Pi 4 can stand in for the probe later.

---

## 1. Wiring

### Setup A — everything up to Gate 0 (no extra parts)
```
 PC USB-A port
      │
 [USB power meter]   ← reads V, mA, mWh for everything after it
      │
 Micro-USB DATA cable
      │
 Pico 2  (powered, flashed and talked to over this one cable)
```
- **Power:** comes over the USB cable. Nothing else is plugged into the Pico.
- **Flashing:** hold **BOOTSEL** while plugging in. The Pico shows up as an `RP2350` drive, and you copy the `.uf2` onto it. Later, `picotool load -x` does the same without pressing the button.
- **Data and test link:** the firmware uses `stdio_usb`, so the Pico shows up as a COM port. The PC test harness sends inputs and reads outputs and cycle counts over that port.
- **Energy:** the meter measures the whole board, including the USB chip. A Pico draws only about 20–40 mA, and cheap meters have a resolution of roughly 1 mWh or 10 mA. To get a usable number:
  1. Log the idle energy for a fixed time.
  2. Log the energy while running N inferences in a loop for the same time.
  3. Energy per inference = (busy − idle) / N.
- **Check:** the meter must pass the USB **data** lines through. If the Pico isn't detected through the meter but is detected without it, the meter only passes power.

### Setup B — optional: Pi 4 as SWD debugger and UART bridge (later, only if you need breakpoints)
This needs **female-to-female jumper wires** (shopping list item 5). Both boards use 3.3 V logic. **Never connect the Pi's 5 V pins.**

| Pi 4 pin (physical) | → | Pico 2 | Purpose |
|---|---|---|---|
| Pin 22 (GPIO25) | → | SWCLK (3-pin debug connector) | SWD clock |
| Pin 18 (GPIO24) | → | SWDIO (debug connector) | SWD data |
| Pin 20 (GND) | → | GND (debug connector) | Common ground |
| Pin 8 (GPIO14, TX) | → | Pin 2 (GP1, UART0 RX) | Serial Pi → Pico |
| Pin 10 (GPIO15, RX) | → | Pin 1 (GP0, UART0 TX) | Serial Pico → Pi |
| Pin 6 (GND) | → | Pin 3 (GND) | Common ground |

Before buying anything for Setup B, check the Pico's 3-pin debug connector:
- **A small white JST-SH socket:** jumper wires won't fit. You need a JST-SH-to-Dupont cable, which comes with the Debug Probe.
- **Bare pads:** these need soldering, so skip Setup B.

Setup B also needs a microSD card and power supply for the Pi 4 (items 7–8, if you don't have them). On the Pi you'd run OpenOCD with its GPIO SWD interface config.

**Recommendation:** start with Setup A and buy nothing yet. If you later need step-through debugging, the Debug Probe (£11.50) is simpler than Setup B. Keep the Pi 4 for the later Pi 4 phase.

---

## 2. Steps to Gate 0

| # | Step | Done when |
|---|---|---|
| 0 | Wire Setup A. Plug in with BOOTSEL held. | An `RP2350` drive appears with the meter in line. The meter shows ~5 V and a few mA. |
| 1 | Toolchain: VS Code **"Raspberry Pi Pico"** extension. It installs the Pico SDK, arm-none-eabi-gcc, CMake, Ninja, picotool and OpenOCD. Board = `pico2`. | It builds its own blink example. |
| 2 | Bring-up firmware `firmware/rp2350/hello/`: LED blink + `stdio_usb` printing "hello" + a cycle-counter timing of a fixed loop. Flash it via BOOTSEL. | LED blinks. The COM port prints hello and timings. `picotool info` reads the board. |
| 2b | Meter sanity check: idle vs busy-loop for 60 s each. Record the meter's resolution. | Busy minus idle gives a stable mWh difference across 3 runs. |
| 3 | `train/`: Python 3.12 venv with PyTorch (CUDA). Clone BitNetMCU and train its ternary MNIST model. Also train variants with ~70% and ~90% zero weights. | Accuracy close to BitNetMCU's reported numbers (99.45% final vs 99.52% reported; one seed). Weights exported. |
| 4 | `firmware/baselines/ref_c/`: plain-C inference with packed weights (ternary fc1). | Logits identical to the integer golden model (`harness/golden.py`) on every test image. Prediction agreement with PyTorch logged. *(Revised 11 Nov: the board does integer arithmetic, so 100% agreement with float PyTorch isn't a meaningful target. See `harness/README.md`.)* |
| 5 | `firmware/baselines/cmsis_nn/`: int8 CMSIS-NN build of the same network. | Runs, bit-exact with its own golden model, and its accuracy is logged. |
| 6 | `harness/`: Python script (pyserial) that sends test images over the COM port, reads class and cycles, compares with PyTorch, and writes a CSV to `results/` using the HANDOFF §6 metric columns. | **Gate 0:** both builds are bit-exact with their golden models, and both have their numbers logged. Energy (step 2b) to be added. |

Repo layout follows HANDOFF §10: `train/ compiler/ firmware/rp2350/ firmware/baselines/ harness/ results/`. Also update HANDOFF §3 and §9 with the answers so far:
- Raspberry Pi = Pi 4
- Owned: one Pico 2 with headers, USB meter, cable. No Debug Probe.
- Code repo = `Projects/model-as-firmware`

## Progress (10 Nov 2026)
- **Steps 0–2 done.** The board enumerates as an RP2350 and is flashed with `picotool load -f -x`, so BOOTSEL is no longer needed after the first flash. `firmware/rp2350/hello/` prints over USB (COM3). The DWT cycle counter reads 150,251 cycles for a 1 ms wait at 150 MHz, and the timing loop varies by ±108 cycles out of 700k.
  - Two of the micro-USB cables on hand are **charge-only**, and with them the Pico doesn't show up at all.
- **Step 2b blocked.** The power meter is USB-C, so it needs an adapter or a USB-C cable before it can go in line.
- **Step 3 done.** See `train/README.md`. The ternary CNN80 reaches 99.48% best and 99.45% final, against 99.52% reported by BitNetMCU. The 70% and 90% fc1-zero variants are trained and reach 99.44% and 99.50% final, so sparsity costs no accuracy. Headers are in `train/export/` and weights in `train/models/`. In this model only fc1 is ternary; the convolutions are 8-bit and fc2/fc3 are 4-bit.

## Progress (11 Nov 2026)
- **Parts ordered:** a USB-C cable for the power meter (unblocks step 2b) and a Raspberry Pi Debug Probe.
- **Steps 4–6 done; Gate 0 met except for energy.** All builds are bit-exact with their golden models on all 10,000 test images. Full table in `results/README.md`, write-up in `notes/LOG.md`.
  - Plain-C reference: 99.47% accuracy, 99.98% agreement with PyTorch, 10.21 ms per image from flash and 9.52 ms from SRAM.
  - CMSIS-NN int8, same weights: 99.44%, 18.70 ms from flash and 14.52 ms from SRAM. **14.52 ms is the speed bar for Gate 1.**
  - CMSIS-NN's int8 fc1 runs 7.4x faster from SRAM than from flash (its weights overflow the 16 KB XIP cache). The packed ternary weights fit.
- **Still to do for Gate 0:** step 2b (energy per inference), then add `energy_mj` to the results. *(Update 13 Nov: the cable and the Debug Probe arrived, but the USB meter reads only 0–4 mA in 1 mA steps for the Pico 2, which is too coarse. Step 2b needs an INA226 breakout or a Nordic PPK2.)*

## 3. Open items
- Exact Pi 4 RAM and whether you have a microSD card and power supply for it. This only matters for Setup B or the later Pi phase.
- One Pico is enough until idea F, which needs 3.

## Verification
Each step's "Done when" column is its check. Gate 0 is checked end to end with `harness/phase0_matrix.sh final`, which builds, flashes and runs every configuration. Each run must report `logits == golden 10000/10000`.
