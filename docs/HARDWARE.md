# Hardware

What every result in this repo was measured on, and what you need to repeat it. All parts are plug-in: no soldering.

## What was used

| Part | Details | Needed for |
|---|---|---|
| **Raspberry Pi Pico 2**, with headers | RP2350: two Cortex-M33 cores at **150 MHz** (the SDK default, not overclocked), 520 KB SRAM, 4 MB flash. The "with headers" version, so nothing needs soldering. | every board result |
| **Raspberry Pi Debug Probe** | CMSIS-DAP over USB, with its 3-pin JST SWD cable. Firmware updated to debugprobe v2.3.1 (see [notes/pico-toolchain.md](../notes/pico-toolchain.md)). | fast, button-free flashing over SWD. Optional: everything also flashes over the Pico's own USB with picotool. |
| **Micro-USB *data* cable** for the Pico | Power, flashing (fallback) and the test link (USB serial) all go over this one cable. Many cheap cables are charge-only; with those the Pico doesn't show up at all. | every board result |
| **A PC** | Windows 11 with Git Bash. Training used an NVIDIA RTX 3060 Ti (8 GB). | building, the test harness, training |

The PC only matters for build time and training time: the board results don't depend on it. Training on a CPU works but is much slower; see [REPRODUCE.md](REPRODUCE.md) for the training times on the GPU above.

Also on hand but **not used for any result**: a Raspberry Pi 4 (for a later phase) and an inline USB power meter, which turned out too coarse for the Pico 2 (it reads 0–4 mA in 1 mA steps). This is why energy isn't measured yet.

## Wiring

```
PC USB port ──── Micro-USB data cable ──── Pico 2          (power + USB serial test link + picotool fallback)
PC USB port ──── Debug Probe ──── 3-pin JST cable ──── Pico 2 "DEBUG" connector   (SWD flashing)
```

- Plug the probe's **D** port (SWD) into the Pico 2's 3-pin **DEBUG** connector (a small white JST-SH socket labelled DEBUG). The probe's **U** port (UART) isn't used. If your board has three bare DEBUG pads instead of a socket, the cable won't fit without soldering: skip the probe and flash over USB.
- Nothing else is connected to the Pico.
- **First flash of a new board without the probe:** hold the **BOOTSEL** button while plugging in USB. The Pico appears as a drive called `RP2350`; copy a `.uf2` onto it. After that, every build in this repo answers picotool's reboot request, so the button isn't needed again.
- On the PC the Pico shows up as a serial port: `COM<n>` on Windows (it was `COM3` here), `/dev/ttyACM<n>` on Linux, `/dev/cu.usbmodem<n>` on macOS. Every tool takes it as `--port` or `PORT=`.
- **For energy measurements** (not done yet), unplug the SWD cable: its ground is a second return path around any meter on the USB side.

## Suggested extras (not needed to reproduce the results)

| Part | What it would add |
|---|---|
| **INA226 current-sensor breakout**, or a **Nordic PPK2** | Energy per inference. The tooling is ready (`harness/measure.sh energy`, [harness/README.md](../harness/README.md)); a typical USB meter is too coarse for the Pico 2's 20–40 mA. |
| **Two more Pico 2 boards**, a breadboard and jumper wires | Idea F: one model split across three boards ([FEASIBILITY-STUDY.md](../FEASIBILITY-STUDY.md)). Not tried. |
| **Pimoroni Pico Plus 2** (RP2350 with 8 MB PSRAM, 16 MB flash) | Idea G: models larger than SRAM, with an SRAM cache in front of slower memory. Not tried. |
| **An ESP32-S3 board** (e.g. N16R8: 16 MB flash, 8 MB PSRAM) | A second chip, to check that the compiler's gains carry over. Not tried. |
| **Raspberry Pi 4 or 5**, microSD card and power supply | The planned bare-metal Pi phase (Phase 4). Not started. |
