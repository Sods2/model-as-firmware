# Pico 2 toolchain: notes and gotchas

How the toolchain behaved on the machine the results were measured on (Windows 11, Git Bash, the VS Code "Raspberry Pi Pico" extension), and the quirks found on the way. For setting up from scratch, start with [docs/SETUP.md](../docs/SETUP.md); this page is the detail behind it.

## Where the extension puts things

The VS Code "Raspberry Pi Pico" extension installs everything under `~/.pico-sdk/` (`%USERPROFILE%\.pico-sdk\` on Windows). [harness/env.sh](../harness/env.sh) looks there first, then on PATH.

| Tool | Path under `~/.pico-sdk/` |
|---|---|
| Pico SDK | `sdk/2.3.1` |
| GCC | `toolchain/15_2_Rel1/bin/arm-none-eabi-gcc` (also `arm-none-eabi-size`) |
| CMake | `cmake/v4.3.4/bin/cmake` |
| Ninja | `ninja/v1.13.2/ninja` |
| picotool | `picotool/2.3.1/picotool/picotool` |
| OpenOCD | `openocd/0.12.0+dev/openocd` (Raspberry Pi fork, used with the Debug Probe) |

None of these are on PATH by default; `harness/env.sh` puts them there for the scripts.

## Build by hand (without VS Code)

```bash
. harness/env.sh      # toolchain on PATH (or use your own, with PICO_SDK_PATH set)
cmake -S firmware/baselines/ref_c -B firmware/baselines/ref_c/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DMODEL=tern80
ninja -C firmware/baselines/ref_c/build
arm-none-eabi-size firmware/baselines/ref_c/build/ref_c.elf
```

- A Release build gives `-O3 -g -mcpu=cortex-m33 -march=armv8-m.main+fp+dsp`, Secure ARM mode, 150 MHz.
- Every project includes `firmware/rp2350/hello/pico_sdk_import.cmake`, so a plain `PICO_SDK_PATH` works without the extension.

## Debug Probe (SWD)

The Raspberry Pi Debug Probe enumerates as CMSIS-DAP v2 (VID:PID 2E8A:000C), with its own UART bridge as a second serial port. Its **D** port goes to the Pico 2's 3-pin DEBUG connector.

```bash
"$OPENOCD" ${OPENOCD_SCRIPTS:+-s "$OPENOCD_SCRIPTS"} -f interface/cmsis-dap.cfg -f target/rp2350.cfg \
    -c "adapter speed 5000" -c "init; targets; exit"
```

- This finds `rp2350.cm0` and `rp2350.cm1` (Cortex-M33 r1p0, 8 breakpoints, 4 watchpoints each).
- **Update the probe firmware** to debugprobe v2.3.1 or later (`debugprobe.uf2` from github.com/raspberrypi/debugprobe releases). Version 1.0.1 made OpenOCD warn and fall back to a slow workaround.
- To update: hold the BOOTSEL button inside the probe's case while plugging in its USB. It mounts as `RPI-RP2` (`INFO_UF2.TXT` says "Model: Raspberry Pi RP2", not "RP2040"); copy the `.uf2` onto it. Don't confuse it with the Pico 2's own BOOTSEL drive, which is labelled `RP2350`.
- Unplug the SWD cable for energy measurements: its ground is a second return path around any USB-side meter.

### Flashing over SWD (the default)

`harness/flash.sh` (sourced by `measure.sh`, `phase0_matrix.sh` and the launcher) does this, with `picotool` as the fallback:

```bash
"$OPENOCD" ${OPENOCD_SCRIPTS:+-s "$OPENOCD_SCRIPTS"} -f interface/cmsis-dap.cfg -f target/rp2350.cfg \
    -c "adapter speed 5000" -c "program {build/mafc.elf} verify" -c "cold_reset" -c "shutdown"
```

- Load the `.elf`, not the `.uf2`. OpenOCD writes by load address, so `copy_to_ram` builds work too.
- **Gotcha: use `cold_reset`, not `reset`.** `program ... verify reset exit` resets the cores with SYSRESETREQ, but the USB controller never drops off the bus. Windows keeps the COM port listed, and opening it fails with `PermissionError(13, 'A device attached to the system is not functioning.')`. `cold_reset` (defined in `target/rp2350.cfg`) resets the whole chip except the debug port, so USB enumerates again within about 2 s.
- For the same reason, `wait_port` in `flash.sh` waits for the firmware to answer `?` rather than for the port to appear.
- About 3–4 s per build from the start of flashing until the board answers, and no BOOTSEL step.

## Flashing over USB (no Debug Probe)

```bash
"$PICOTOOL" load -f -x build/ref_c.uf2
```

- `-f` asks the running firmware, over its USB stdio interface, to reboot into BOOTSEL. This only works if the running firmware uses `stdio_usb` (all of this repo's firmware does).
- The very first flash of a new board: hold BOOTSEL while plugging it in, then copy the `.uf2` onto the `RP2350` drive (or `picotool load -x` without `-f`).
- **Quirk:** sometimes `-f` reports "no accessible RP-series devices in BOOTSEL mode were found" even though the reboot worked, because Windows enumerates the BOOTSEL device slowly. Wait about 3 s and run `load -x` again without `-f`. `flash.sh` retries this way.
- After loading, the serial port takes 1–2 s to come back.
- Many micro-USB cables are **charge-only**. With those the board doesn't show up at all.

## Talking to the board

- `stdio_usb` gives a serial port (VID:PID 2E8A:0009): a `COM` port on Windows, `/dev/ttyACM*` on Linux, `/dev/cu.usbmodem*` on macOS. The baud rate is ignored.
- Binary input through `getchar_timeout_us` works: there's no CRLF translation on input. Output `\n` comes out as `\r\n`.
- The harness protocol is in [firmware/baselines/ref_c/pico_main.c](../firmware/baselines/ref_c/pico_main.c): `?` returns info, and `I` + 256 bytes returns `R pred cycles logits…`. A full 10k-image run takes about 3 minutes, mostly USB round trips.
- Timing uses the DWT cycle counter (`DEMCR.TRCENA`, `DWT_CTRL.CYCCNTENA`, `DWT_CYCCNT` at `0xE0001004`). It counts at the system clock (150 MHz).

## Python

- **Gotcha (Windows):** Python's `open()` defaults to the system code page (cp1252), not UTF-8. Scripts that rewrite docs must pass `encoding='utf-8'`, or characters like `–` and `±` get corrupted. Set `PYTHONIOENCODING=utf-8` when printing them.

## Measuring flash traffic

- The RP2350 has XIP cache counters: `xip_ctrl_hw->ctr_hit` and `ctr_acc` (`hardware/structs/xip.h`, registers `XIP_CTR_HIT` / `XIP_CTR_ACC` at `XIP_CTRL_BASE + 0x0c / 0x10`; write to clear). Misses = acc − hit; times the 8-byte line size, that gives about the bytes fetched through the cache. Uncached and DMA/stream reads are not counted as hits.
- `pico_main.c` clears and sums the two counters around every `ref_infer()` (command `X`); `run.py` turns them into `flash_read_kb` (FINDINGS F22).
- The XIP stream FIFO (`stream_addr`, `stream_ctr`, read via `xip_aux_hw->stream` with `DREQ_XIP_STREAM`) is the fast path for DMA from flash (FINDINGS F21).

## Gotcha: `arm-none-eabi-size` counts code in SRAM as text

- Functions placed in SRAM with `__not_in_flash_func` / `__time_critical_func` land in `.data`, which is an executable section. Berkeley-format `size` therefore reports them as **text** and `data` as 0, even though they occupy SRAM at run time (an 86 KB fc1 showed up as 9.7 KB of RAM).
- Use `arm-none-eabi-size -A` and sum the sections at `0x20000000`–`0x2007FFFF` (main SRAM; the stacks are in the scratch banks above). `harness/run.py` does this.

## Gotcha: measure.sh hides compiler errors

- `measure.sh` sends cmake and ninja output to `/dev/null`, so a build error just stops the batch (`set -e`) with nothing printed after the cmake lines. If a batch ends early with no `board:` line, rebuild the failing folder by hand (`ninja -C <build dir>`) to see the error.

## Gotcha: only one program can use the board

- The serial port can only be opened by one process. A second `run.py` (or the live view's `bridge.py`) fails with `PermissionError(13, 'Access is denied.')` on Windows, or "device busy" elsewhere. The bridge and the launcher check for this and say what holds the board.
- Two `measure.sh` runs at once also collide on the build folders (`ninja: error: failed recompaction: Permission denied`).
- On Windows, Git Bash `ps` only lists MSYS processes. To see which Python process holds the board, use PowerShell:
  `powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -match 'run.py|bridge.py' } | Select ProcessId,CommandLine"`
- Matching on the command line alone can match the shell that runs the check, so filter on the process name as above.
