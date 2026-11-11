# Pico 2 toolchain on this PC

*Last checked 11 Nov 2026.*

## Where things are

Everything comes from the VS Code "Raspberry Pi Pico" extension, under `%USERPROFILE%\.pico-sdk\`:

| Tool | Path |
|---|---|
| Pico SDK | `sdk/2.3.1` |
| GCC | `toolchain/15_2_Rel1/bin/arm-none-eabi-gcc.exe` (also `arm-none-eabi-size.exe`) |
| CMake | `cmake/v4.3.4/bin/cmake.exe` |
| Ninja | `ninja/v1.13.2/ninja.exe` |
| picotool | `picotool/2.3.1/picotool/picotool.exe` |
| OpenOCD | `openocd/` (for when the Debug Probe arrives) |

None of these are on PATH by default. The PC also has an old 32-bit MinGW gcc 9.2 at `C:\MinGW\bin\gcc.exe`, which is fine for small host test programs but can't build DLLs for 64-bit Python.

## Build from Git Bash (without VS Code)

```bash
P=~/.pico-sdk
export PATH="$P/cmake/v4.3.4/bin:$P/ninja/v1.13.2:$P/toolchain/15_2_Rel1/bin:$PATH"
cmake -S firmware/baselines/ref_c -B firmware/baselines/ref_c/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DMODEL=tern80
ninja -C firmware/baselines/ref_c/build
arm-none-eabi-size firmware/baselines/ref_c/build/ref_c.elf
```

- A Release build gives `-O3 -g -mcpu=cortex-m33 -march=armv8-m.main+fp+dsp`, Secure ARM mode, 150 MHz.
- A project outside `firmware/rp2350/hello/` can reuse that folder's `pico_sdk_import.cmake` by path (see `ref_c/CMakeLists.txt`).

## Flashing over USB (no Debug Probe)

```bash
PT=~/.pico-sdk/picotool/2.3.1/picotool/picotool.exe
"$PT" load -f -x build/ref_c.uf2
```

- `-f` asks the running firmware, over its USB stdio interface, to reboot into BOOTSEL. This only works if the running firmware uses `stdio_usb`.
- **Quirk:** sometimes `-f` reports "no accessible RP-series devices in BOOTSEL mode were found" even though the reboot worked, because Windows enumerates the BOOTSEL device slowly. Wait about 3 s and run `load -x` again without `-f`. `picotool info` shows whether the board is in BOOTSEL mode.
- After loading, COM3 takes 1–2 s to come back.
- Two of the micro-USB cables on hand are **charge-only**. With those the board doesn't show up at all.

## Talking to the board

- `stdio_usb` gives COM3 (VID:PID 2E8A:0009). The baud rate is ignored.
- Binary input through `getchar_timeout_us` works: there's no CRLF translation on input. Output `\n` comes out as `\r\n`.
- The harness protocol is in [firmware/baselines/ref_c/pico_main.c](../firmware/baselines/ref_c/pico_main.c): `?` returns info, and `I` + 256 bytes returns `R pred cycles logits…`. A full 10k-image run takes about 3 minutes, mostly USB round trips.
- Timing uses the DWT cycle counter (`DEMCR.TRCENA`, `DWT_CTRL.CYCCNTENA`, `DWT_CYCCNT` at `0xE0001004`). It counts at the system clock (150 MHz).

## Python environment

`train/.venv` (Python 3.12) has torch, torchvision and BitNetMCU's requirements, plus `pyserial` (added 11 Nov 2026) for the harness.

- **Gotcha:** Python's `open()` on this PC defaults to cp1252, not UTF-8. Scripts that rewrite docs must pass `encoding='utf-8'`, or characters like `–` and `±` get corrupted.

## Measuring flash traffic (open question)

- **Correction (12 Nov 2026):** the RP2350 *has* XIP cache counters: `xip_ctrl_hw->ctr_hit` and `ctr_acc` (`hardware/structs/xip.h`, registers `XIP_CTR_HIT` / `XIP_CTR_ACC` at `XIP_CTRL_BASE + 0x0c / 0x10`; write to clear). The 11 Nov check that found "none" looked in the wrong header. Misses = acc − hit; times the 8-byte line size, that gives about the bytes fetched through the cache. Uncached and DMA/stream reads are not counted as hits.
- The XIP stream FIFO (`stream_addr`, `stream_ctr`, read via `xip_aux_hw->stream` with `DREQ_XIP_STREAM`) is the fast path for DMA from flash (FINDINGS F21).
- Options still to check: work out bytes read from the weight layout and the 16 KB cache (analytically); the QMI or bus-performance counters in the RP2350 datasheet; or compare the same build with `RUN_FROM_RAM` on and off, as the CMSIS-NN fc1 experiment already did (524k vs 71k cycles).

## Gotcha: `arm-none-eabi-size` counts code in SRAM as text

- Functions placed in SRAM with `__not_in_flash_func` / `__time_critical_func` land in `.data`, which is an executable section. Berkeley-format `size` therefore reports them as **text** and `data` as 0, even though they occupy SRAM at run time (found 11 Nov 2026: an 86 KB fc1 showed up as 9.7 KB of RAM).
- Use `arm-none-eabi-size -A` and sum the sections at `0x20000000`–`0x2007FFFF` (main SRAM; the stacks are in the scratch banks above). `harness/run.py` does this since 11 Nov 2026.

## Gotcha: only one thing can use the board, and Git Bash `ps` doesn't see Windows processes

- COM3 can only be opened by one process. A second `run.py` (or the live-view `bridge.py`) fails with `could not open port 'COM3': PermissionError(13, 'Access is denied.')`.
- Two `measure.sh` runs at once also collide on the build folders (`ninja: error: failed recompaction: Permission denied`).
- To see what holds the board, use PowerShell, not Git Bash `ps`, which only lists MSYS processes:
  `powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'run.py|bridge.py' } | Select ProcessId,CommandLine"`
- Happened on 11 Nov 2026: a `measure.sh` started with `&` from a tool shell looked dead to `ps` but kept running, flashing and measuring. Start long runs as real background jobs and check them with the command above before starting another.
- Don't use that check inside a shell loop that waits for `run.py` to finish: the loop's own command line contains the text `run.py`, so it matches itself and never ends. This happened on 11 Nov 2026, and a watcher loop hung for hours. Exclude the current process (`$_.ProcessId -ne $PID` doesn't help, because the match is the parent bash), match `python.exe` with `$_.Name -eq 'python.exe'`, or just wait for the CSV rows.
