# Flashing helpers for the Pico 2, sourced by measure.sh, phase0_matrix.sh and the launcher (bash, from the repo root).
# Tool locations, PY and PORT come from harness/env.sh (override any of them in the environment).
#
#   flash <path/to/build.uf2>    loads the build and waits for the board's USB serial port to come back.
#
# FLASH=auto (default): over SWD with the Debug Probe (OpenOCD loads the matching .elf, verifies it and resets the
#                       board); if that fails (no probe, cable unplugged), falls back to picotool over USB.
# FLASH=swd:            SWD only; FLASH=usb: picotool only (-f reboots the running firmware into BOOTSEL).
. harness/env.sh
FLASH=${FLASH:-auto}

wait_port() {
  # Wait until the firmware answers '?' on $PORT. Windows can keep listing a stale COM port after a reset, so a
  # port that merely exists isn't enough.
  $PY - "$PORT" <<'PYEOF' || { echo "board did not come back on $PORT"; return 1; }
import sys, time, serial
port, end = sys.argv[1], time.time() + 30
while time.time() < end:
    try:
        with serial.Serial(port, 115200, timeout=1) as s:
            time.sleep(0.3); s.reset_input_buffer(); s.write(b'?')
            if s.readline().startswith(b'INFO'):
                sys.exit(0)
    except (serial.SerialException, OSError):
        pass
    time.sleep(1)
sys.exit(1)
PYEOF
}

flash_swd() {
  # The .elf next to the .uf2. OpenOCD writes by load address, so copy_to_ram builds work too. A plain 'reset'
  # (SYSRESETREQ) restarts the cores but leaves USB looking connected, so the port stays dead ("device not
  # functioning"); cold_reset resets the whole chip, which drops USB and makes it enumerate again.
  local elf=${1%.uf2}.elf
  "$OPENOCD" ${OPENOCD_SCRIPTS:+-s "$OPENOCD_SCRIPTS"} -f interface/cmsis-dap.cfg -f target/rp2350.cfg -c "adapter speed 5000" \
      -c "program {$elf} verify" -c "cold_reset" -c "shutdown" >"${TMPDIR:-/tmp}/openocd_flash.log" 2>&1 || return 1
}

flash_usb() {
  # -f reboots the running firmware into BOOTSEL; Windows is sometimes too slow to see it, so retry without -f
  "$PICOTOOL" load -f -x "$1" >/dev/null 2>&1 && return 0
  for _ in 1 2 3 4 5; do sleep 2; "$PICOTOOL" load -x "$1" >/dev/null 2>&1 && return 0; done
  return 1
}

flash() {
  case $FLASH in
    swd)  flash_swd "$1" || { echo "SWD flashing $1 failed (log: ${TMPDIR:-/tmp}/openocd_flash.log)"; return 1; } ;;
    usb)  flash_usb "$1" || { echo "flashing $1 failed"; return 1; } ;;
    auto) flash_swd "$1" || { echo "  (SWD flash failed, using picotool)"; flash_usb "$1"; } \
            || { echo "flashing $1 failed"; return 1; } ;;
    *)    echo "unknown FLASH=$FLASH (auto|swd|usb)"; return 1 ;;
  esac
  wait_port
}
