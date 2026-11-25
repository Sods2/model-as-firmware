# Tool locations for the harness scripts (sourced by flash.sh, measure.sh, phase0_matrix.sh and train/lm/overnight.sh).
#
# Every variable below can be set in the environment first. Otherwise it is filled in from, in order:
#   1. the Raspberry Pi Pico VS Code extension's install in $PICO_SDK_HOME (default ~/.pico-sdk), at the versions the
#      results were measured with, if it is there;
#   2. whatever is on PATH.
#
#   PY              Python with requirements.txt installed  (train/.venv, then python3/python)
#   PORT            the board's USB serial port              (COM3 on Windows, /dev/ttyACM0 elsewhere)
#   PICO_SDK_HOME   the extension's folder                   (~/.pico-sdk)
#   PICOTOOL        picotool                                 (2.3.1)
#   OPENOCD         OpenOCD, Raspberry Pi fork               (0.12.0+dev); OPENOCD_SCRIPTS = its scripts folder
#   cmake, ninja and arm-none-eabi-* come from PATH; the extension's CMake 4.3.4, Ninja 1.13.2 and GCC 15.2.Rel1
#   folders are put in front of PATH when they exist.

case ${OSTYPE:-} in msys*|cygwin*|win32*) _mf_windows=1 ;; *) _mf_windows= ;; esac
PICO_SDK_HOME=${PICO_SDK_HOME:-$HOME/.pico-sdk}

# _mf_tool <name> <candidate path without .exe>...: the first candidate that exists (with or without .exe), else the
# name on PATH, else the bare name so the error message names the missing tool.
_mf_tool() {
  local name=$1 c; shift
  for c in "$@"; do
    [ -x "$c" ] && { echo "$c"; return; }
    [ -x "$c.exe" ] && { echo "$c.exe"; return; }
  done
  command -v "$name" 2>/dev/null || echo "$name"
}

for _mf_d in "$PICO_SDK_HOME/toolchain/15_2_Rel1/bin" "$PICO_SDK_HOME/ninja/v1.13.2" "$PICO_SDK_HOME/cmake/v4.3.4/bin"; do
  [ -d "$_mf_d" ] && PATH="$_mf_d:$PATH"
done
export PATH

if [ -z "${PY:-}" ]; then
  if   [ -x train/.venv/Scripts/python.exe ]; then PY=train/.venv/Scripts/python.exe
  elif [ -x train/.venv/bin/python ];         then PY=train/.venv/bin/python
  else PY=$(command -v python3 2>/dev/null || command -v python 2>/dev/null || echo python3)
  fi
fi
if [ -z "${PORT:-}" ]; then
  if [ -n "$_mf_windows" ]; then PORT=COM3; else PORT=/dev/ttyACM0; fi
fi
PICOTOOL=${PICOTOOL:-$(_mf_tool picotool "$PICO_SDK_HOME/picotool/2.3.1/picotool/picotool")}
OPENOCD=${OPENOCD:-$(_mf_tool openocd "$PICO_SDK_HOME/openocd/0.12.0+dev/openocd")}
if [ -z "${OPENOCD_SCRIPTS:-}" ] && [ -d "$PICO_SDK_HOME/openocd/0.12.0+dev/scripts" ]; then
  OPENOCD_SCRIPTS=$PICO_SDK_HOME/openocd/0.12.0+dev/scripts
fi
export PY PORT PICO_SDK_HOME PICOTOOL OPENOCD OPENOCD_SCRIPTS
unset _mf_d
